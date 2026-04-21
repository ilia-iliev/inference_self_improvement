#!/usr/bin/env python3
"""
Orchestrator for the online eval loop.

Subcommands:
  submit          Build solution image, run eval, verify, score, write
                  /solution/last_result.json. Promote if score >= 1.05 and match.
                  First run auto-initializes baseline timing.
  dev             Build solution image, run eval on first N dev prompts,
                  verify against dev_reference. No baseline comparison.

Runs on the host via: uv run --with httpx --with transformers --with sentencepiece \
    python3 judge/submit.py <subcommand>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx  # noqa: F401  # imported for side effect of failing fast if missing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import load_prompts, run_eval  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
JUDGE_DIR = REPO_ROOT / "judge"
SOLUTION_DIR = REPO_ROOT / "solution"
BASELINE_DIR = JUDGE_DIR / "baseline"
BASELINE_JSON = JUDGE_DIR / "baseline.json"
LAST_RESULT_JSON = SOLUTION_DIR / "last_result.json"
EVAL_PROMPTS = DATA_DIR / "judge" / "eval_prompts.jsonl"
EVAL_REFERENCE = DATA_DIR / "judge" / "eval_reference.jsonl"
DEV_PROMPTS = DATA_DIR / "agent" / "dev_prompts.jsonl"
DEV_REFERENCE = DATA_DIR / "agent" / "dev_reference.jsonl"

BASELINE_TAG = "gemma4-baseline:current"
SOLUTION_TAG = "gemma4-solution:latest"
CONTAINER_NAME = "gemma4-challenge-server"
HOST_PORT = 8000
BASE_URL = f"http://localhost:{HOST_PORT}"

PROMOTION_THRESHOLD = 1.05

HF_CACHE = os.environ.get("HF_CACHE_DIR") or (Path.home() / ".cache" / "huggingface").as_posix()
MODEL_PATH = os.environ.get("MODEL_PATH", "google/gemma-3-1b-it")
TEXT_ONLY = os.environ.get("TEXT_ONLY", "1") == "1"
# Host-side tokenizer: resolve MODEL_PATH to the HF cache snapshot dir so we
# don't need to re-fetch or mount the container-side path.
if MODEL_PATH.startswith("/"):
    HOST_TOKENIZER_PATH = MODEL_PATH
else:
    _cache_root = Path(HF_CACHE) / "hub" / f"models--{MODEL_PATH.replace('/', '--')}" / "snapshots"
    _snaps = sorted(_cache_root.iterdir()) if _cache_root.is_dir() else []
    if not _snaps:
        sys.exit(f"ERROR: no HF cache snapshot for {MODEL_PATH} at {_cache_root}")
    HOST_TOKENIZER_PATH = str(_snaps[-1])


# ---------- shell / docker primitives ----------

def sh(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=True, **kwargs)


def build_image(build_ctx: Path, tag: str) -> None:
    sh(["docker", "build", "-t", tag, str(build_ctx)])


def stop_container(name: str) -> None:
    subprocess.run(
        ["docker", "stop", name], check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run_container(tag: str, name: str) -> None:
    stop_container(name)
    sh([
        "docker", "run", "-d", "--rm",
        "--gpus", "all",
        "-e", f"MODEL_PATH={MODEL_PATH}",
        "-e", f"TEXT_ONLY={'1' if TEXT_ONLY else '0'}",
        "-e", "HF_HUB_OFFLINE=1",
        "-v", f"{HF_CACHE}:/root/.cache/huggingface:ro",
        "-v", f"{DATA_DIR}:/data:ro",
        "-p", f"{HOST_PORT}:8000",
        "--name", name,
        tag,
    ])


# ---------- measurement + verification ----------

async def measure_timing(prompts: list[dict]) -> dict:
    return await run_eval(BASE_URL, prompts, ready_timeout_s=300.0, request_timeout_s=900.0)


def verify(responses: list[dict], reference: list[dict]) -> tuple[bool, int, str | None]:
    """Tokenize both sides and compare IDs. Returns (all_match, num_mismatched, first_mismatch_id)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HOST_TOKENIZER_PATH)
    ref_by_id = {r["id"]: r["completion"] for r in reference}
    resp_by_id = {r["id"]: r["completion"] for r in responses}
    mismatched = 0
    first_bad: str | None = None
    for pid, ref_text in ref_by_id.items():
        if pid not in resp_by_id:
            mismatched += 1
            first_bad = first_bad or pid
            continue
        if tok.encode(ref_text, add_special_tokens=False) != tok.encode(resp_by_id[pid], add_special_tokens=False):
            mismatched += 1
            first_bad = first_bad or pid
    return mismatched == 0, mismatched, first_bad


# ---------- eval flow ----------

def eval_image(tag: str, prompts: list[dict]) -> dict:
    """Build+run assumed done. Start container, measure, stop. Return eval result dict."""
    run_container(tag, CONTAINER_NAME)
    try:
        return asyncio.run(measure_timing(prompts))
    finally:
        stop_container(CONTAINER_NAME)


def require_eval_inputs() -> None:
    for p in (EVAL_PROMPTS, EVAL_REFERENCE):
        if not p.exists():
            sys.exit(f"ERROR: {p} missing. Run ./run_challenge.sh data first.")


# ---------- subcommands ----------

def ensure_baseline() -> dict:
    """Return the cached baseline dict; build + measure it on first call."""
    if BASELINE_JSON.exists():
        return json.loads(BASELINE_JSON.read_text())
    print("No baseline found. Building baseline image...", file=sys.stderr)
    build_image(BASELINE_DIR, BASELINE_TAG)
    print("Running baseline eval...", file=sys.stderr)
    result = eval_image(BASELINE_TAG, load_prompts(EVAL_PROMPTS))
    if not result.get("ready"):
        sys.exit(f"ERROR: baseline never became ready: {result.get('error')}")
    reference = load_prompts(EVAL_REFERENCE)
    all_match, mismatched, first_bad = verify(result["responses"], reference)
    if not all_match:
        sys.exit(f"ERROR: baseline outputs diverge from reference ({mismatched} mismatched, first={first_bad})")
    baseline = {
        "image_tag": BASELINE_TAG,
        "timing_seconds": result["wall_time_seconds"],
        "num_prompts": len(result["responses"]),
    }
    BASELINE_JSON.write_text(json.dumps(baseline, indent=2))
    print(f"baseline.json written: timing={result['wall_time_seconds']:.2f}s", file=sys.stderr)
    return baseline


def cmd_submit() -> int:
    require_eval_inputs()
    baseline = ensure_baseline()
    print("Building solution image...", file=sys.stderr)
    build_image(SOLUTION_DIR, SOLUTION_TAG)
    print("Running solution eval...", file=sys.stderr)
    result = eval_image(SOLUTION_TAG, load_prompts(EVAL_PROMPTS))
    payload: dict = {
        "agent_time": None,
        "baseline_time": baseline["timing_seconds"],
        "num_mismatched": None,
        "first_mismatch_id": None,
        "score": 0.0,
        "promoted": False,
        "error": None,
    }
    if not result.get("ready"):
        payload["error"] = f"not ready: {result.get('error')}"
        LAST_RESULT_JSON.write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return 1
    payload["agent_time"] = result["wall_time_seconds"]
    reference = load_prompts(EVAL_REFERENCE)
    all_match, mismatched, first_bad = verify(result["responses"], reference)
    payload["num_mismatched"] = mismatched
    payload["first_mismatch_id"] = first_bad
    if not all_match:
        payload["error"] = "token mismatch"
    elif payload["agent_time"] >= payload["baseline_time"]:
        payload["error"] = "slower than baseline"
    else:
        payload["score"] = payload["baseline_time"] / payload["agent_time"]
        if payload["score"] >= PROMOTION_THRESHOLD:
            # Promote: retag solution image as the new baseline, re-measure.
            print("Promoting solution → baseline, re-measuring...", file=sys.stderr)
            sh(["docker", "tag", SOLUTION_TAG, BASELINE_TAG])
            new_result = eval_image(BASELINE_TAG, load_prompts(EVAL_PROMPTS))
            new_match, new_mis, _ = verify(new_result["responses"], reference)
            if not new_match:
                payload["error"] = "promotion re-measure failed verification"
            else:
                BASELINE_JSON.write_text(json.dumps({
                    "image_tag": BASELINE_TAG,
                    "timing_seconds": new_result["wall_time_seconds"],
                    "num_prompts": len(new_result["responses"]),
                }, indent=2))
                payload["promoted"] = True
    LAST_RESULT_JSON.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0 if payload["error"] is None else 1


def cmd_dev(n: int) -> int:
    for p in (DEV_PROMPTS, DEV_REFERENCE):
        if not p.exists():
            sys.exit(f"ERROR: {p} missing. Run ./run_challenge.sh data first.")
    prompts = load_prompts(DEV_PROMPTS)[:n]
    reference = load_prompts(DEV_REFERENCE)[:n]
    print(f"Building solution image...", file=sys.stderr)
    build_image(SOLUTION_DIR, SOLUTION_TAG)
    print(f"Running dev eval on {len(prompts)} prompts...", file=sys.stderr)
    result = eval_image(SOLUTION_TAG, prompts)
    if not result.get("ready"):
        print(json.dumps({"error": f"not ready: {result.get('error')}"}, indent=2))
        return 1
    _, mismatched, first_bad = verify(result["responses"], reference)
    payload = {
        "wall_time_seconds": result["wall_time_seconds"],
        "num_prompts": len(prompts),
        "num_mismatched": mismatched,
        "first_mismatch_id": first_bad,
    }
    print(json.dumps(payload, indent=2))
    return 0 if mismatched == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="cmd", required=True)
    subs.add_parser("submit")
    dev_p = subs.add_parser("dev")
    dev_p.add_argument("-n", type=int, default=10, help="number of dev prompts (default 10)")
    args = p.parse_args()
    if args.cmd == "dev":
        return cmd_dev(args.n)
    return cmd_submit()


if __name__ == "__main__":
    sys.exit(main())
