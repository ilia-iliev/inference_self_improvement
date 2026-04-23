#!/usr/bin/env python3
"""
Build the solution image, run eval, verify, score, promote if score >= 1.05
and all tokens match. First run auto-initializes baseline timing.

Runs on the host via: uv run --with httpx --with transformers --with sentencepiece \
    python3 judge/submit.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
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

BASELINE_TAG = "gemma4-baseline:current"
SOLUTION_TAG = "gemma4-solution:latest"
CONTAINER_NAME = "gemma4-challenge-server"
HOST_PORT = 8000
BASE_URL = f"http://localhost:{HOST_PORT}"

PROMOTION_THRESHOLD = 1.05

MODEL_PATH = os.environ["MODEL_PATH"]
HOST_TOKENIZER_PATH = MODEL_PATH


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
    submit_gpu = os.environ.get("SUBMIT_GPU", "1")
    sh([
        "docker", "run", "-d", "--rm",
        "--gpus", f"device={submit_gpu}",
        "-e", f"MODEL_PATH={MODEL_PATH}",
        "-e", "HF_HUB_OFFLINE=1",
        "-v", "/mnt/hdd2:/mnt/hdd2:ro",
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


def ensure_solution() -> None:
    """Seed solution/ from judge/baseline/ on first submit (no Dockerfile present)."""
    if (SOLUTION_DIR / "Dockerfile").exists():
        return
    print("solution/ has no Dockerfile — seeding from judge/baseline/...", file=sys.stderr)
    SOLUTION_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("server.py", "_common.py", "Dockerfile"):
        shutil.copy2(BASELINE_DIR / name, SOLUTION_DIR / name)


def cmd_submit() -> int:
    require_eval_inputs()
    ensure_solution()
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


if __name__ == "__main__":
    sys.exit(cmd_submit())
