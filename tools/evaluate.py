#!/usr/bin/env python3
"""Fire dev prompts at :8000, compare to reference, report timing + mismatches.

Requires a running solution container (see start_inference.py). Tokenizes
both sides at $MODEL_PATH (host mount). Exits non-zero on any mismatch.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

DEV_PROMPTS = Path("/workspace/data/agent/dev_prompts.jsonl")
DEV_REFERENCE = Path("/workspace/data/agent/dev_reference.jsonl")
BASE_URL = "http://localhost:8000"
MODEL_PATH = os.environ["MODEL_PATH"]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


async def fire_all(prompts: list[dict]) -> tuple[list[dict], float]:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=900.0) as client:
        async def one(p: dict) -> dict:
            r = await client.post("/v1/completions", json=p)
            r.raise_for_status()
            return r.json()
        t0 = time.perf_counter()
        responses = await asyncio.gather(*(one(p) for p in prompts))
        return responses, time.perf_counter() - t0


def probe_ready() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5.0)
    except Exception as e:
        print(f"server not reachable at {BASE_URL}: {e}", file=sys.stderr)
        print("Run `python3 /workspace/start_inference.py` first.", file=sys.stderr)
        return False
    if r.status_code != 200 or r.json().get("status") != "ready":
        print(f"server not ready: {r.status_code} {r.text}", file=sys.stderr)
        return False
    return True


def main() -> int:
    if not DEV_PROMPTS.exists() or not DEV_REFERENCE.exists():
        print("missing /workspace/data/agent/dev_{prompts,reference}.jsonl",
              file=sys.stderr)
        return 2
    if not probe_ready():
        return 2

    prompts = load_jsonl(DEV_PROMPTS)
    reference = load_jsonl(DEV_REFERENCE)
    print(f"firing {len(prompts)} prompts ...", file=sys.stderr, flush=True)
    responses, wall = asyncio.run(fire_all(prompts))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    ref_by_id = {r["id"]: r["completion"] for r in reference}
    resp_by_id = {r["id"]: r["completion"] for r in responses}

    mismatches = 0
    first_bad: str | None = None
    for pid, ref_text in ref_by_id.items():
        got = resp_by_id.get(pid)
        if got is None:
            mismatches += 1
            first_bad = first_bad or pid
            continue
        if tok.encode(ref_text, add_special_tokens=False) != tok.encode(
            got, add_special_tokens=False
        ):
            mismatches += 1
            first_bad = first_bad or pid

    print(
        f"wall_time={wall:.3f}s  mismatches={mismatches}/{len(prompts)}"
        f"  first_bad={first_bad}"
    )
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
