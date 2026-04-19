#!/usr/bin/env python3
"""
Judge for the Gemma 4 2B inference throughput challenge.

Inputs:
  /data/eval_reference.jsonl    — HF greedy reference (ground truth)
  /judge/baseline_timing.json   — baseline wall-clock
  /solution/outputs.jsonl       — agent outputs
  /solution/timing.json         — agent wall-clock (written by /solution/run.py)

Output: /judge/score.json

Scoring:
  any mismatch            → 0
  agent_time >= baseline  → 0
  else                    → baseline_time / agent_time
"""

import json
import sys
from pathlib import Path

from transformers import AutoProcessor

MODEL_PATH = "/models/gemma4-2b"
EVAL_REFERENCE = Path("/data/eval_reference.jsonl")
BASELINE_TIMING = Path("/judge/baseline_timing.json")
SOLUTION_OUTPUTS = Path("/solution/outputs.jsonl")
SOLUTION_TIMING = Path("/solution/timing.json")
SCORE_OUTPUT = Path("/judge/score.json")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def verify(references, agent_outputs, processor):
    ref_by_id = {r["id"]: r for r in references}
    agent_by_id = {a["id"]: a for a in agent_outputs}
    tok = processor.tokenizer

    details, all_match = [], True
    for pid, ref in ref_by_id.items():
        if pid not in agent_by_id:
            all_match = False
            details.append({"id": pid, "match": False, "reason": "missing from agent output"})
            continue
        ref_ids = tok.encode(ref["completion"], add_special_tokens=False)
        agent_ids = tok.encode(agent_by_id[pid]["completion"], add_special_tokens=False)
        if ref_ids == agent_ids:
            details.append({"id": pid, "match": True, "num_tokens": len(ref_ids)})
        else:
            all_match = False
            diverge = next(
                (j for j in range(min(len(ref_ids), len(agent_ids))) if ref_ids[j] != agent_ids[j]),
                min(len(ref_ids), len(agent_ids)),
            )
            details.append({
                "id": pid, "match": False, "reason": "token mismatch",
                "ref_len": len(ref_ids), "agent_len": len(agent_ids),
                "first_divergence": diverge,
                "ref_snippet": ref["completion"][:200],
                "agent_snippet": agent_by_id[pid]["completion"][:200],
            })
    return all_match, details


def require(path, hint):
    if not path.exists():
        print(f"ERROR: {path} not found. {hint}")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  Gemma 4 2B Inference Throughput Challenge — Judge")
    print("=" * 60)

    require(EVAL_REFERENCE, "Run: ./run_challenge.sh reference")
    require(BASELINE_TIMING, "Run: ./run_challenge.sh baseline")
    require(SOLUTION_OUTPUTS, "Run: ./run_challenge.sh solution")
    require(SOLUTION_TIMING, "Re-run the solution so it writes timing.json.")

    references = load_jsonl(EVAL_REFERENCE)
    agent_outputs = load_jsonl(SOLUTION_OUTPUTS)
    baseline_time = json.loads(BASELINE_TIMING.read_text())["wall_time_seconds"]
    agent_time = json.loads(SOLUTION_TIMING.read_text())["wall_time_seconds"]
    print(f"Loaded {len(references)} references, {len(agent_outputs)} agent outputs")
    print(f"Baseline time: {baseline_time:.2f}s   Agent time: {agent_time:.2f}s")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    all_match, details = verify(references, agent_outputs, processor)
    matched = sum(1 for d in details if d["match"])
    mismatched = len(details) - matched
    print(f"Verification: {matched} matched, {mismatched} mismatched")

    if not all_match:
        score, reason = 0.0, f"Output mismatch: {mismatched} of {len(details)} differ from reference"
        for d in details:
            if not d["match"]:
                print(f"  {d['id']}: {d.get('reason')}")
    elif agent_time >= baseline_time:
        score, reason = 0.0, f"Agent ({agent_time:.2f}s) not faster than baseline ({baseline_time:.2f}s)"
    else:
        score = baseline_time / agent_time
        reason = f"Agent is {score:.2f}x faster than baseline"

    SCORE_OUTPUT.write_text(json.dumps({
        "score": score, "reason": reason,
        "agent_time_seconds": agent_time, "baseline_time_seconds": baseline_time,
        "num_prompts": len(references),
        "num_matched": matched, "num_mismatched": mismatched,
        "details": details,
    }, indent=2))

    print(f"\n{'=' * 60}\n  SCORE: {score:.4f}\n  {reason}\n{'=' * 60}")
    print(f"  Details written to {SCORE_OUTPUT}")
    return score


if __name__ == "__main__":
    sys.exit(0 if main() > 0 else 1)
