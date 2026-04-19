#!/usr/bin/env python3
"""
Judge for the Gemma 4 2B inference throughput challenge.

Procedure:
  1. Load HF reference outputs (ground truth)
  2. Load vLLM baseline timing
  3. Run the agent's solution, measure wall-clock time
  4. Verify token-for-token match against reference
  5. Compute score

Scoring:
  if any output mismatches reference → score = 0
  elif agent_time >= baseline_time  → score = 0
  else → score = baseline_time / agent_time  (>1.0 means faster)

Reads:
  /data/eval_reference.jsonl        — HF greedy reference outputs
  /judge/baseline_timing.json       — baseline wall-clock time
  /solution/outputs.jsonl           — agent's outputs

Writes:
  /judge/score.json                 — final score + details
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

MODEL_PATH = "/models/gemma4-2b"
EVAL_PROMPTS = Path("/judge/eval_prompts.jsonl")
EVAL_REFERENCE = Path("/data/eval_reference.jsonl")
BASELINE_TIMING = Path("/judge/baseline_timing.json")
SOLUTION_SCRIPT = Path("/solution/run.py")
SOLUTION_OUTPUTS = Path("/solution/outputs.jsonl")
SOLUTION_TIMING = Path("/solution/timing.json")
SCORE_OUTPUT = Path("/judge/score.json")


def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def verify_outputs(
    references: list[dict],
    agent_outputs: list[dict],
    processor,
) -> tuple[bool, list[dict]]:
    """
    Verify that agent outputs match references token-for-token.

    Compares by tokenizing both completions and checking token ID equality.
    Returns (all_match, details).
    """
    ref_by_id = {r["id"]: r for r in references}
    agent_by_id = {a["id"]: a for a in agent_outputs}

    details = []
    all_match = True

    # Check all reference IDs are present in agent output
    missing = set(ref_by_id.keys()) - set(agent_by_id.keys())
    if missing:
        print(f"ERROR: Agent output missing {len(missing)} prompts: {sorted(missing)[:5]}...")
        all_match = False
        for mid in missing:
            details.append({"id": mid, "match": False, "reason": "missing from agent output"})

    for pid, ref in ref_by_id.items():
        if pid not in agent_by_id:
            continue

        agent = agent_by_id[pid]
        ref_text = ref["completion"]
        agent_text = agent["completion"]

        # Tokenize both and compare token IDs
        ref_tokens = processor.tokenizer.encode(ref_text, add_special_tokens=False)
        agent_tokens = processor.tokenizer.encode(agent_text, add_special_tokens=False)

        if ref_tokens == agent_tokens:
            details.append({
                "id": pid,
                "match": True,
                "num_tokens": len(ref_tokens),
            })
        else:
            all_match = False
            # Find first divergence point
            diverge_idx = 0
            for j in range(min(len(ref_tokens), len(agent_tokens))):
                if ref_tokens[j] != agent_tokens[j]:
                    diverge_idx = j
                    break
            else:
                diverge_idx = min(len(ref_tokens), len(agent_tokens))

            details.append({
                "id": pid,
                "match": False,
                "reason": "token mismatch",
                "ref_len": len(ref_tokens),
                "agent_len": len(agent_tokens),
                "first_divergence": diverge_idx,
                "ref_snippet": ref_text[:200],
                "agent_snippet": agent_text[:200],
            })

    return all_match, details


def run_agent_solution(prompts_path: Path) -> float:
    """
    Run the agent's solution script and return wall-clock time.
    """
    if not SOLUTION_SCRIPT.exists():
        print(f"ERROR: Solution script not found at {SOLUTION_SCRIPT}")
        sys.exit(1)

    # Clean previous outputs
    if SOLUTION_OUTPUTS.exists():
        SOLUTION_OUTPUTS.unlink()

    cmd = [sys.executable, str(SOLUTION_SCRIPT), str(prompts_path)]
    print(f"Running agent solution: {' '.join(cmd)}")

    t_start = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 minute timeout (baseline is ~15 min; give 2x margin)
    )
    t_end = time.perf_counter()

    if result.returncode != 0:
        print(f"ERROR: Solution exited with code {result.returncode}")
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        sys.exit(1)

    wall_time = t_end - t_start
    print(f"Agent solution completed in {wall_time:.2f}s")

    if not SOLUTION_OUTPUTS.exists():
        print(f"ERROR: Solution did not produce {SOLUTION_OUTPUTS}")
        sys.exit(1)

    return wall_time


def main():
    print("=" * 60)
    print("  Gemma 4 2B Inference Throughput Challenge — Judge")
    print("=" * 60)

    # Step 1: Load references
    if not EVAL_REFERENCE.exists():
        print(f"ERROR: Reference file not found at {EVAL_REFERENCE}")
        print("Run generate_reference.py first.")
        sys.exit(1)
    references = load_jsonl(EVAL_REFERENCE)
    print(f"Loaded {len(references)} reference outputs")

    # Step 2: Load baseline timing
    if not BASELINE_TIMING.exists():
        print(f"ERROR: Baseline timing not found at {BASELINE_TIMING}")
        print("Run run_vllm_baseline.py first.")
        sys.exit(1)
    with open(BASELINE_TIMING) as f:
        baseline = json.load(f)
    baseline_time = baseline["wall_time_seconds"]
    print(f"Baseline time: {baseline_time:.2f}s")

    # Step 3: Read agent solution timing (solution must be run separately first)
    if not SOLUTION_OUTPUTS.exists():
        print(f"ERROR: Solution outputs not found at {SOLUTION_OUTPUTS}")
        print("Run the solution first: ./run_challenge.sh solution")
        sys.exit(1)
    if not SOLUTION_TIMING.exists():
        print(f"ERROR: Solution timing not found at {SOLUTION_TIMING}")
        print("Re-run the solution so it writes timing.json.")
        sys.exit(1)
    with open(SOLUTION_TIMING) as f:
        agent_time = json.load(f)["wall_time_seconds"]
    print(f"Agent time: {agent_time:.2f}s (from {SOLUTION_TIMING})")

    # Step 4: Load agent outputs and verify
    agent_outputs = load_jsonl(SOLUTION_OUTPUTS)
    print(f"Loaded {len(agent_outputs)} agent outputs")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    all_match, details = verify_outputs(references, agent_outputs, processor)

    matched = sum(1 for d in details if d.get("match"))
    mismatched = sum(1 for d in details if not d.get("match"))
    print(f"Verification: {matched} matched, {mismatched} mismatched")

    # Step 5: Compute score
    if not all_match:
        score = 0.0
        reason = f"Output mismatch: {mismatched} of {len(details)} outputs differ from reference"
    elif agent_time >= baseline_time:
        score = 0.0
        reason = f"Agent ({agent_time:.2f}s) not faster than baseline ({baseline_time:.2f}s)"
    else:
        score = baseline_time / agent_time
        reason = f"Agent is {score:.2f}x faster than baseline"

    # Print mismatches for debugging
    if not all_match:
        print("\nMismatched outputs:")
        for d in details:
            if not d.get("match"):
                print(f"  {d['id']}: {d.get('reason', 'unknown')}")
                if "ref_snippet" in d:
                    print(f"    ref:   {d['ref_snippet'][:100]}...")
                    print(f"    agent: {d['agent_snippet'][:100]}...")

    # Write score
    score_data = {
        "score": score,
        "reason": reason,
        "agent_time_seconds": agent_time,
        "baseline_time_seconds": baseline_time,
        "num_prompts": len(references),
        "num_matched": matched,
        "num_mismatched": mismatched,
        "details": details,
    }
    with open(SCORE_OUTPUT, "w") as f:
        json.dump(score_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  SCORE: {score:.4f}")
    print(f"  {reason}")
    print(f"{'=' * 60}")
    print(f"  Details written to {SCORE_OUTPUT}")

    return score


if __name__ == "__main__":
    score = main()
    sys.exit(0 if score > 0 else 1)
