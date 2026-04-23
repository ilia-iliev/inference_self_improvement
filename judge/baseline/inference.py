#!/usr/bin/env python3
"""
Greedy single-request inference — offline reference generator.

Shares model loading + input construction with server.py via _common, so
the outputs are byte-identical to what the baseline server produces.

CLI:
  python3 inference.py <prompts.jsonl> <outputs.jsonl> [<timing.json>]
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MAX_NEW_TOKENS, build_inputs, load  # noqa: E402


def run(prompts_path: Path, outputs_path: Path, timing_path: Path | None = None):
    prompts = [json.loads(l) for l in open(prompts_path)]
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    model, proc, tok, device = load()

    results = []
    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        inputs = build_inputs(p["prompt"], p["modality"], proc, device,
                              image=p.get("image"), audio=p.get("audio"))
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        new = out[0, n_in:]
        results.append({
            "id": p["id"], "prompt": p["prompt"], "modality": p["modality"],
            "completion": tok.decode(new, skip_special_tokens=True),
            "num_input_tokens": int(n_in),
            "num_output_tokens": int(new.shape[0]),
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(prompts)}] {time.perf_counter() - t0:.1f}s elapsed")

    wall = time.perf_counter() - t0
    outputs_path = Path(outputs_path)
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(outputs_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(results)} outputs to {outputs_path} in {wall:.2f}s")

    if timing_path:
        Path(timing_path).write_text(json.dumps({
            "wall_time_seconds": wall,
            "num_prompts": len(results),
            "total_output_tokens": sum(r["num_output_tokens"] for r in results),
            "prompts_file": str(prompts_path),
        }, indent=2))
        print(f"Wrote timing to {timing_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 inference.py <prompts.jsonl> <outputs.jsonl> [<timing.json>]")
        sys.exit(1)
    run(Path(sys.argv[1]), Path(sys.argv[2]),
        Path(sys.argv[3]) if len(sys.argv) > 3 else None)
