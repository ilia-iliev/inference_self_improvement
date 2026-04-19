#!/usr/bin/env python3
"""
Naive solution: delegates to /baseline/inference.py.

Replace the call site with an optimized implementation (batching, compile,
dual-GPU, etc.) that matches HF greedy outputs token-for-token.
"""

import sys
from pathlib import Path

sys.path.insert(0, "/baseline")
from inference import run  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run.py <prompts.jsonl>")
        sys.exit(1)
    run(Path(sys.argv[1]), Path("/solution/outputs.jsonl"), Path("/solution/timing.json"))
