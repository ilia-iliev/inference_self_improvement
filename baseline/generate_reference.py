#!/usr/bin/env python3
"""
Generate canonical greedy reference outputs using HuggingFace transformers.

Processes prompts ONE AT A TIME with greedy decoding (no sampling, no beam search)
to produce the ground-truth outputs that all solutions must match token-for-token.

This is intentionally slow — it's the correctness oracle, not a performance target.

Reads:
  /data/dev_prompts.jsonl
  /data/eval_prompts.jsonl  (if it exists)

Writes:
  /data/dev_reference.jsonl
  /data/eval_reference.jsonl
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_inputs

MODEL_PATH = "/models/gemma4-2b"
MAX_NEW_TOKENS = 256
DEV_PROMPTS = Path("/data/dev_prompts.jsonl")
DEV_REFERENCE = Path("/data/dev_reference.jsonl")
EVAL_PROMPTS = Path("/data/eval_prompts.jsonl")
EVAL_REFERENCE = Path("/data/eval_reference.jsonl")


def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    print("Model loaded.")
    return model, processor


def generate_references(prompts_path: Path, output_path: Path, model, processor):
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line))

    print(f"Generating references for {len(prompts)} prompts from {prompts_path}...")
    device = next(model.parameters()).device
    results = []

    for i, p in enumerate(prompts):
        t0 = time.perf_counter()

        inputs = build_inputs(p, processor, device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        new_token_ids = output_ids[0, input_len:]
        completion = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        elapsed = time.perf_counter() - t0

        results.append({
            "id": p["id"],
            "prompt": p["prompt"],
            "modality": p["modality"],
            "completion": completion,
            "num_input_tokens": int(input_len),
            "num_output_tokens": int(new_token_ids.shape[0]),
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(prompts)}] {p['id']} ({p['modality']}) — "
                  f"{new_token_ids.shape[0]} toks, {elapsed:.2f}s")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = sum(r["num_output_tokens"] for r in results)
    print(f"  Written {len(results)} references to {output_path} ({total} output tokens)")
    return results


def main():
    model, processor = load_model()

    if DEV_PROMPTS.exists():
        generate_references(DEV_PROMPTS, DEV_REFERENCE, model, processor)

    if EVAL_PROMPTS.exists():
        generate_references(EVAL_PROMPTS, EVAL_REFERENCE, model, processor)

    print("Reference generation complete.")


if __name__ == "__main__":
    main()
