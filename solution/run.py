#!/usr/bin/env python3
"""
Solution template for the Gemma 4 2B inference throughput challenge.

Interface:
  python3 /solution/run.py <prompts.jsonl>

Input format (each line):
  {"id": "...", "modality": "text|image|audio", "prompt": "...",
   "image": "<path>", "audio": "<path>"}

Output: writes /solution/outputs.jsonl, each line:
  {"id": "...", "prompt": "...", "completion": "..."}

Constraints:
  - Greedy decoding, max 256 new tokens
  - Outputs must match HF single-request greedy reference token-for-token
  - May use any technique: batching, torch.compile, CUDA graphs, dual-GPU, etc.

This template is the NAIVE single-request baseline. Replace with your
optimized version to beat the vLLM baseline.
"""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = "/models/gemma4-2b"
MAX_NEW_TOKENS = 256
OUTPUT_PATH = Path("/solution/outputs.jsonl")
TIMING_PATH = Path("/solution/timing.json")


def build_prompt_text(prompt: str, modality: str) -> str:
    if modality == "image":
        media = "<start_of_image><image_soft_token><end_of_image>\n"
    elif modality == "audio":
        media = "<start_of_audio><audio_soft_token><end_of_audio>\n"
    else:
        media = ""
    return f"<start_of_turn>user\n{media}{prompt}<end_of_turn>\n<start_of_turn>model\n"


def build_inputs(p: dict, processor, device):
    text = build_prompt_text(p["prompt"], p["modality"])
    kwargs = {"text": text, "return_tensors": "pt"}
    if p["modality"] == "image":
        kwargs["images"] = Image.open(p["image"]).convert("RGB")
    elif p["modality"] == "audio":
        import librosa
        audio_data, _ = librosa.load(p["audio"], sr=16000)
        kwargs["audio"] = audio_data
    inputs = processor(**kwargs)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run.py <prompts.jsonl>")
        sys.exit(1)

    prompts_path = Path(sys.argv[1])
    with open(prompts_path) as f:
        prompts = [json.loads(line) for line in f]
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    device = next(model.parameters()).device
    results = []
    t_start = time.perf_counter()

    for p in prompts:
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
        results.append({
            "id": p["id"],
            "prompt": p["prompt"],
            "completion": completion,
        })

    wall_time = time.perf_counter() - t_start
    print(f"Generated {len(results)} completions in {wall_time:.2f}s")

    with open(OUTPUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Written to {OUTPUT_PATH}")

    with open(TIMING_PATH, "w") as f:
        json.dump({
            "wall_time_seconds": wall_time,
            "num_prompts": len(results),
            "prompts_file": str(prompts_path),
        }, f, indent=2)
    print(f"Timing written to {TIMING_PATH}")


if __name__ == "__main__":
    main()
