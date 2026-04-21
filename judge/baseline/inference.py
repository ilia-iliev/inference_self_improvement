#!/usr/bin/env python3
"""
Greedy single-request inference.

Modality toggled by TEXT_ONLY env var (see judge/docker/.env.example):
  TEXT_ONLY=1 → AutoTokenizer + AutoModelForCausalLM (Gemma 3 1B IT etc.)
  TEXT_ONLY=0 → AutoProcessor + AutoModelForImageTextToText (Gemma 4 etc.)

CLI:
  python3 inference.py <prompts.jsonl> <outputs.jsonl> [<timing.json>]
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

MODEL_PATH = os.environ.get("MODEL_PATH", "google/gemma-3-1b-it")
MAX_NEW_TOKENS = 256
TEXT_ONLY = os.environ.get("TEXT_ONLY", "1") == "1"

# Multimodal (Gemma 4) template tokens. Unused in text-only mode.
_MEDIA_TOKEN = {"image": "<|image|>\n", "audio": "<|audio|>\n"}


def _gpu_max_memory() -> dict:
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA GPUs available")
    return {i: f"{torch.cuda.get_device_properties(i).total_memory // (1024**3)}GiB"
            for i in range(n)}


def load():
    max_memory = _gpu_max_memory()
    if TEXT_ONLY:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        proc = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
        ).eval()
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL_PATH)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
        ).eval()
    device = next(model.parameters()).device
    assert device.type == "cuda", f"model loaded on {device}, expected cuda"
    return model, proc, device


def build_inputs(p: dict, proc, device) -> dict:
    if TEXT_ONLY:
        text = proc.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = proc(text, return_tensors="pt")
    else:
        from PIL import Image
        text = (f"<|turn>user\n{_MEDIA_TOKEN.get(p['modality'], '')}"
                f"{p['prompt']}<turn|>\n<|turn>model\n")
        kwargs = {"text": text, "return_tensors": "pt"}
        if p["modality"] == "image":
            kwargs["images"] = Image.open(p["image"]).convert("RGB")
        elif p["modality"] == "audio":
            import librosa
            kwargs["audio"], _ = librosa.load(p["audio"], sr=16000)
        inputs = proc(**kwargs)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}


def tokenizer_of(proc):
    return proc if TEXT_ONLY else proc.tokenizer


def run(prompts_path: Path, outputs_path: Path, timing_path: Path | None = None):
    prompts = [json.loads(l) for l in open(prompts_path)]
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    model, proc, device = load()
    tok = tokenizer_of(proc)

    results = []
    t0 = time.perf_counter()
    for i, p in enumerate(prompts):
        inputs = build_inputs(p, proc, device)
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
