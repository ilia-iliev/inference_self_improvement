#!/usr/bin/env python3
"""
vLLM baseline for the Gemma 4 2B inference throughput challenge.

Runs all eval prompts through vLLM with greedy decoding, records wall-clock time.
If vLLM doesn't support Gemma 4's multimodal architecture, falls back to an
optimized HuggingFace pipeline as the baseline.

Writes:
  /judge/baseline_outputs.jsonl
  /judge/baseline_timing.json
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import build_inputs, build_prompt_text

EVAL_PROMPTS = Path("/judge/eval_prompts.jsonl")
DEV_PROMPTS = Path("/data/dev_prompts.jsonl")
BASELINE_OUTPUTS = Path("/judge/baseline_outputs.jsonl")
BASELINE_TIMING = Path("/judge/baseline_timing.json")
MODEL_PATH = "/models/gemma4-2b"
MAX_NEW_TOKENS = 256


def load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def try_vllm_baseline(prompts: list[dict]):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed, falling back to HF baseline.")
        return None

    try:
        print("Attempting vLLM baseline...")
        llm = LLM(
            model=MODEL_PATH,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

        vllm_requests = []
        for p in prompts:
            req = {"prompt": build_prompt_text(p["prompt"], p["modality"])}
            mm_data = {}
            if p["modality"] == "image":
                from PIL import Image
                mm_data["image"] = Image.open(p["image"]).convert("RGB")
            elif p["modality"] == "audio":
                import librosa
                audio_data, _ = librosa.load(p["audio"], sr=16000)
                mm_data["audio"] = audio_data
            if mm_data:
                req["multi_modal_data"] = mm_data
            vllm_requests.append(req)

        print(f"Running vLLM on {len(prompts)} prompts...")
        t_start = time.perf_counter()
        outputs = llm.generate(vllm_requests, sampling_params, use_tqdm=True)
        wall_time = time.perf_counter() - t_start

        results = []
        for p, o in zip(prompts, outputs):
            results.append({
                "id": p["id"],
                "prompt": p["prompt"],
                "modality": p["modality"],
                "completion": o.outputs[0].text,
                "num_output_tokens": len(o.outputs[0].token_ids),
            })

        print(f"vLLM baseline: {wall_time:.2f}s for {len(prompts)} prompts")
        return results, wall_time

    except Exception as e:
        print(f"vLLM failed: {e}")
        import traceback; traceback.print_exc()
        print("Falling back to HF baseline.")
        return None


def hf_baseline(prompts: list[dict]):
    """Fallback baseline: optimized HF pipeline with torch.compile."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print("Running HuggingFace baseline...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    # Skip torch.compile for mixed multimodal: reduce-overhead mode uses CUDA graphs
    # which break on variable input shapes across text/image/audio prompts.

    results = []
    device = next(model.parameters()).device
    t_start = time.perf_counter()

    for i, p in enumerate(prompts):
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
            "modality": p["modality"],
            "completion": completion,
            "num_output_tokens": int(new_token_ids.shape[0]),
        })

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}s elapsed")

    wall_time = time.perf_counter() - t_start
    print(f"HF baseline: {wall_time:.2f}s for {len(prompts)} prompts")
    return results, wall_time


def main():
    if EVAL_PROMPTS.exists():
        prompts_path = EVAL_PROMPTS
    elif DEV_PROMPTS.exists():
        prompts_path = DEV_PROMPTS
    else:
        print("ERROR: No prompt files found.")
        sys.exit(1)

    prompts = load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    result = try_vllm_baseline(prompts)
    if result is None:
        result = hf_baseline(prompts)
    results, wall_time = result

    BASELINE_OUTPUTS.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_OUTPUTS, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    timing = {
        "wall_time_seconds": wall_time,
        "num_prompts": len(prompts),
        "total_output_tokens": sum(r["num_output_tokens"] for r in results),
        "prompts_file": str(prompts_path),
    }
    with open(BASELINE_TIMING, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\nBaseline complete:")
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  Outputs:   {BASELINE_OUTPUTS}")
    print(f"  Timing:    {BASELINE_TIMING}")


if __name__ == "__main__":
    main()
