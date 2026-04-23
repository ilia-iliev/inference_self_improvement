"""Shared model loading + input construction for the baseline.

Both inference.py (offline reference generator) and server.py (serving
protocol) load Gemma the same way and construct identical inputs. Keep
this module the single source of truth; the promotion flow depends on
byte-identical tokens between the two.

Modality is detected automatically from MODEL_PATH/config.json:
  vision_config / audio_config present → AutoProcessor + AutoModelForImageTextToText
  neither present                       → AutoTokenizer + AutoModelForCausalLM
"""
from __future__ import annotations

import os

import torch
from transformers import AutoConfig

MODEL_PATH = os.environ["MODEL_PATH"]
MAX_NEW_TOKENS = 256

_cfg = AutoConfig.from_pretrained(MODEL_PATH)
TEXT_ONLY = not (hasattr(_cfg, "vision_config") or hasattr(_cfg, "audio_config"))


# Gemma 4 ships its own chat template (chat_template.jinja). We build the
# per-modality message content and let processor.apply_chat_template render
# the text with the right special tokens; the processor then expands the
# inline <|image|>/<|audio|> markers to the correct soft-token count when
# called with images=/audio=.
def _mm_messages(prompt: str, modality: str) -> list[dict]:
    content: list[dict] = []
    if modality == "image":
        content.append({"type": "image"})
    elif modality == "audio":
        content.append({"type": "audio"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _gpu_max_memory() -> dict:
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA GPUs available")
    return {i: f"{torch.cuda.get_device_properties(i).total_memory // (1024**3)}GiB"
            for i in range(n)}


def load():
    """Load model + processor. Returns (model, processor, tokenizer, device)."""
    max_memory = _gpu_max_memory()
    if TEXT_ONLY:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        proc = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
        ).eval()
        tok = proc
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(MODEL_PATH)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
        ).eval()
        tok = proc.tokenizer
    device = next(model.parameters()).device
    assert device.type == "cuda", f"model loaded on {device}, expected cuda"
    return model, proc, tok, device


def build_inputs(prompt: str, modality: str, proc, device,
                 image: str | None = None, audio: str | None = None) -> dict:
    if TEXT_ONLY:
        if modality != "text":
            raise ValueError(f"text-only model but modality={modality!r}")
        text = proc.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = proc(text, return_tensors="pt")
    else:
        from PIL import Image
        text = proc.apply_chat_template(
            _mm_messages(prompt, modality),
            tokenize=False, add_generation_prompt=True,
        )
        kwargs = {"text": text, "return_tensors": "pt"}
        if modality == "image":
            if not image:
                raise ValueError("image modality requires 'image' field")
            kwargs["images"] = Image.open(image).convert("RGB")
        elif modality == "audio":
            if not audio:
                raise ValueError("audio modality requires 'audio' field")
            import librosa
            kwargs["audio"], _ = librosa.load(audio, sr=16000)
        inputs = proc(**kwargs)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}
