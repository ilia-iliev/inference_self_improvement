"""
Baseline inference server. Implements the challenge protocol (see README.md):

  GET  /health           -> {"status": "ready"|"loading"}
  POST /v1/completions   -> {"id", "completion"}

Modality toggled by TEXT_ONLY env var (see judge/docker/.env.example).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "google/gemma-3-1b-it")
MAX_NEW_TOKENS = 256
TEXT_ONLY = os.environ.get("TEXT_ONLY", "1") == "1"

_MEDIA_TOKEN = {"image": "<|image|>\n", "audio": "<|audio|>\n"}

state: dict = {}


def _gpu_max_memory() -> dict:
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA GPUs available")
    return {i: f"{torch.cuda.get_device_properties(i).total_memory // (1024**3)}GiB"
            for i in range(n)}


def _build_inputs(req: "CompletionRequest"):
    proc = state["processor"]
    device = state["device"]
    if TEXT_ONLY:
        if req.modality != "text":
            raise HTTPException(400, f"TEXT_ONLY=1 but request modality={req.modality!r}")
        text = proc.apply_chat_template(
            [{"role": "user", "content": req.prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = proc(text, return_tensors="pt")
    else:
        from PIL import Image
        text = (f"<|turn>user\n{_MEDIA_TOKEN.get(req.modality, '')}"
                f"{req.prompt}<turn|>\n<|turn>model\n")
        kwargs = {"text": text, "return_tensors": "pt"}
        if req.modality == "image":
            if not req.image:
                raise HTTPException(400, "image modality requires 'image' field")
            kwargs["images"] = Image.open(req.image).convert("RGB")
        elif req.modality == "audio":
            if not req.audio:
                raise HTTPException(400, "audio modality requires 'audio' field")
            import librosa
            kwargs["audio"], _ = librosa.load(req.audio, sr=16000)
        inputs = proc(**kwargs)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    max_memory = _gpu_max_memory()
    if TEXT_ONLY:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        state["processor"] = AutoTokenizer.from_pretrained(MODEL_PATH)
        state["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
        ).eval()
        state["tokenizer"] = state["processor"]
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        state["processor"] = AutoProcessor.from_pretrained(MODEL_PATH)
        state["model"] = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory,
        ).eval()
        state["tokenizer"] = state["processor"].tokenizer
    state["device"] = next(state["model"].parameters()).device
    assert state["device"].type == "cuda", f"model on {state['device']}, expected cuda"
    state["ready"] = True
    yield


app = FastAPI(lifespan=lifespan)


class CompletionRequest(BaseModel):
    id: str
    modality: str
    prompt: str
    image: str | None = None
    audio: str | None = None


class CompletionResponse(BaseModel):
    id: str
    completion: str


@app.get("/health")
def health():
    return {"status": "ready" if state.get("ready") else "loading"}


@app.post("/v1/completions", response_model=CompletionResponse)
def complete(req: CompletionRequest):
    if not state.get("ready"):
        raise HTTPException(503, "not ready")
    inputs = _build_inputs(req)
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = state["model"].generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new = out[0, n_in:]
    return CompletionResponse(
        id=req.id,
        completion=state["tokenizer"].decode(new, skip_special_tokens=True),
    )
