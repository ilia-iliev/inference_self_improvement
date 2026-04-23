"""
Baseline inference server. Implements the challenge protocol (see README.md):

  GET  /health           -> {"status": "ready"|"loading"}
  POST /v1/completions   -> {"id", "completion"}

Model loading + prompt construction are shared with inference.py via _common.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import MAX_NEW_TOKENS, build_inputs, load  # noqa: E402

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        model, proc, tok, device = await loop.run_in_executor(pool, load)
    state.update(model=model, processor=proc, tokenizer=tok, device=device, ready=True)
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
    try:
        inputs = build_inputs(req.prompt, req.modality, state["processor"], state["device"],
                              image=req.image, audio=req.audio)
    except ValueError as e:
        raise HTTPException(400, str(e))
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = state["model"].generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return CompletionResponse(
        id=req.id,
        completion=state["tokenizer"].decode(out[0, n_in:], skip_special_tokens=True),
    )
