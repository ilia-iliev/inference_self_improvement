"""
HTTP client the judge uses to drive an inference container through the
challenge protocol (see README.md). No baseline/agent-specific logic.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx


def load_prompts(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


async def wait_for_ready(client: httpx.AsyncClient, timeout_s: float = 300.0) -> bool:
    """Poll /health until it reports 'ready'. Return True on success, False on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = await client.get("/health")
            if r.status_code == 200 and r.json().get("status") == "ready":
                return True
        except httpx.RequestError:
            pass
        await asyncio.sleep(1.0)
    return False


async def _one(
    client: httpx.AsyncClient, prompt: dict, request_timeout_s: float
) -> dict:
    r = await client.post("/v1/completions", json=prompt, timeout=request_timeout_s)
    r.raise_for_status()
    return r.json()


async def fire_all(
    client: httpx.AsyncClient, prompts: list[dict], request_timeout_s: float = 120.0
) -> tuple[list[dict], float]:
    """Fire all prompts concurrently. Return (responses_in_input_order, wall_seconds)."""
    t0 = time.perf_counter()
    responses = await asyncio.gather(
        *(_one(client, p, request_timeout_s) for p in prompts)
    )
    return responses, time.perf_counter() - t0


async def run_eval(
    base_url: str,
    prompts: list[dict],
    ready_timeout_s: float = 300.0,
    request_timeout_s: float = 120.0,
) -> dict:
    """Wait for readiness, fire all prompts, return timing + responses (or error)."""
    async with httpx.AsyncClient(base_url=base_url) as client:
        if not await wait_for_ready(client, ready_timeout_s):
            return {"ready": False, "error": f"not ready after {ready_timeout_s}s"}
        responses, wall = await fire_all(client, prompts, request_timeout_s)
        return {"ready": True, "wall_time_seconds": wall, "responses": responses}
