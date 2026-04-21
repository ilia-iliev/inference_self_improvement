"""
Unit tests for judge/client.py. Uses httpx.ASGITransport to run a fake
FastAPI app in-process — no docker, no GPU, no network.

Run: uv run --with httpx --with fastapi --with pytest --with pytest-asyncio pytest tests/test_client.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judge.client import fire_all, run_eval, wait_for_ready  # noqa: E402


def _app(on_complete=None, health_states=None):
    """Fake server. health_states: list of dicts returned in sequence by /health."""
    app = FastAPI()
    health_q = list(health_states) if health_states is not None else [{"status": "ready"}]

    @app.get("/health")
    async def health():
        return health_q.pop(0) if len(health_q) > 1 else health_q[0]

    @app.post("/v1/completions")
    async def complete(body: dict):
        if on_complete:
            return await on_complete(body)
        return {"id": body["id"], "completion": f"echo:{body['prompt']}"}

    return app


def _client_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_wait_for_ready_succeeds_immediately():
    async with _client_for(_app()) as c:
        assert await wait_for_ready(c, timeout_s=5.0)


@pytest.mark.asyncio
async def test_wait_for_ready_flips_from_loading():
    app = _app(health_states=[{"status": "loading"}, {"status": "loading"}, {"status": "ready"}])
    async with _client_for(app) as c:
        assert await wait_for_ready(c, timeout_s=10.0)


@pytest.mark.asyncio
async def test_wait_for_ready_times_out():
    app = _app(health_states=[{"status": "loading"}])
    async with _client_for(app) as c:
        assert not await wait_for_ready(c, timeout_s=2.0)


@pytest.mark.asyncio
async def test_fire_all_returns_in_input_order():
    prompts = [{"id": f"p{i}", "modality": "text", "prompt": f"q{i}"} for i in range(5)]
    async with _client_for(_app()) as c:
        responses, wall = await fire_all(c, prompts)
    assert [r["id"] for r in responses] == ["p0", "p1", "p2", "p3", "p4"]
    assert [r["completion"] for r in responses] == [f"echo:q{i}" for i in range(5)]
    assert wall >= 0


@pytest.mark.asyncio
async def test_fire_all_is_concurrent():
    """If the server takes 0.2s per request and we fire 5 concurrently, total should be ~0.2s, not ~1s."""
    async def slow(body):
        await asyncio.sleep(0.2)
        return {"id": body["id"], "completion": "ok"}

    prompts = [{"id": f"p{i}", "modality": "text", "prompt": "x"} for i in range(5)]
    async with _client_for(_app(on_complete=slow)) as c:
        _, wall = await fire_all(c, prompts)
    assert wall < 0.8, f"expected concurrent (~0.2s), got {wall:.2f}s"


@pytest.mark.asyncio
async def test_fire_all_propagates_server_error():
    from fastapi import HTTPException

    async def boom(body):
        raise HTTPException(status_code=500, detail="nope")

    prompts = [{"id": "p0", "modality": "text", "prompt": "x"}]
    async with _client_for(_app(on_complete=boom)) as c:
        with pytest.raises(httpx.HTTPStatusError):
            await fire_all(c, prompts)


@pytest.mark.asyncio
async def test_run_eval_happy_path(monkeypatch):
    """run_eval builds its own client via base_url; patch httpx.AsyncClient to use ASGITransport."""
    app = _app()
    original = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.ASGITransport(app=app)
        kwargs.setdefault("base_url", "http://test")
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    prompts = [{"id": "p0", "modality": "text", "prompt": "x"}]
    result = await run_eval("http://ignored", prompts, ready_timeout_s=2.0)
    assert result["ready"] is True
    assert result["responses"][0]["completion"] == "echo:x"
    assert result["wall_time_seconds"] > 0


@pytest.mark.asyncio
async def test_run_eval_reports_unready(monkeypatch):
    app = _app(health_states=[{"status": "loading"}])
    original = httpx.AsyncClient

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.ASGITransport(app=app)
        kwargs.setdefault("base_url", "http://test")
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched)
    result = await run_eval("http://ignored", [], ready_timeout_s=1.0)
    assert result["ready"] is False
    assert "error" in result
