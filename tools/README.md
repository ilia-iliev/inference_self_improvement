# Agent Sandbox

The runtime an LLM agent operates inside. Gives it a GPU-capable shell with python/uv/torch/vllm pre-installed, scoped to the files it's allowed to touch.

## What the agent sees

Mounted into `/workspace/`:

- `solution/` — **rw**. The agent edits its Dockerfile, server.py, and any scratch here.
- `data/agent/` — ro. Dev prompts + HF greedy references for local testing.
- `judge/` — ro. Baseline reference impl (`judge/baseline/`) and current baseline timing (`judge/baseline.json`). Submit/client internals are visible but read-only.
- `PROMPT.md` — ro. The task definition.

**Not mounted**: `data/judge/` (held-out eval set). There is no docker socket — the agent cannot launch containers.

## Available inside the sandbox

- `python3`, `uv` (add/remove packages freely, they're scoped to the container).
- `torch`, `transformers`, `vllm`, `accelerate` pre-installed.
- `nvidia-smi`, full GPU access.
- Network (for `uv add`, `pip install`, etc.). HF Hub reads are offline — all model weights must come from the mounted HF cache.

## Submission flow

The agent cannot run the judge from inside the sandbox (no docker access). The loop is:

1. Host: `./run_challenge.sh agent` drops the agent into the sandbox shell.
2. Agent: iterates on `/workspace/solution/`, experiments with `python3`, `uv`, etc.
3. Agent: exits when ready to be scored.
4. Host: `./run_challenge.sh submit` (or `dev` for fast iteration) builds the agent's Dockerfile and scores it. Result lands in `solution/last_result.json`, which the agent sees on next entry.
