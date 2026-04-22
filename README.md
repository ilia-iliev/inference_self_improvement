# Gemma Inference Throughput Challenge

A self-improving eval harness. An agent iteratively edits `solution/`, the harness builds it, times it against a baseline on held-out prompts, and promotes the solution to baseline when it's fast enough and outputs are still bit-exact.

## Layout

```
PROMPT.md              agent-facing problem statement
solution/              agent's workspace (Dockerfile + code)
judge/                 harness (host-side orchestration + baseline + docker infra)
  baseline/            reference impl the agent has to beat
  docker/              base image, compose, .env
  submit.py            build → run → verify → score → promote
  client.py            HTTP driver
  baseline.json        current baseline timing
data/
  agent/               dev prompts + refs + assets (visible)
  judge/               eval prompts + refs + assets (held out)
tools/                 permission allowlist for the agent runner
tests/                 preflight + client unit tests (no docker, no GPU)
run_challenge.sh       one-shot CLI entry points
```

## Flow

```
                       ┌────────────────┐
                       │ data/judge/    │  held-out prompts + HF greedy refs
                       └───────┬────────┘
                               │
    ┌──────────────────┐       │      ┌────────────┐
    │ judge/baseline/  │─────┐ │ ┌───►│ judge/     │  build images, time runs,
    └──────────────────┘     │ │ │    │ submit.py  │  verify tokens, score,
                             ▼ ▼ │    └─────┬──────┘  promote if ≥ 1.05×
                          docker │          │
                          images │          ▼
                             ▲   │   baseline.json
    ┌──────────────────┐     │   │   last_result.json
    │ solution/        │─────┘   │
    │   (agent edits)  │◄────────┘
    └──────────────────┘  PROMPT.md points agent here
```

## Protocol

Both baseline and solution images expose the same HTTP API on `:8000`:

- `GET /health` → `200 {"status": "ready"|"loading"}`. Judge polls until `ready` or the 5-minute startup cap.
- `POST /v1/completions` body `{"id", "modality" ∈ {text,image,audio}, "prompt", "image"?, "audio"?}` → `200 {"id", "completion"}`. `image`/`audio` are paths under the read-only `/data` mount.
- Greedy decode, `max_new_tokens=256`, token-for-token identical to the HF reference.
- Judge fires all eval prompts concurrently; timing = wall clock from first POST to last response.

## Scoring

```
any token mismatch             → 0
agent_time ≥ baseline_time     → 0
else                           → baseline_time / agent_time
```

Promotion gate: `score ≥ 1.05` and all outputs match → solution image is retagged as the new baseline and re-measured. Persisted in `judge/baseline.json`.

## Host requirements

- ≥1 NVIDIA GPU, driver ≥570 (CUDA 12.8)
- Docker + compose + `nvidia-container-toolkit`
- Target model in the local HF cache (default: `google/gemma-3-1b-it`)

## Configuration

Copy `judge/docker/.env.example` → `judge/docker/.env`:

- `HF_CACHE_DIR` — host path, mounted read-only at `/root/.cache/huggingface`
- `MODEL_PATH` — HF repo id (offline-resolved) or absolute in-container path
- `TEXT_ONLY` — `1` uses `AutoTokenizer + AutoModelForCausalLM`; `0` switches to `AutoProcessor + AutoModelForImageTextToText` for image/audio prompts

## Running

Setup (once):
```
./run_challenge.sh preflight      # validate .env, no docker
./run_challenge.sh build          # build base image (heavy)
./run_challenge.sh data           # generate prompts + assets + HF greedy refs
```

Per iteration:
```
# agent edits solution/Dockerfile and/or solution/server.py
./run_challenge.sh submit         # build, evaluate, score, maybe promote
                                  # auto-inits baseline timing on first run
# see solution/last_result.json
```

LLM-driven loop (optional; drives the edit step on the agent's behalf):
```
./run_challenge.sh serve-up       # local vLLM endpoint on :8001 (GPU AGENT_LLM_GPU)
./run_challenge.sh loop [N]       # N iterations of agent-run → submit (default 5)
./run_challenge.sh serve-down
```

Tests (no docker or GPU):
```
bash tests/test_preflight.sh
uv run --with httpx --with fastapi --with pytest --with pytest-asyncio pytest tests/test_client.py
```
