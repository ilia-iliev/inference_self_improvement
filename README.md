This is an example implementation for the challenge below. My home setup is two 3090s - one reserved for the agent loop and another for iterating on solutions. I constrained the iteration to dockerfiles.

I used gemma4-e4b (full weights) for both the agent loop and the inference improvement. The system does loop through hypotheses but I don't think the model is powerful enough to beat the baseline. Nevertheless, it was a great learniong experience.

I used a coding assistant (claude code)


# Inference Throughput Challenge

A self-improving eval harness. An agent iteratively edits `solution/`, the harness builds it, times it against a baseline on held-out prompts, and promotes the solution to baseline when it's fast enough and outputs are still bit-exact.

## Layout

```
PROMPT.md              agent-facing problem statement
solution/              agent's workspace (seeded from judge/baseline/ via init-solution)
judge/                 harness
  baseline/            reference impl (server.py + inference.py share _common.py)
  docker/              base image, compose, .env
  submit.py            build → run → verify → score → promote
  client.py            HTTP driver
  baseline.json        current baseline timing (auto-generated on first submit)
data/
  agent/               dev prompts + refs + assets (visible)
  judge/               eval prompts + refs + assets (held out)
tools/                 agent sandbox image + runner
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
else                           → 1
```

Promotion gate: `score ≥ 1.05` and all outputs match → solution image is retagged as the new baseline and re-measured. Persisted in `judge/baseline.json`.

## Host requirements

- ≥1 NVIDIA GPU, driver ≥570 (CUDA 12.8)
- Docker + compose + `nvidia-container-toolkit`
- Target model in the local HF cache or on disk (set via `MODEL_PATH` in `judge/docker/.env`)

## Configuration

Copy `judge/docker/.env.example` → `judge/docker/.env`:

- `MODEL_PATH` — absolute path to the model directory (must be reachable inside containers via the disk-level volume mount). Modality (text-only vs image/audio) is detected automatically from the model's `config.json`.

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
                                  # first run seeds solution/ from judge/baseline/
                                  # and initializes baseline timing
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
