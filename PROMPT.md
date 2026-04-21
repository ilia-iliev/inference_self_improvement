# Task: Beat the Baseline

Maximize inference throughput on this host's GPU(s). You are given a working reference implementation ("baseline"). Ship a faster one that produces the **same tokens**, wrapped in a Docker image that serves the protocol below.

## Constraints

- **Hardware.** NVIDIA GPU(s), CUDA 12.8. The model must live on CUDA.
- **Decoding.** Greedy, `max_new_tokens=256`, token-for-token identical to reference
- **Batching.** The judge fires all eval prompts concurrently against your server. Timing is wall clock from first POST to last response.
- **Deliverable.** `solution/Dockerfile` + whatever it references. The built image must serve `:8000`.

## Protocol (your server must implement both endpoints)

- `GET /health` → `200 {"status": "ready"|"loading"}`. Judge polls until `ready` (5-min startup cap).
- `POST /v1/completions` body `{"id", "modality" ∈ {text,image,audio}, "prompt", "image"?, "audio"?}` → `200 {"id", "completion"}`. `image`/`audio` are paths under the read-only `/data` mount inside the container.

## Scoring

```
any token mismatch             → 0
agent_time ≥ baseline_time     → 0
else                           → baseline_time / agent_time
```

To be accepted (promoted to the new baseline) you need `score ≥ 1.05` and all tokens match.

## Current baseline

- Code: `judge/baseline/server.py` (plain HF greedy in FastAPI)
- Latest timing: `judge/baseline.json`

The judge re-runs the baseline on the same host whenever it measures you, so timing is apples-to-apples.

## Available data

- `data/agent/dev_prompts.jsonl` — 100 prompts you can use freely
- `data/agent/dev_reference.jsonl` — HF greedy outputs for those prompts (ground truth for local testing)
- `data/agent/assets/` — media for image/audio prompts

The eval set under `data/judge/` is held out from you.

## Iterate

Edit `solution/Dockerfile` and/or `solution/server.py`, then:

```
./run_challenge.sh submit   # build, run, verify, score, maybe promote
```

Result lands in `solution/last_result.json`.
