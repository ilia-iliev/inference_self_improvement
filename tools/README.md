# Agent Tool Surface

The allowlist in `settings.json` is the source of truth. This file explains *why* each category is allowed or denied.

## Allowed

- **File I/O inside `solution/`** — the agent's working directory. Read/Write/Edit freely.
- **Read-only everywhere else** — so the agent can inspect the baseline, dev prompts, harness code.
- **`uv run` / `uv pip` / `uv add`** — test code with real torch/transformers on the GPU before baking into the Dockerfile.
- **`python3`** — same, for one-off probes.
- **`nvidia-smi`** — inspect GPU, VRAM, driver.
- **`docker build`** — iterate on the Dockerfile locally without the full submit cost.
- **`./run_challenge.sh`** — the official build+run+score path.

## Denied

- **Writes outside `solution/`** — the judge, baseline, and data are the invariants. If the agent could edit them, "score" becomes meaningless.
- **Reads into `data/judge/`** — held-out eval set. Leaking it would let the agent tune to the exact prompts.
- **`docker run` / `docker exec`** — launching containers directly sidesteps the judge's measurement protocol. Only the harness runs containers.
- **`sudo`, `rm -rf`** — unnecessary for the task; block the obvious footguns.
