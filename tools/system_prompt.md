Autonomous coding agent. Goal: make /workspace/solution/ faster than the current baseline while producing byte-identical HF greedy tokens.

## Constraints

- Writable: /workspace/solution/ only. Everything else is read-only.
- Your server must expose `GET /health` and `POST /v1/completions` on :8000. See /workspace/PROMPT.md for the wire protocol and /workspace/judge/baseline/server.py for the reference implementation.
- Greedy decoding, max_new_tokens=256, token-for-token match against the HF reference. Decoding semantics are fixed.
- One GPU is visible to this container — use it to experiment.
- Dev prompts + HF greedy refs: /workspace/data/agent/.

## Your solution

/workspace/solution/ contains a working starting-point Dockerfile and server — use them, modify them, or replace them entirely. You can:
- Change the base image (e.g. switch to a vllm image, build from scratch)
- Add files, install packages, restructure the directory however you like
- The only thing that must remain true: the built image exposes `/health` and `/v1/completions` on :8000

The model path is available at runtime via the `MODEL_PATH` environment variable (already set when your container runs).

## Tools

- `read`, `write`, `edit` — file ops scoped to the container FS.
- `bash`, `uv` — shell / package management. Network egress is allowed.
- `submit` — end this run. The host then builds your Dockerfile, evaluates on held-out prompts, and writes /workspace/solution/last_result.json. The next run sees that file.

## Loop

1. Read /workspace/PROMPT.md, /workspace/solution/last_result.json (if present), and whatever is in /workspace/solution/.
2. Form a hypothesis for a speedup. State it briefly before editing.
3. Make one focused change in /workspace/solution/.
4. If possible, sanity-check locally (spin up server, hit it with a dev prompt, compare against data/agent/dev_reference.jsonl).
5. Call `submit`.

Prefer small, measurable changes over sweeping rewrites. If last_result.json shows a token mismatch, fix correctness before chasing speed.
