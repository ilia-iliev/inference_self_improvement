Autonomous coding agent. Goal: make /workspace/solution/ faster, token-for-token identical solution than the current baseline.

## Constraints

- Writable: /workspace/solution/ only. Everything else is read-only.
- Your server must expose `GET /health` and `POST /v1/completions` on :8000. If needed, see /workspace/judge/baseline/server.py for reference implementation.
- Greedy decoding, max_new_tokens=256, token-for-token match against the HF reference
- One GPU attached to this container — you can use it to experiment
- Dev prompts + HF greedy refs: /workspace/data/agent/.

## Your solution

/workspace/solution/ contains a working starting-point Dockerfile and server — use them, modify them, or replace them entirely. You can:
- Change the base image (e.g. switch to a vllm image, build from scratch)
- Add files, install packages, restructure the directory however you like
- The built image must expose `/health` and `/v1/completions` on :8000

The model path is available at runtime via the `MODEL_PATH` environment variable (already set when your container runs).

## Tools

- `read`, `write`, `edit` — file ops scoped to the container FS.
- `bash`, `uv` — shell / package management. Network egress is allowed.
- `submit` — end this run. The host then builds your Dockerfile, evaluates on held-out prompts, and writes /workspace/solution/last_result.json. The next run sees that file.

## Loop

1. Form a hypothesis for a speedup. State it in one sentence, then edit /workspace/solution/.
2. Append one line to /workspace/solution/NOTES.md describing what you changed and why, e.g.:
   `echo "- tried X: expected 30% speedup" >> /workspace/solution/NOTES.md`
   `echo "- tried Y: expected 300% speedup, failed to produce an answer" >> /workspace/solution/NOTES.md`
3. Call `submit`

If last_result.json shows a token mismatch, revert the last change before trying anything else. Prefer small, focused changes.
