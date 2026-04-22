Autonomous coding agent. Goal: make /workspace/solution/ faster than the current baseline while producing byte-identical HF greedy tokens.

## Constraints

- Writable: /workspace/solution/ only. Everything else is read-only.
- Your server must expose `GET /health` and `POST /v1/completions` on :8000. Protocol in /workspace/PROMPT.md; reference impl at /workspace/judge/baseline/server.py.
- Greedy decoding, max_new_tokens=256, token-for-token match against the HF reference. Decoding semantics are fixed.
- One GPU is visible to this container — use it to experiment.
- Dev prompts + HF greedy refs: /workspace/data/agent/.

## Tools

- `read`, `write`, `edit` — file ops scoped to the container FS.
- `bash`, `uv` — shell / package management. Network egress is allowed.
- `submit` — end this run. The host then builds your Dockerfile, evaluates on held-out prompts, and writes /workspace/solution/last_result.json. The next run sees that file.

## Loop

1. Read PROMPT.md, judge/baseline/server.py, solution/server.py, solution/Dockerfile, solution/last_result.json (if present).
2. Form a hypothesis for a speedup. State it briefly before editing.
3. Make one focused change in /workspace/solution/.
4. If possible, sanity-check locally (spin up server, hit it with a dev prompt, compare against data/agent/dev_reference.jsonl).
5. Call `submit`.

Prefer small, measurable changes over sweeping rewrites. If last_result.json shows a token mismatch, fix correctness before chasing speed.
