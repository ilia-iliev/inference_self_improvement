#!/usr/bin/env python3
"""Agent loop: OpenAI-compatible chat completions + tool dispatch.

Runs inside the agent sandbox container. Exits when the model calls `submit`
(marker file written) or stops producing tool calls.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_tools import SUBMIT_MARKER, TOOL_SCHEMAS, dispatch  # noqa: E402

WORKSPACE = Path("/workspace")
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.md"
DEFAULT_MAX_TURNS = 200


def build_user_message() -> str:
    parts = [
        "# Task",
        (WORKSPACE / "PROMPT.md").read_text(),
        "",
        "# Layout",
        "- /workspace/solution/              (rw, your workspace)",
        "- /workspace/judge/baseline/        (ro, reference impl to beat)",
        "- /workspace/data/agent/            (ro, dev prompts + HF greedy refs)",
        "- /workspace/PROMPT.md              (ro, full task)",
    ]
    last = WORKSPACE / "solution" / "last_result.json"
    if last.exists():
        parts += ["", "# Last eval result", last.read_text()]
    return "\n".join(parts)


def fmt_preview(s: str, n: int = 240) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("AGENT_LLM_URL", "http://localhost:8001/v1"))
    ap.add_argument("--model", default=os.environ.get("AGENT_LLM_MODEL", "google/gemma-4-e4b-it"))
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("AGENT_MAX_TURNS", DEFAULT_MAX_TURNS)))
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    if SUBMIT_MARKER.exists():
        SUBMIT_MARKER.unlink()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PATH.read_text()},
        {"role": "user", "content": build_user_message()},
    ]

    print(f"[agent] endpoint={args.base_url} model={args.model} max_turns={args.max_turns}",
          file=sys.stderr, flush=True)

    for turn in range(args.max_turns):
        print(f"\n[turn {turn}] → model", file=sys.stderr, flush=True)
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            max_tokens=args.max_tokens,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            print(f"[turn {turn}] say: {fmt_preview(msg.content)}", file=sys.stderr, flush=True)

        if not msg.tool_calls:
            print(f"[turn {turn}] no tool calls — stopping.", file=sys.stderr, flush=True)
            return 0

        for tc in msg.tool_calls:
            name = tc.function.name
            arguments = tc.function.arguments or "{}"
            print(f"[turn {turn}] call: {name}({fmt_preview(arguments)})",
                  file=sys.stderr, flush=True)
            result = dispatch(name, arguments)
            print(f"[turn {turn}]  -> {fmt_preview(result)}", file=sys.stderr, flush=True)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        if SUBMIT_MARKER.exists():
            print(f"[turn {turn}] submit — exiting for host handoff.", file=sys.stderr, flush=True)
            return 0

    print(f"[agent] hit max_turns={args.max_turns} without submit.", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
