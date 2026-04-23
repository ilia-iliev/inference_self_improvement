#!/usr/bin/env python3
"""Agent loop: OpenAI-compatible chat completions + tool dispatch.

Runs inside the agent sandbox container. Exits when the model calls `submit`
(marker file written) or stops producing tool calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import BadRequestError, OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_tools import SUBMIT_MARKER, TOOL_SCHEMAS, dispatch  # noqa: E402

WORKSPACE = Path("/workspace")
SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "system_prompt.md"
DEFAULT_MAX_TURNS = 200
MAX_NUDGES = 3
# Keep system + initial user msg + this many most-recent messages.
CONTEXT_WINDOW_MSGS = 40
# After this many consecutive all-stub turns, force a submit.
STUB_LOOP_LIMIT = 3

NUDGE_MSG = (
    "You described your next step but did not call a tool. "
    "Please proceed immediately with a tool call."
)

FORCE_SUBMIT_MSG = (
    "You have re-read all files multiple times without making changes. "
    "Your solution already exists at /workspace/solution/. "
    "Stop reading and call submit() now to evaluate what you have."
)


def build_user_message() -> str:
    parts = [
        "# Task",
        (WORKSPACE / "PROMPT.md").read_text(),
        "",
        "# Layout",
        "- /workspace/solution/              (rw, your workspace)",
        "- /workspace/solution/last_result.json  (rw, last eval result — read at /workspace/solution/last_result.json)",
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


def _all_stubs(results: list[str]) -> bool:
    """Return True if every tool result this turn was non-productive (stubs or read errors)."""
    if not results:
        return False
    return all("already read" in r or '"error"' in r for r in results)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("AGENT_LLM_URL", "http://localhost:8001/v1"))
    ap.add_argument("--model", default=os.environ.get("AGENT_LLM_MODEL", "google/gemma-4-e4b-it"))
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("AGENT_MAX_TURNS", DEFAULT_MAX_TURNS)))
    ap.add_argument("--max-tokens", type=int, default=2048)
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

    window = CONTEXT_WINDOW_MSGS

    def trim(msgs: list, n: int) -> list:
        if len(msgs) <= 2 + n:
            return msgs
        trimmed = msgs[:2] + msgs[-n:]
        print(f"[agent] context trimmed to {len(trimmed)} messages (window={n})",
              file=sys.stderr, flush=True)
        return trimmed

    nudges = 0
    stub_turns = 0  # consecutive turns where all tool results were read-cache stubs

    for turn in range(args.max_turns):
        print(f"\n[turn {turn}] → model", file=sys.stderr, flush=True)

        # After too many all-stub turns, force a submit call.
        if stub_turns >= STUB_LOOP_LIMIT:
            print(f"[agent] stub loop detected — forcing submit", file=sys.stderr, flush=True)
            messages.append({"role": "user", "content": FORCE_SUBMIT_MSG})
            stub_turns = 0
            forced_tc = "required"
            forced_choice: object = {"type": "function", "function": {"name": "submit"}}
        else:
            forced_choice = "required"

        # Retry with progressively smaller window on context-length overflow.
        w = window
        while True:
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=trim(messages, w),
                    tools=TOOL_SCHEMAS,
                    tool_choice=forced_choice,
                    max_tokens=args.max_tokens,
                )
                break
            except BadRequestError as e:
                if w > 6:
                    w = max(6, w // 2)
                    print(f"[agent] BadRequest — retrying with window={w}: {e}",
                          file=sys.stderr, flush=True)
                else:
                    print(f"[agent] BadRequest at min window — exiting gracefully: {e}",
                          file=sys.stderr, flush=True)
                    return 0

        if w < window:
            window = w
            print(f"[agent] window permanently reduced to {window}",
                  file=sys.stderr, flush=True)

        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if msg.content:
            print(f"[turn {turn}] say: {fmt_preview(msg.content)}", file=sys.stderr, flush=True)

        if not msg.tool_calls:
            if nudges < MAX_NUDGES:
                nudges += 1
                print(f"[turn {turn}] no tool calls (nudge {nudges}/{MAX_NUDGES})",
                      file=sys.stderr, flush=True)
                messages.append({"role": "user", "content": NUDGE_MSG})
                continue
            print(f"[turn {turn}] no tool calls after {MAX_NUDGES} nudges — stopping.",
                  file=sys.stderr, flush=True)
            return 0

        nudges = 0

        turn_results: list[str] = []
        for tc in msg.tool_calls:
            name = tc.function.name
            arguments = tc.function.arguments or "{}"
            print(f"[turn {turn}] call: {name}({fmt_preview(arguments)})",
                  file=sys.stderr, flush=True)
            result = dispatch(name, arguments)
            print(f"[turn {turn}]  -> {fmt_preview(result)}", file=sys.stderr, flush=True)
            turn_results.append(result)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        if _all_stubs(turn_results):
            stub_turns += 1
            print(f"[agent] all-stub turn ({stub_turns}/{STUB_LOOP_LIMIT})",
                  file=sys.stderr, flush=True)
        else:
            stub_turns = 0

        if SUBMIT_MARKER.exists():
            print(f"[turn {turn}] submit — exiting for host handoff.", file=sys.stderr, flush=True)
            return 0

    print(f"[agent] hit max_turns={args.max_turns} without submit.", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
