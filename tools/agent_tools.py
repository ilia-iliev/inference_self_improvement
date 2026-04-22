"""Tool implementations + OpenAI function schemas for the agent runner.

Pure stdlib, no imports from `tools` (to avoid name clash with this dir).
Safe to unit-test on the host without docker — every handler operates on
plain files / subprocess, with scoping enforced by the container's mounts.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

DEFAULT_TIMEOUT = 300
MAX_OUTPUT_CHARS = 16000  # keep tool results from blowing up context

SUBMIT_MARKER = Path("/workspace/solution/.submit-request")

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the full contents of a file as text.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Overwrite a file. Parent dirs auto-created.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Replace the unique occurrence of old_string with new_string in the file. "
                "Fails if old_string is missing or appears more than once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a shell command. Returns stdout, stderr, exit_code. "
                f"Default timeout {DEFAULT_TIMEOUT}s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "uv",
            "description": (
                "Shortcut for `uv <args>`. Use for package mgmt and running scripts "
                "(e.g. args='add torch', args='run script.py')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "args": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["args"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Signal that the current /workspace/solution/ is ready for the host-side "
                "evaluator. Writes a marker file; the runner exits; the host then runs "
                "`./run_challenge.sh submit` and updates solution/last_result.json for "
                "the next iteration."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _run(cmd: str, timeout: int | None) -> dict:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout or DEFAULT_TIMEOUT,
        )
        return {
            "stdout": _truncate(r.stdout),
            "stderr": _truncate(r.stderr),
            "exit_code": r.returncode,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "stdout": _truncate((e.stdout or b"").decode(errors="replace")),
            "stderr": _truncate((e.stderr or b"").decode(errors="replace")
                                + f"\n[timed out after {timeout or DEFAULT_TIMEOUT}s]"),
            "exit_code": -1,
        }


def _truncate(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    head = MAX_OUTPUT_CHARS // 2
    tail = MAX_OUTPUT_CHARS - head
    return s[:head] + f"\n...[truncated {len(s) - MAX_OUTPUT_CHARS} chars]...\n" + s[-tail:]


def t_read(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {"error": f"not a file: {path}"}
    return {"content": _truncate(p.read_text())}


def t_write(path: str, content: str) -> dict:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"ok": True, "bytes": len(content)}


def t_edit(path: str, old_string: str, new_string: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {"error": f"not a file: {path}"}
    src = p.read_text()
    count = src.count(old_string)
    if count == 0:
        return {"error": "old_string not found"}
    if count > 1:
        return {"error": f"old_string matches {count} times; expand it until unique"}
    p.write_text(src.replace(old_string, new_string, 1))
    return {"ok": True}


def t_bash(command: str, timeout: int | None = None) -> dict:
    return _run(command, timeout)


def t_uv(args: str, timeout: int | None = None) -> dict:
    return _run(f"uv {args}", timeout)


def t_submit() -> dict:
    SUBMIT_MARKER.parent.mkdir(parents=True, exist_ok=True)
    SUBMIT_MARKER.write_text("")
    return {"ok": True, "note": "submit marker written; runner will exit"}


DISPATCH = {
    "read": t_read,
    "write": t_write,
    "edit": t_edit,
    "bash": t_bash,
    "uv": t_uv,
    "submit": t_submit,
}


def dispatch(name: str, arguments_json: str) -> str:
    """Route a tool call. Returns a JSON-serialized result string."""
    try:
        kwargs = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid json arguments: {e}"})
    fn = DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        return json.dumps(fn(**kwargs))
    except TypeError as e:
        return json.dumps({"error": f"bad arguments for {name}: {e}"})
