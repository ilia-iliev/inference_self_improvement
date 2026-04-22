"""
Unit tests for tools/agent_tools.py. Pure Python, no docker, no GPU, no LLM.

Run: uv run --with pytest pytest tests/test_agent_tools.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import agent_tools  # noqa: E402
from agent_tools import (  # noqa: E402
    DISPATCH,
    MAX_OUTPUT_CHARS,
    TOOL_SCHEMAS,
    dispatch,
    t_bash,
    t_edit,
    t_read,
    t_submit,
    t_uv,
    t_write,
)


# ---------- schemas ----------

def test_schemas_cover_every_tool():
    schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert schema_names == set(DISPATCH.keys())


def test_schema_shape_is_openai_function_calling():
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert set(fn.keys()) >= {"name", "description", "parameters"}
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


# ---------- dispatch error paths ----------

def test_dispatch_unknown_tool_returns_error():
    r = json.loads(dispatch("nope", "{}"))
    assert "error" in r and "unknown" in r["error"]


def test_dispatch_malformed_json_returns_error():
    r = json.loads(dispatch("read", "{not json"))
    assert "error" in r and "invalid json" in r["error"]


def test_dispatch_missing_required_arg_returns_error():
    r = json.loads(dispatch("read", "{}"))
    assert "error" in r and "bad arguments" in r["error"]


def test_dispatch_empty_arguments_is_ok_for_no_arg_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_tools, "SUBMIT_MARKER", tmp_path / "marker")
    r = json.loads(dispatch("submit", ""))
    assert r.get("ok") is True


# ---------- read ----------

def test_read_returns_content(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello")
    assert t_read(str(p)) == {"content": "hello"}


def test_read_missing_returns_error(tmp_path):
    r = t_read(str(tmp_path / "nope.txt"))
    assert "error" in r


def test_read_truncates_large_files(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("x" * (MAX_OUTPUT_CHARS + 10_000))
    r = t_read(str(p))
    assert len(r["content"]) <= MAX_OUTPUT_CHARS + 200  # truncation marker overhead
    assert "truncated" in r["content"]


# ---------- write ----------

def test_write_creates_parents_and_round_trips(tmp_path):
    p = tmp_path / "a" / "b" / "c.txt"
    r = t_write(str(p), "payload")
    assert r == {"ok": True, "bytes": 7}
    assert p.read_text() == "payload"


def test_write_overwrites(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("old")
    t_write(str(p), "new")
    assert p.read_text() == "new"


# ---------- edit ----------

def test_edit_unique_match_succeeds(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("foo bar baz")
    r = t_edit(str(p), "bar", "qux")
    assert r == {"ok": True}
    assert p.read_text() == "foo qux baz"


def test_edit_missing_substring_errors(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("foo")
    r = t_edit(str(p), "bar", "qux")
    assert "error" in r and "not found" in r["error"]
    assert p.read_text() == "foo"  # unchanged


def test_edit_ambiguous_match_errors(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("aa aa aa")
    r = t_edit(str(p), "aa", "b")
    assert "error" in r and "3 times" in r["error"]
    assert p.read_text() == "aa aa aa"  # unchanged


def test_edit_missing_file_errors(tmp_path):
    r = t_edit(str(tmp_path / "nope.txt"), "x", "y")
    assert "error" in r and "not a file" in r["error"]


# ---------- bash ----------

def test_bash_captures_stdout_stderr_and_exit_code():
    r = t_bash("echo hi && echo err >&2 && exit 3")
    assert r["stdout"].strip() == "hi"
    assert r["stderr"].strip() == "err"
    assert r["exit_code"] == 3


def test_bash_timeout_returns_exit_neg1():
    r = t_bash("sleep 5", timeout=1)
    assert r["exit_code"] == -1
    assert "timed out" in r["stderr"]


def test_bash_truncates_large_stdout():
    # Emit ~2x MAX_OUTPUT_CHARS worth of output.
    n = MAX_OUTPUT_CHARS * 2
    r = t_bash(f"python3 -c \"print('x'*{n}, end='')\"")
    assert r["exit_code"] == 0
    assert "truncated" in r["stdout"]


# ---------- uv ----------

def test_uv_delegates_to_bash_with_prefix(monkeypatch):
    calls = []

    def fake_run(cmd, timeout):
        calls.append((cmd, timeout))
        return {"stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(agent_tools, "_run", fake_run)
    t_uv("add torch", timeout=42)
    assert calls == [("uv add torch", 42)]


# ---------- submit ----------

def test_submit_writes_marker(tmp_path, monkeypatch):
    marker = tmp_path / "solution" / ".submit-request"
    monkeypatch.setattr(agent_tools, "SUBMIT_MARKER", marker)
    r = t_submit()
    assert r["ok"] is True
    assert marker.exists()
