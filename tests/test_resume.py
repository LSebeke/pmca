from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmca.resume import ResumeError, load_resume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def _minimal_log(tmp_path: Path) -> Path:
    """One user turn + one assistant turn — the minimum valid log."""
    p = tmp_path / "chat_2025-01-01_00-00-00.jsonl"
    _write_jsonl(p, [
        {"timestamp": "t", "role": "user", "content": "hello", "rag_chunks": [], "attachments": []},
        {"timestamp": "t", "role": "assistant", "content": "hi there"},
    ])
    return p


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_load_resume_raises_on_missing_file(tmp_path):
    with pytest.raises(ResumeError, match="not found"):
        load_resume(tmp_path / "nonexistent.jsonl")


def test_load_resume_raises_on_malformed_json(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    p.write_text("not json\n")
    with pytest.raises(ResumeError, match="line 1"):
        load_resume(p)


def test_load_resume_reports_all_malformed_line_numbers(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    p.write_text(
        json.dumps({"timestamp": "t", "role": "user", "content": "hi", "rag_chunks": [], "attachments": []}) + "\n"
        + "bad line\n"
        + json.dumps({"timestamp": "t", "role": "assistant", "content": "reply"}) + "\n"
        + "another bad\n"
    )
    with pytest.raises(ResumeError) as exc_info:
        load_resume(p)
    msg = str(exc_info.value)
    assert "line 2" in msg
    assert "line 4" in msg


def test_load_resume_raises_on_zero_valid_turns(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [{"timestamp": "t", "role": "system", "content": "ignore me"}])
    with pytest.raises(ResumeError, match="(?i)no.*turn"):
        load_resume(p)


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------

def test_load_resume_returns_history_with_role_and_content(tmp_path):
    p = _minimal_log(tmp_path)
    result = load_resume(p)
    assert result.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_load_resume_preserves_turn_order(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"timestamp": "t", "role": "user", "content": "q1", "rag_chunks": [], "attachments": []},
        {"timestamp": "t", "role": "assistant", "content": "a1"},
        {"timestamp": "t", "role": "user", "content": "q2", "rag_chunks": [], "attachments": []},
        {"timestamp": "t", "role": "assistant", "content": "a2"},
    ])
    result = load_resume(p)
    assert [m["content"] for m in result.history] == ["q1", "a1", "q2", "a2"]


def test_load_resume_skips_non_user_assistant_roles(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"timestamp": "t", "role": "system", "content": "sys"},
        {"timestamp": "t", "role": "user", "content": "hi", "rag_chunks": [], "attachments": []},
        {"timestamp": "t", "role": "assistant", "content": "hello"},
    ])
    result = load_resume(p)
    assert len(result.history) == 2
    assert result.history[0]["role"] == "user"


# ---------------------------------------------------------------------------
# last_assistant_message
# ---------------------------------------------------------------------------

def test_load_resume_returns_last_assistant_message(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"timestamp": "t", "role": "user", "content": "q1", "rag_chunks": [], "attachments": []},
        {"timestamp": "t", "role": "assistant", "content": "first answer"},
        {"timestamp": "t", "role": "user", "content": "q2", "rag_chunks": [], "attachments": []},
        {"timestamp": "t", "role": "assistant", "content": "final answer"},
    ])
    result = load_resume(p)
    assert result.last_assistant_message == "final answer"


# ---------------------------------------------------------------------------
# resumed_context — attachments
# ---------------------------------------------------------------------------

def test_load_resume_includes_attachment_in_resumed_context(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {
            "timestamp": "t", "role": "user", "content": "see CONTEXT_1",
            "rag_chunks": [],
            "attachments": [{"identifier": "CONTEXT_1", "path": "/src/foo.py", "content": "def foo(): pass", "size_warning": False}],
        },
        {"timestamp": "t", "role": "assistant", "content": "ok"},
    ])
    result = load_resume(p)
    assert "[RESUMED_CONTEXT]" in result.resumed_context
    assert "CONTEXT_1" in result.resumed_context
    assert "def foo(): pass" in result.resumed_context


def test_load_resume_deduplicates_attachments_by_identifier(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    att = {"identifier": "CONTEXT_1", "path": "/src/foo.py", "content": "x = 1", "size_warning": False}
    _write_jsonl(p, [
        {"timestamp": "t", "role": "user", "content": "turn1", "rag_chunks": [], "attachments": [att]},
        {"timestamp": "t", "role": "assistant", "content": "r1"},
        {"timestamp": "t", "role": "user", "content": "turn2", "rag_chunks": [], "attachments": [att]},
        {"timestamp": "t", "role": "assistant", "content": "r2"},
    ])
    result = load_resume(p)
    assert result.resumed_context.count("CONTEXT_1") == 1


# ---------------------------------------------------------------------------
# resumed_context — RAG chunks
# ---------------------------------------------------------------------------

def test_load_resume_includes_rag_chunk_in_resumed_context(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {
            "timestamp": "t", "role": "user", "content": "hi",
            "rag_chunks": [{"label": "fn `foo`", "source": "/src/a.py", "content": "def foo(): pass"}],
            "attachments": [],
        },
        {"timestamp": "t", "role": "assistant", "content": "ok"},
    ])
    result = load_resume(p)
    assert "fn `foo`" in result.resumed_context
    assert "def foo(): pass" in result.resumed_context


def test_empty_resumed_context_when_no_attachments_or_rag(tmp_path):
    p = _minimal_log(tmp_path)
    result = load_resume(p)
    assert result.resumed_context == ""


# ---------------------------------------------------------------------------
# jsonl_path
# ---------------------------------------------------------------------------

def test_load_resume_returns_jsonl_path(tmp_path):
    p = _minimal_log(tmp_path)
    result = load_resume(p)
    assert result.jsonl_path == p
