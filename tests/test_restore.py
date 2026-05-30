from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmca.restore import ResumeError, restore_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


def _minimal_log(tmp_path: Path) -> Path:
    """Minimum valid log: system_prompt + one user/assistant exchange."""
    p = tmp_path / "chat_2025-01-01_00-00-00.jsonl"
    _write_jsonl(p, [
        {"type": "system_prompt", "content": "You are helpful."},
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "hello", "attachments": []},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "hi there"},
    ])
    return p


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_load_resume_raises_on_missing_file(tmp_path):
    with pytest.raises(ResumeError, match="not found"):
        restore_session(tmp_path / "nonexistent.jsonl")


def test_load_resume_raises_on_malformed_json(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    p.write_text("not json\n")
    with pytest.raises(ResumeError, match="line 1"):
        restore_session(p)


def test_load_resume_reports_all_malformed_line_numbers(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    p.write_text(
        json.dumps({"type": "system_prompt", "content": "sp"}) + "\n"
        + "bad line\n"
        + json.dumps({"type": "exchange", "timestamp": "t", "role": "user", "content": "hi", "rag_chunks": [], "attachments": []}) + "\n"
        + json.dumps({"type": "exchange", "timestamp": "t", "role": "assistant", "content": "reply"}) + "\n"
        + "another bad\n"
    )
    with pytest.raises(ResumeError) as exc_info:
        restore_session(p)
    msg = str(exc_info.value)
    assert "line 2" in msg
    assert "line 5" in msg


def test_load_resume_raises_on_zero_valid_turns(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [{"type": "system_prompt", "content": "sp"}])
    with pytest.raises(ResumeError, match="(?i)no.*turn"):
        restore_session(p)


def test_load_resume_raises_if_no_system_prompt_entry(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "hi", "rag_chunks": [], "attachments": []},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "ok"},
    ])
    with pytest.raises(ResumeError, match="(?i)system.?prompt"):
        restore_session(p)


# ---------------------------------------------------------------------------
# system_prompt
# ---------------------------------------------------------------------------

def test_load_resume_returns_system_prompt(tmp_path):
    p = _minimal_log(tmp_path)
    result = restore_session(p)
    assert result.system_prompt == "You are helpful."


# ---------------------------------------------------------------------------
# startup_docs
# ---------------------------------------------------------------------------

def test_load_resume_returns_startup_docs(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"type": "system_prompt", "content": "sp"},
        {"type": "startup_doc", "path": "/docs/a.md", "content": "# A"},
        {"type": "startup_doc", "path": "/docs/b.md", "content": "# B"},
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "hi", "rag_chunks": [], "attachments": []},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "ok"},
    ])
    result = restore_session(p)
    assert result.startup_docs == [(Path("/docs/a.md"), "# A"), (Path("/docs/b.md"), "# B")]


def test_load_resume_returns_empty_startup_docs_when_none(tmp_path):
    p = _minimal_log(tmp_path)
    result = restore_session(p)
    assert result.startup_docs == []


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------

def test_load_resume_returns_history_with_role_and_content(tmp_path):
    p = _minimal_log(tmp_path)
    result = restore_session(p)
    assert result.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_load_resume_preserves_turn_order(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"type": "system_prompt", "content": "sp"},
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "q1", "rag_chunks": [], "attachments": []},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "a1"},
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "q2", "rag_chunks": [], "attachments": []},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "a2"},
    ])
    result = restore_session(p)
    assert [m["content"] for m in result.history] == ["q1", "a1", "q2", "a2"]


def test_load_resume_skips_non_exchange_entries_in_history(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"type": "system_prompt", "content": "sp"},
        {"type": "startup_doc", "path": "/d.md", "content": "doc"},
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "hi", "rag_chunks": [], "attachments": []},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "hello"},
    ])
    result = restore_session(p)
    assert len(result.history) == 2
    assert result.history[0]["role"] == "user"


# ---------------------------------------------------------------------------
# last_assistant_message
# ---------------------------------------------------------------------------

def test_load_resume_returns_last_assistant_message(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"type": "system_prompt", "content": "sp"},
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "q1", "rag_chunks": [], "attachments": []},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "first answer"},
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "q2", "rag_chunks": [], "attachments": []},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "final answer"},
    ])
    result = restore_session(p)
    assert result.last_assistant_message == "final answer"


# ---------------------------------------------------------------------------
# session_attachments
# ---------------------------------------------------------------------------

def test_load_resume_returns_session_attachments(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"type": "system_prompt", "content": "sp"},
        {
            "type": "exchange", "timestamp": "t", "role": "user", "content": "see CONTEXT_1",
            "rag_chunks": [],
            "attachments": [{"identifier": "CONTEXT_1", "path": "/src/foo.py", "content": "def foo(): pass", "size_warning": False}],
        },
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "ok"},
    ])
    result = restore_session(p)
    assert len(result.session_attachments) == 1
    att = result.session_attachments[0]
    assert att.identifier == "CONTEXT_1"
    assert att.path == Path("/src/foo.py")
    assert att.content == "def foo(): pass"
    assert att.size_warning is False


def test_load_resume_deduplicates_session_attachments_by_identifier(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    att = {"identifier": "CONTEXT_1", "path": "/src/foo.py", "content": "x = 1", "size_warning": False}
    _write_jsonl(p, [
        {"type": "system_prompt", "content": "sp"},
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "t1", "rag_chunks": [], "attachments": [att]},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "r1"},
        {"type": "exchange", "timestamp": "t", "role": "user", "content": "t2", "rag_chunks": [], "attachments": [att]},
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "r2"},
    ])
    result = restore_session(p)
    assert len(result.session_attachments) == 1


def test_load_resume_session_attachments_empty_when_none(tmp_path):
    p = _minimal_log(tmp_path)
    result = restore_session(p)
    assert result.session_attachments == []


# ---------------------------------------------------------------------------
# jsonl_path and next_attachment_n
# ---------------------------------------------------------------------------

def test_load_resume_returns_jsonl_path(tmp_path):
    p = _minimal_log(tmp_path)
    result = restore_session(p)
    assert result.jsonl_path == p


def test_load_resume_next_attachment_n_continues_from_max(tmp_path):
    p = tmp_path / "chat_ts.jsonl"
    _write_jsonl(p, [
        {"type": "system_prompt", "content": "sp"},
        {
            "type": "exchange", "timestamp": "t", "role": "user", "content": "hi",
            "rag_chunks": [],
            "attachments": [
                {"identifier": "CONTEXT_1", "path": "/a.py", "content": "x", "size_warning": False},
                {"identifier": "CONTEXT_2", "path": "/b.py", "content": "y", "size_warning": False},
            ],
        },
        {"type": "exchange", "timestamp": "t", "role": "assistant", "content": "ok"},
    ])
    result = restore_session(p)
    assert result.next_attachment_n == 3


def test_load_resume_next_attachment_n_is_1_when_no_attachments(tmp_path):
    p = _minimal_log(tmp_path)
    result = restore_session(p)
    assert result.next_attachment_n == 1


# ---------------------------------------------------------------------------
# No resumed_context field
# ---------------------------------------------------------------------------

def test_load_resume_result_has_no_resumed_context_field(tmp_path):
    p = _minimal_log(tmp_path)
    result = restore_session(p)
    assert not hasattr(result, "resumed_context")


def test_load_resume_result_has_no_session_rag_chunks_field(tmp_path):
    p = _minimal_log(tmp_path)
    result = restore_session(p)
    assert not hasattr(result, "session_rag_chunks")
