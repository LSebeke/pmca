import json
from pathlib import Path

import pytest

from pmca.logger import SessionLogger
from pmca.types import Attachment, Chunk


def _chunk(tmp_path: Path) -> Chunk:
    return Chunk(content="def foo(): pass", source_file=tmp_path / "code.py", label="function `foo` (lines 1–1)")


def _attachment(tmp_path: Path) -> Attachment:
    return Attachment(path=tmp_path / "secret.py", content="code", identifier="CONTEXT_1", size_warning=False)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# log_exchange — structure
# ---------------------------------------------------------------------------

def test_log_exchange_writes_two_lines(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("Hello", "Hi there", [], [])
    logger.close()

    lines = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert len(lines) == 2


def test_log_exchange_roles(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("Hello", "Hi there", [], [])
    logger.close()

    user, asst = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert user["role"] == "user"
    assert asst["role"] == "assistant"


def test_log_exchange_content(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("user msg", "asst msg", [], [])
    logger.close()

    user, asst = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert user["content"] == "user msg"
    assert asst["content"] == "asst msg"


def test_user_entry_includes_rag_chunks(tmp_path):
    chunk = _chunk(tmp_path)
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("msg", "resp", [chunk], [])
    logger.close()

    user, _ = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert "rag_chunks" in user
    assert len(user["rag_chunks"]) == 1
    assert user["rag_chunks"][0]["label"] == chunk.label
    assert user["rag_chunks"][0]["content"] == chunk.content
    assert "source" in user["rag_chunks"][0]


def test_user_entry_includes_attachments(tmp_path):
    att = _attachment(tmp_path)
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("msg", "resp", [], [att])
    logger.close()

    user, _ = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert "attachments" in user
    assert len(user["attachments"]) == 1
    assert user["attachments"][0]["identifier"] == "CONTEXT_1"
    assert user["attachments"][0]["size_warning"] is False


def test_assistant_entry_has_no_rag_or_attachments(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("msg", "resp", [_chunk(tmp_path)], [_attachment(tmp_path)])
    logger.close()

    _, asst = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert "rag_chunks" not in asst
    assert "attachments" not in asst


def test_entries_have_timestamp_field(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("msg", "resp", [], [])
    logger.close()

    for entry in _read_jsonl(tmp_path / "chat_ts.jsonl"):
        assert "timestamp" in entry


# ---------------------------------------------------------------------------
# log_debug
# ---------------------------------------------------------------------------

def test_log_debug_writes_to_log_file(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_debug("something went wrong")
    logger.close()

    content = (tmp_path / "debug_ts.log").read_text()
    assert "something went wrong" in content


def test_log_debug_line_includes_timestamp(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_debug("test message")
    logger.close()

    line = (tmp_path / "debug_ts.log").read_text().strip()
    assert line.startswith("[")


# ---------------------------------------------------------------------------
# Flush (data visible without close)
# ---------------------------------------------------------------------------

def test_jsonl_flushed_immediately(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("msg", "resp", [], [])
    # no close() yet
    lines = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert len(lines) == 2
    logger.close()


def test_log_flushed_immediately(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_debug("hello")
    content = (tmp_path / "debug_ts.log").read_text()
    assert "hello" in content
    logger.close()


# ---------------------------------------------------------------------------
# Append (second logger reuses files)
# ---------------------------------------------------------------------------

def test_second_logger_appends_not_overwrites(tmp_path):
    SessionLogger(tmp_path, "ts").log_exchange("first", "r1", [], [])
    # don't close — simulate crash; reopen same files
    logger2 = SessionLogger(tmp_path, "ts")
    logger2.log_exchange("second", "r2", [], [])
    logger2.close()

    lines = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert len(lines) == 4
