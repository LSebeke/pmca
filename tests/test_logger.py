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


def test_user_entry_attachment_includes_content(tmp_path):
    att = _attachment(tmp_path)
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("msg", "resp", [], [att])
    logger.close()

    user, _ = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert user["attachments"][0]["content"] == att.content


# ---------------------------------------------------------------------------
# SessionLogger.from_existing
# ---------------------------------------------------------------------------

def test_from_existing_appends_to_existing_jsonl(tmp_path):
    jsonl = tmp_path / "chat_ts.jsonl"
    SessionLogger(tmp_path, "ts").log_exchange("first", "r1", [], [])

    logger2 = SessionLogger.from_existing(jsonl)
    logger2.log_exchange("second", "r2", [], [])
    logger2.close()

    lines = _read_jsonl(jsonl)
    assert len(lines) == 4


def test_from_existing_infers_debug_log_path(tmp_path):
    jsonl = tmp_path / "chat_2025-01-01_12-00-00.jsonl"
    SessionLogger(tmp_path, "2025-01-01_12-00-00").log_exchange("x", "y", [], [])

    logger2 = SessionLogger.from_existing(jsonl)
    logger2.log_debug("resumed debug")
    logger2.close()

    debug_log = tmp_path / "debug_2025-01-01_12-00-00.log"
    assert debug_log.exists()
    assert "resumed debug" in debug_log.read_text()


def test_from_existing_appends_to_existing_debug_log(tmp_path):
    jsonl = tmp_path / "chat_ts.jsonl"
    logger1 = SessionLogger(tmp_path, "ts")
    logger1.log_debug("original entry")
    logger1.close()

    logger2 = SessionLogger.from_existing(jsonl)
    logger2.log_debug("resumed entry")
    logger2.close()

    debug_log = tmp_path / "debug_ts.log"
    content = debug_log.read_text()
    assert "original entry" in content
    assert "resumed entry" in content


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
# log_session_start
# ---------------------------------------------------------------------------

def test_log_session_start_writes_system_prompt_entry(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_session_start("You are a pirate.", [])
    logger.close()

    lines = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert len(lines) == 1
    entry = lines[0]
    assert entry["type"] == "system_prompt"
    assert entry["content"] == "You are a pirate."


def test_log_session_start_writes_startup_doc_entries(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_session_start("prompt", [(Path("/docs/a.md"), "content a"), (Path("/docs/b.md"), "content b")])
    logger.close()

    lines = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert len(lines) == 3
    assert lines[0]["type"] == "system_prompt"
    assert lines[1] == {"type": "startup_doc", "path": "/docs/a.md", "content": "content a"}
    assert lines[2] == {"type": "startup_doc", "path": "/docs/b.md", "content": "content b"}


def test_log_session_start_no_startup_docs_writes_one_entry(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_session_start("prompt", [])
    logger.close()

    lines = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert len(lines) == 1


def test_log_exchange_entries_have_type_exchange(tmp_path):
    logger = SessionLogger(tmp_path, "ts")
    logger.log_exchange("hello", "hi", [], [])
    logger.close()

    user, asst = _read_jsonl(tmp_path / "chat_ts.jsonl")
    assert user["type"] == "exchange"
    assert asst["type"] == "exchange"


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
