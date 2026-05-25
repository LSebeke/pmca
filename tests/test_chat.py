from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from pmca.chat import ChatSession
from pmca.config import Config
from pmca.types import Attachment, Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides) -> Config:
    defaults = dict(
        name="test", model="gpt-4o-mini", system_prompt="You are helpful.",
        rag_files=[], top_k_chunks=3, log_folder=Path("/tmp/logs"),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _chunk(label: str = "fn `foo`") -> Chunk:
    return Chunk(content="def foo(): pass", source_file=Path("/a.py"), label=label)


def _attachment(identifier: str = "CONTEXT_1") -> Attachment:
    return Attachment(
        path=Path("/secret.py"), content="secret code",
        identifier=identifier, size_warning=False,
    )


def _make_session(config=None, *, unsafe=False):
    cfg = config or _config()
    store = MagicMock()
    store.query.return_value = []
    logger = MagicMock()
    return ChatSession(config=cfg, store=store, logger=logger, unsafe=unsafe), store, logger


# ---------------------------------------------------------------------------
# process() — RAG query
# ---------------------------------------------------------------------------

def test_process_calls_store_query(capsys):
    session, store, _ = _make_session()
    store.query.return_value = []

    with patch("pmca.chat.chat_completion", return_value="reply"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hello world")

    store.query.assert_called_once_with("hello world", session.top_k)


def test_process_stores_rag_chunks(capsys):
    session, store, _ = _make_session()
    chunks = [_chunk()]
    store.query.return_value = chunks

    with patch("pmca.chat.chat_completion", return_value="reply"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    assert session._last_rag_chunks == chunks


# ---------------------------------------------------------------------------
# process() — message assembly
# ---------------------------------------------------------------------------

def test_process_sends_system_prompt_first():
    session, store, _ = _make_session()

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    assert messages[0] == {"role": "system", "content": "You are helpful."}


def test_process_includes_rag_system_message_when_chunks_retrieved():
    session, store, _ = _make_session()
    store.query.return_value = [_chunk("fn `foo`")]

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    rag_msg = messages[1]
    assert rag_msg["role"] == "system"
    assert "[RAG_1]" in rag_msg["content"]
    assert "fn `foo`" in rag_msg["content"]


def test_process_omits_rag_message_when_no_chunks():
    session, store, _ = _make_session()
    store.query.return_value = []

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    assert not any("[RAG" in m.get("content", "") for m in messages)


def test_process_includes_attachment_system_messages():
    session, store, _ = _make_session()
    att = _attachment("CONTEXT_1")

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[att.path]):
            with patch("pmca.chat.resolve_attachments", return_value=[att]):
                with patch("pmca.chat.substitute_identifiers", return_value="hi"):
                    session.process("hi")

    messages = mock_cc.call_args[0][0]
    att_msg = next(m for m in messages if "[CONTEXT_1]" in m.get("content", ""))
    assert att_msg["role"] == "system"
    assert "secret code" in att_msg["content"]


def test_process_omits_attachment_messages_when_none():
    session, store, _ = _make_session()

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    assert not any("[CONTEXT" in m.get("content", "") for m in messages)


def test_startup_docs_appear_after_system_prompt_before_rag():
    doc_path = Path("/fake/framework.md")
    cfg = _config(startup_docs=[(doc_path, "# Framework")])
    session, store, _ = _make_session(cfg)
    store.query.return_value = [_chunk()]

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful."
    assert "[STARTUP_DOC]" in messages[1]["content"]
    assert "/fake/framework.md" in messages[1]["content"]
    assert "# Framework" in messages[1]["content"]
    assert "[RAG_1]" in messages[2]["content"]


def test_each_startup_doc_is_separate_system_message():
    doc1 = Path("/fake/doc1.md")
    doc2 = Path("/fake/doc2.md")
    cfg = _config(startup_docs=[(doc1, "content one"), (doc2, "content two")])
    session, store, _ = _make_session(cfg)

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    startup_messages = [m for m in messages if "[STARTUP_DOC]" in m.get("content", "")]
    assert len(startup_messages) == 2
    assert "content one" in startup_messages[0]["content"]
    assert "content two" in startup_messages[1]["content"]


def test_no_startup_doc_messages_when_startup_docs_empty():
    session, store, _ = _make_session()

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    assert not any("[STARTUP_DOC]" in m.get("content", "") for m in messages)


def test_process_message_order_system_rag_attachment_history_user():
    session, store, _ = _make_session()
    store.query.return_value = [_chunk()]
    session.history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]
    att = _attachment("CONTEXT_1")

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[att.path]):
            with patch("pmca.chat.resolve_attachments", return_value=[att]):
                with patch("pmca.chat.substitute_identifiers", return_value="new q"):
                    session.process("new q")

    messages = mock_cc.call_args[0][0]
    roles = [m["role"] for m in messages]
    # system (base), system (rag), system (attachment), user (history),
    # assistant (history), user (current)
    assert roles == ["system", "system", "system", "user", "assistant", "user"]


def test_process_current_user_message_is_last():
    session, store, _ = _make_session()

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("my question")

    messages = mock_cc.call_args[0][0]
    assert messages[-1] == {"role": "user", "content": "my question"}


# ---------------------------------------------------------------------------
# process() — history update
# ---------------------------------------------------------------------------

def test_process_appends_user_and_assistant_to_history():
    session, store, _ = _make_session()

    with patch("pmca.chat.chat_completion", return_value="great answer"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("my question")

    assert len(session.history) == 2
    assert session.history[0] == {"role": "user", "content": "my question"}
    assert session.history[1] == {"role": "assistant", "content": "great answer"}


def test_process_does_not_append_history_on_abort():
    session, store, _ = _make_session()
    from pmca.attachments import AttachmentAborted

    with patch("pmca.chat.parse_attachment_paths", return_value=[Path("/f.py")]):
        with patch("pmca.chat.resolve_attachments", side_effect=AttachmentAborted()):
            session.process("hi")

    assert session.history == []


# ---------------------------------------------------------------------------
# process() — attachment counter
# ---------------------------------------------------------------------------

def test_next_attachment_n_advances_after_successful_turn(tmp_path):
    session, store, _ = _make_session()
    att1 = _attachment("CONTEXT_1")
    att2 = _attachment("CONTEXT_2")

    with patch("pmca.chat.chat_completion", return_value="r"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[att1.path, att2.path]):
            with patch("pmca.chat.resolve_attachments", return_value=[att1, att2]):
                with patch("pmca.chat.substitute_identifiers", return_value="hi"):
                    session.process("hi")

    assert session._next_attachment_n == 3


def test_next_attachment_n_passes_to_resolve_attachments(tmp_path):
    session, store, _ = _make_session()
    session._next_attachment_n = 5
    att = _attachment("CONTEXT_5")

    with patch("pmca.chat.chat_completion", return_value="r"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[att.path]):
            with patch("pmca.chat.resolve_attachments", return_value=[att]) as mock_resolve:
                with patch("pmca.chat.substitute_identifiers", return_value="hi"):
                    session.process("hi")

    mock_resolve.assert_called_once_with([att.path], session.config.max_attachment_kb, session.unsafe, start_n=5)


def test_next_attachment_n_does_not_advance_on_abort(tmp_path):
    session, store, _ = _make_session()
    from pmca.attachments import AttachmentAborted

    with patch("pmca.chat.parse_attachment_paths", return_value=[Path("/f.py")]):
        with patch("pmca.chat.resolve_attachments", side_effect=AttachmentAborted()):
            session.process("hi")

    assert session._next_attachment_n == 1


# ---------------------------------------------------------------------------
# process() — return value
# ---------------------------------------------------------------------------

def test_process_returns_response_and_turns_dropped():
    session, store, _ = _make_session()

    with patch("pmca.chat.chat_completion", return_value="answer"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            response, turns_dropped = session.process("hi")

    assert response == "answer"
    assert turns_dropped == 0


def test_process_returns_none_response_on_abort():
    session, store, _ = _make_session()
    from pmca.attachments import AttachmentAborted

    with patch("pmca.chat.parse_attachment_paths", return_value=[Path("/f.py")]):
        with patch("pmca.chat.resolve_attachments", side_effect=AttachmentAborted()):
            result = session.process("hi")

    assert result[0] is None


# ---------------------------------------------------------------------------
# process() — logging
# ---------------------------------------------------------------------------

def test_process_calls_log_exchange():
    session, store, logger = _make_session()
    store.query.return_value = [_chunk()]

    with patch("pmca.chat.chat_completion", return_value="reply"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    logger.log_exchange.assert_called_once()


def test_process_does_not_log_on_abort():
    session, store, logger = _make_session()
    from pmca.attachments import AttachmentAborted

    with patch("pmca.chat.parse_attachment_paths", return_value=[Path("/f.py")]):
        with patch("pmca.chat.resolve_attachments", side_effect=AttachmentAborted()):
            session.process("hi")

    logger.log_exchange.assert_not_called()


# ---------------------------------------------------------------------------
# resumed_context injection
# ---------------------------------------------------------------------------

def test_resumed_context_injected_after_system_prompt(tmp_path):
    session, store, _ = _make_session()
    session.resumed_context = "prior context block"

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    assert messages[0]["content"] == "You are helpful."
    assert messages[1] == {"role": "system", "content": "prior context block"}


def test_resumed_context_not_injected_when_none(tmp_path):
    session, store, _ = _make_session()
    assert session.resumed_context is None

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    assert not any(m.get("content") == "prior context block" for m in messages)


# ---------------------------------------------------------------------------
# _trim_history
# ---------------------------------------------------------------------------

def test_trim_history_no_op_when_within_budget():
    session, _, _ = _make_session(_config(history_token_budget=4000))
    session.history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    dropped = session._trim_history()
    assert dropped == 0
    assert len(session.history) == 2


def test_trim_history_drops_oldest_pair():
    # each message ≈ 4000 chars / 4 = 1000 tokens; budget = 500 → must drop
    session, _, _ = _make_session(_config(history_token_budget=500))
    session.history = [
        {"role": "user", "content": "a" * 1000},
        {"role": "assistant", "content": "b" * 1000},
        {"role": "user", "content": "c" * 100},
        {"role": "assistant", "content": "d" * 100},
    ]
    dropped = session._trim_history()
    assert dropped == 1
    assert session.history[0]["content"] == "c" * 100


def test_trim_history_returns_count_of_dropped_pairs():
    session, _, _ = _make_session(_config(history_token_budget=1))
    session.history = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
        {"role": "user", "content": "c" * 100},
        {"role": "assistant", "content": "d" * 100},
    ]
    dropped = session._trim_history()
    assert dropped == 2


# ---------------------------------------------------------------------------
# rotate_logger
# ---------------------------------------------------------------------------

def test_rotate_logger_closes_old_logger():
    session, _, old_logger = _make_session()
    with patch("pmca.chat.SessionLogger"):
        with patch("pmca.chat.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-05-24_12-00-00"
            session.rotate_logger()
    old_logger.close.assert_called_once()


def test_rotate_logger_assigns_new_logger():
    session, _, old_logger = _make_session()
    with patch("pmca.chat.SessionLogger") as MockLogger:
        with patch("pmca.chat.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-05-24_12-00-01"
            session.rotate_logger()
    assert session.logger is MockLogger.return_value


def test_rotate_logger_resets_next_attachment_n():
    session, _, _ = _make_session()
    session._next_attachment_n = 7
    with patch("pmca.chat.SessionLogger"):
        with patch("pmca.chat.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-05-24_12-00-00"
            session.rotate_logger()
    assert session._next_attachment_n == 1


def test_rotate_logger_returns_new_jsonl_path():
    session, _, _ = _make_session(_config(log_folder=Path("/tmp/logs")))
    with patch("pmca.chat.SessionLogger"):
        with patch("pmca.chat.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-05-24_12-00-02"
            result = session.rotate_logger()
    assert result == Path("/tmp/logs/chat_2026-05-24_12-00-02.jsonl")


def test_trim_history_returns_turned_dropped_in_process():
    session, store, _ = _make_session(_config(history_token_budget=1))
    session.history = [
        {"role": "user", "content": "x" * 100},
        {"role": "assistant", "content": "y" * 100},
    ]

    with patch("pmca.chat.chat_completion", return_value="ok"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            _, turns_dropped = session.process("new q")

    assert turns_dropped == 1
