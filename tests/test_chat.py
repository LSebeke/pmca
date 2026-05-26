from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from pmca.chat import ChatSession, _build_system_context
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
# _build_system_context
# ---------------------------------------------------------------------------

def test_build_system_context_empty_fields_returns_none():
    assert _build_system_context([]) is None


def test_build_system_context_unknown_only_returns_none():
    assert _build_system_context(["bogus", "nope"]) is None


def test_build_system_context_datetime_field():
    result = _build_system_context(["datetime"])
    assert result is not None
    assert "Session started:" in result
    assert "OS:" not in result
    assert "Shell:" not in result


def test_build_system_context_os_field():
    import platform
    result = _build_system_context(["os"])
    assert result is not None
    assert f"OS: {platform.system()}" in result
    assert "Session started:" not in result
    assert "Shell:" not in result


def test_build_system_context_shell_field():
    result = _build_system_context(["shell"])
    assert result is not None
    assert "Shell:" in result
    assert "Session started:" not in result
    assert "OS:" not in result


def test_build_system_context_all_fields_in_order():
    result = _build_system_context(["datetime", "os", "shell"])
    assert result is not None
    lines = result.splitlines()
    assert lines[0].startswith("Session started:")
    assert lines[1].startswith("OS:")
    assert lines[2].startswith("Shell:")


def test_build_system_context_order_fixed_regardless_of_input_order():
    result = _build_system_context(["shell", "datetime", "os"])
    assert result is not None
    lines = result.splitlines()
    assert lines[0].startswith("Session started:")
    assert lines[1].startswith("OS:")
    assert lines[2].startswith("Shell:")


def test_build_system_context_mixed_known_and_unknown():
    result = _build_system_context(["datetime", "unknown_field"])
    assert result is not None
    assert "Session started:" in result
    assert "unknown_field" not in result


def test_build_system_context_shell_uses_comspec_when_shell_absent(monkeypatch):
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.setenv("COMSPEC", "C:\\Windows\\System32\\cmd.exe")
    result = _build_system_context(["shell"])
    assert "C:\\Windows\\System32\\cmd.exe" in result


def test_build_system_context_shell_prefers_shell_over_comspec(monkeypatch):
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    monkeypatch.setenv("COMSPEC", "cmd.exe")
    result = _build_system_context(["shell"])
    assert "/usr/bin/zsh" in result
    assert "cmd.exe" not in result


# ---------------------------------------------------------------------------
# system context injection into messages
# ---------------------------------------------------------------------------

def test_system_context_computed_once_at_init_not_per_call():
    with patch("pmca.chat._build_system_context", return_value="[CTX]") as mock_build:
        session, store, _ = _make_session()

    assert mock_build.call_count == 1
    assert session._system_context == "[CTX]"


def test_system_context_omitted_from_messages_when_fields_empty():
    session, store, _ = _make_session()  # default: system_context_fields=[]
    assert session._system_context is None

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    assert all("Session started:" not in m.get("content", "") for m in messages)
    assert all("OS:" not in m.get("content", "") for m in messages)


def test_system_context_is_second_system_message_when_fields_set():
    cfg = _config(system_context_fields=["datetime", "os", "shell"])
    session, store, _ = _make_session(cfg)

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    messages = mock_cc.call_args[0][0]
    assert messages[1]["role"] == "system"
    assert "Session started:" in messages[1]["content"]
    assert "OS:" in messages[1]["content"]
    assert "Shell:" in messages[1]["content"]


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
    rag_msg = next(m for m in messages if "[RAG_1]" in m.get("content", ""))
    assert rag_msg["role"] == "system"
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
    # No system context injected (default system_context_fields=[]), so startup doc is messages[1]
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
    # system (base), system (session_attachments), system (session_rag_chunks),
    # user (history), assistant (history), user (current)
    # No system context message — default system_context_fields=[]
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
# process() — session_attachments persistence
# ---------------------------------------------------------------------------

def test_session_attachments_empty_at_start():
    session, _, _ = _make_session()
    assert session.session_attachments == []


def test_session_attachments_accumulates_after_turn():
    session, store, _ = _make_session()
    att = _attachment("CONTEXT_1")

    with patch("pmca.chat.chat_completion", return_value="r"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[att.path]):
            with patch("pmca.chat.resolve_attachments", return_value=[att]):
                with patch("pmca.chat.substitute_identifiers", return_value="hi"):
                    session.process("hi")

    assert session.session_attachments == [att]


def test_session_attachments_appear_in_subsequent_turn_with_no_new_attachments():
    session, store, _ = _make_session()
    att = _attachment("CONTEXT_1")

    with patch("pmca.chat.chat_completion", return_value="r"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[att.path]):
            with patch("pmca.chat.resolve_attachments", return_value=[att]):
                with patch("pmca.chat.substitute_identifiers", return_value="t1"):
                    session.process("t1")

    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("t2")

    messages = mock_cc.call_args[0][0]
    assert any("[CONTEXT_1]" in m.get("content", "") for m in messages)


def test_session_attachments_not_accumulated_on_abort():
    session, store, _ = _make_session()
    from pmca.attachments import AttachmentAborted

    with patch("pmca.chat.parse_attachment_paths", return_value=[Path("/f.py")]):
        with patch("pmca.chat.resolve_attachments", side_effect=AttachmentAborted()):
            session.process("hi")

    assert session.session_attachments == []


# ---------------------------------------------------------------------------
# process() — session_rag_chunks persistence
# ---------------------------------------------------------------------------

def test_session_rag_chunks_empty_at_start():
    session, _, _ = _make_session()
    assert session.session_rag_chunks == []


def test_session_rag_chunks_accumulates_after_turn():
    session, store, _ = _make_session()
    chunk = _chunk("fn `foo`")
    store.query.return_value = [chunk]

    with patch("pmca.chat.chat_completion", return_value="r"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hi")

    assert chunk in session.session_rag_chunks


def test_session_rag_chunks_appear_in_subsequent_turn():
    session, store, _ = _make_session()
    chunk = _chunk("fn `foo`")
    store.query.return_value = [chunk]

    with patch("pmca.chat.chat_completion", return_value="r"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("t1")

    store.query.return_value = []
    with patch("pmca.chat.chat_completion", return_value="r") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("t2")

    messages = mock_cc.call_args[0][0]
    assert any("fn `foo`" in m.get("content", "") for m in messages)


def test_session_rag_chunks_deduplicates_by_source_and_label():
    session, store, _ = _make_session()
    chunk = _chunk("fn `foo`")
    store.query.return_value = [chunk]

    with patch("pmca.chat.chat_completion", return_value="r"):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("t1")
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("t2")

    assert session.session_rag_chunks.count(chunk) == 1


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


def test_rotate_logger_resets_session_attachments():
    session, _, _ = _make_session()
    session.session_attachments = [_attachment("CONTEXT_1")]
    with patch("pmca.chat.SessionLogger"):
        with patch("pmca.chat.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-01_00-00-00"
            session.rotate_logger()
    assert session.session_attachments == []


def test_rotate_logger_resets_session_rag_chunks():
    session, _, _ = _make_session()
    session.session_rag_chunks = [_chunk()]
    with patch("pmca.chat.SessionLogger"):
        with patch("pmca.chat.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-01_00-00-00"
            session.rotate_logger()
    assert session.session_rag_chunks == []


def test_rotate_logger_calls_log_session_start_on_new_logger():
    session, _, _ = _make_session()
    with patch("pmca.chat.SessionLogger") as MockLogger:
        with patch("pmca.chat.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-01-01_00-00-01"
            session.rotate_logger()
    MockLogger.return_value.log_session_start.assert_called_once_with(
        session.config.system_prompt, session.config.startup_docs
    )


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


# ---------------------------------------------------------------------------
# Tool loop
# ---------------------------------------------------------------------------

def test_process_passes_tools_to_chat_completion_when_write_allowed_dirs_set(tmp_path):
    cfg = _config(write_allowed_dirs=[tmp_path])
    session, store, _ = _make_session(cfg)

    with patch("pmca.chat.chat_completion", return_value="done") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("write something")

    _, kwargs = mock_cc.call_args
    assert "tools" in kwargs
    assert kwargs["tools"] is not None


def test_process_passes_no_tools_when_write_allowed_dirs_empty():
    session, store, _ = _make_session(_config(write_allowed_dirs=[]))

    with patch("pmca.chat.chat_completion", return_value="done") as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            session.process("hello")

    _, kwargs = mock_cc.call_args
    assert kwargs.get("tools") is None


def test_process_tool_loop_executes_tool_and_continues(tmp_path):
    from pmca.types import ToolCallRequest
    cfg = _config(write_allowed_dirs=[tmp_path])
    session, store, logger = _make_session(cfg)

    tool_req = ToolCallRequest(
        tool_call_id="call_1",
        name="write_file",
        arguments={"path": str(tmp_path / "out.py"), "content": "x=1\n", "description": "test"},
    )

    with patch("pmca.chat.chat_completion", side_effect=[tool_req, "All done!"]) as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            with patch("pmca.chat.execute_write_file", return_value=(True, "Written: /out.py (4 bytes)")) as mock_exec:
                response, _ = session.process("write a file")

    assert response == "All done!"
    mock_exec.assert_called_once()
    assert mock_cc.call_count == 2


def test_process_tool_loop_logs_tool_call(tmp_path):
    from pmca.types import ToolCallRequest
    cfg = _config(write_allowed_dirs=[tmp_path])
    session, store, logger = _make_session(cfg)

    tool_req = ToolCallRequest(
        tool_call_id="call_1",
        name="write_file",
        arguments={"path": str(tmp_path / "out.py"), "content": "x\n", "description": "test"},
    )

    with patch("pmca.chat.chat_completion", side_effect=[tool_req, "Done"]):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            with patch("pmca.chat.execute_write_file", return_value=(True, "Written: /out.py (2 bytes)")):
                session.process("write")

    logger.log_tool_call.assert_called_once_with(
        tool_call_id="call_1",
        name="write_file",
        arguments=tool_req.arguments,
        approved=True,
        result="Written: /out.py (2 bytes)",
    )


def test_process_second_api_call_includes_tool_result_messages(tmp_path):
    from pmca.types import ToolCallRequest
    cfg = _config(write_allowed_dirs=[tmp_path])
    session, store, _ = _make_session(cfg)

    tool_req = ToolCallRequest(
        tool_call_id="call_1",
        name="write_file",
        arguments={"path": str(tmp_path / "f.py"), "content": "x\n", "description": "d"},
    )

    with patch("pmca.chat.chat_completion", side_effect=[tool_req, "Done"]) as mock_cc:
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            with patch("pmca.chat.execute_write_file", return_value=(False, "Write denied by user. Path: /f.py")):
                session.process("write")

    second_call_messages = mock_cc.call_args_list[1][0][0]
    roles = [m["role"] for m in second_call_messages]
    assert "tool" in roles
    tool_msg = next(m for m in second_call_messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert "denied" in tool_msg["content"].lower()


# ---------------------------------------------------------------------------
# Tool dispatch — read tools
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name,executor_path,executor_result", [
    ("read_file",       "pmca.chat.execute_read_file",       "file content"),
    ("list_dir",        "pmca.chat.execute_list_dir",        "/src/a.py\n/src/b.py"),
    ("search",          "pmca.chat.execute_search",          "match at line 3"),
    ("get_definition",  "pmca.chat.execute_get_definition",  "def foo():\n    pass"),
    ("run_tests",       "pmca.chat.execute_run_tests",       (True, "3 passed")),
    ("edit_file",       "pmca.chat.execute_edit_file",       (True, "Edited: /f.py")),
])
def test_process_dispatches_read_tool(tmp_path, tool_name, executor_path, executor_result):
    from pmca.types import ToolCallRequest
    cfg = _config(read_allowed_dirs=[tmp_path], test_dir=tmp_path)
    session, _, _ = _make_session(cfg)

    tool_req = ToolCallRequest(
        tool_call_id="call_r1",
        name=tool_name,
        arguments={"path": str(tmp_path / "x"), "pattern": "foo", "symbol": "foo",
                   "recursive": False, "context_lines": 3},
    )

    with patch("pmca.chat.chat_completion", side_effect=[tool_req, "Done"]):
        with patch("pmca.chat.parse_attachment_paths", return_value=[]):
            with patch(executor_path, return_value=executor_result) as mock_exec:
                response, _ = session.process("explore")

    assert response == "Done"
    mock_exec.assert_called_once()
