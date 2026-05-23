from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pmca.repl import handle_command, run_repl
from pmca.types import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(**attrs):
    session = MagicMock()
    session.top_k = 3
    session.history_token_budget = 4000
    session._last_rag_chunks = []
    for k, v in attrs.items():
        setattr(session, k, v)
    return session


def _chunk(label: str = "fn `foo`") -> Chunk:
    return Chunk(content="def foo(): pass", source_file=Path("/a.py"), label=label)


def _run(inputs: list, session=None, logger=None):
    """Run the REPL with a fixed sequence of inputs, exiting on EOFError."""
    s = session or _session()
    if session is None:
        s.process.return_value = ("response", 0)
    l = logger or MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.prompt.side_effect = inputs + [EOFError()]
    with patch("pmca.repl.PromptSession", return_value=mock_prompt):
        run_repl(s, l)
    return s, l


# ---------------------------------------------------------------------------
# handle_command — /set
# ---------------------------------------------------------------------------

def test_set_chunksize_updates_top_k():
    session = _session()
    handle_command("/set chunksize=5", session)
    assert session.top_k == 5


def test_set_history_token_budget():
    session = _session()
    handle_command("/set history_token_budget=2000", session)
    assert session.history_token_budget == 2000


def test_set_negative_chunksize_prints_error_and_leaves_unchanged(capsys):
    session = _session(top_k=3)
    handle_command("/set chunksize=-1", session)
    assert session.top_k == 3
    out = capsys.readouterr().out
    assert out.strip()  # something was printed


def test_set_zero_chunksize_prints_error(capsys):
    session = _session(top_k=3)
    handle_command("/set chunksize=0", session)
    assert session.top_k == 3
    assert capsys.readouterr().out.strip()


def test_set_unknown_param_prints_error(capsys):
    session = _session()
    handle_command("/set foobar=1", session)
    out = capsys.readouterr().out
    assert out.strip()


def test_set_non_integer_value_prints_error(capsys):
    session = _session(top_k=3)
    handle_command("/set chunksize=abc", session)
    assert session.top_k == 3
    assert capsys.readouterr().out.strip()


# ---------------------------------------------------------------------------
# handle_command — /rag
# ---------------------------------------------------------------------------

def test_rag_prints_chunk_labels(capsys):
    chunk = _chunk("function `parse` (lines 1–5)")
    session = _session(_last_rag_chunks=[chunk])
    handle_command("/rag", session)
    out = capsys.readouterr().out
    assert "function `parse` (lines 1–5)" in out


def test_rag_prints_chunk_source(capsys):
    chunk = _chunk()
    session = _session(_last_rag_chunks=[chunk])
    handle_command("/rag", session)
    out = capsys.readouterr().out
    assert "/a.py" in out


def test_rag_no_data_prints_notice(capsys):
    session = _session(_last_rag_chunks=[])
    handle_command("/rag", session)
    out = capsys.readouterr().out
    assert out.strip()


# ---------------------------------------------------------------------------
# handle_command — /help
# ---------------------------------------------------------------------------

def test_help_mentions_set(capsys):
    handle_command("/help", _session())
    assert "/set" in capsys.readouterr().out


def test_help_mentions_rag(capsys):
    handle_command("/help", _session())
    assert "/rag" in capsys.readouterr().out


def test_help_mentions_exit(capsys):
    handle_command("/help", _session())
    assert "/exit" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# handle_command — /exit
# ---------------------------------------------------------------------------

def test_exit_raises_system_exit():
    with pytest.raises(SystemExit):
        handle_command("/exit", _session())


# ---------------------------------------------------------------------------
# run_repl — normal input
# ---------------------------------------------------------------------------

def test_non_command_calls_process():
    session, _ = _run(["what is python?"])
    session.process.assert_called_once_with("what is python?")


def test_non_command_prints_response(capsys):
    session = _session()
    session.process.return_value = ("great answer", 0)
    _run(["hi"], session=session)
    assert "great answer" in capsys.readouterr().out


def test_trim_notice_printed_when_turns_dropped(capsys):
    session = _session()
    session.process.return_value = ("response", 2)
    _run(["q"], session=session)
    out = capsys.readouterr().out
    assert "2 earlier turn(s) omitted from context" in out


def test_no_trim_notice_when_zero_dropped(capsys):
    session = _session()
    session.process.return_value = ("response", 0)
    _run(["q"], session=session)
    assert "omitted from context" not in capsys.readouterr().out


def test_aborted_message_no_response_printed(capsys):
    session = _session()
    session.process.return_value = (None, 0)
    _run(["hi"], session=session)
    out = capsys.readouterr().out
    assert "None" not in out
    assert "omitted from context" not in out


def test_empty_input_skipped():
    session, _ = _run(["", "  ", "hello"])
    session.process.assert_called_once_with("hello")


def test_command_dispatched_not_processed(capsys):
    session = _session()
    _run(["/help"], session=session)
    session.process.assert_not_called()


def test_keyboard_interrupt_exits_loop():
    session = _session()
    mock_prompt = MagicMock()
    mock_prompt.prompt.side_effect = [KeyboardInterrupt()]
    with patch("pmca.repl.PromptSession", return_value=mock_prompt):
        run_repl(session, MagicMock())  # should not raise
