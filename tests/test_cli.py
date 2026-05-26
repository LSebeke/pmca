import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from pmca.cli import main
from pmca.resume import ResumedSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_file(tmp_path):
    rag = tmp_path / "code.py"
    rag.write_text("x = 1")
    log_folder = tmp_path / "logs"
    cfg = tmp_path / "Test.yaml"
    cfg.write_text(f"""\
name: Test
model: gpt-4o-mini
system_prompt: "You are helpful."
rag_files:
  - {rag}
log_folder: {log_folder}
""")
    return cfg


def _run(argv, env_extras=None):
    """Call main() with a controlled argv and environment."""
    env = {**os.environ, "OPENAI_API_KEY": "test-key", **(env_extras or {})}
    with patch.dict(os.environ, env, clear=True):
        with patch("sys.argv", argv):
            main()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_main_calls_run_repl(config_file):
    with patch("pmca.cli.VectorStore") as MockStore:
        MockStore.return_value.build.return_value = None
        with patch("pmca.cli.run_repl") as mock_repl:
            _run(["pmca", str(config_file)])

    mock_repl.assert_called_once()


def test_main_passes_session_to_run_repl(config_file):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl") as mock_repl:
            with patch("pmca.cli.ChatSession") as MockSession:
                _run(["pmca", str(config_file)])

    session_instance = MockSession.return_value
    mock_repl.assert_called_once_with(session_instance)


def test_main_closes_logger_on_exit(config_file):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            with patch("pmca.cli.ChatSession") as MockSession:
                _run(["pmca", str(config_file)])

    MockSession.return_value.logger.close.assert_called_once()


def test_main_closes_logger_even_when_repl_raises(config_file):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl", side_effect=RuntimeError("boom")):
            with patch("pmca.cli.ChatSession") as MockSession:
                with pytest.raises(RuntimeError):
                    _run(["pmca", str(config_file)])

    MockSession.return_value.logger.close.assert_called_once()


# ---------------------------------------------------------------------------
# --unsafe flag
# ---------------------------------------------------------------------------

def test_unsafe_flag_sets_session_unsafe(config_file):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            with patch("pmca.cli.ChatSession") as MockSession:
                _run(["pmca", str(config_file), "--unsafe"])

    _, kwargs = MockSession.call_args
    assert kwargs.get("unsafe") is True or MockSession.call_args[1].get("unsafe") is True


def test_no_unsafe_flag_defaults_to_false(config_file):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            with patch("pmca.cli.ChatSession") as MockSession:
                _run(["pmca", str(config_file)])

    kwargs = MockSession.call_args[1]
    assert kwargs.get("unsafe") is False or kwargs.get("unsafe") is None


# ---------------------------------------------------------------------------
# Startup errors
# ---------------------------------------------------------------------------

def test_nonexistent_config_exits_nonzero(tmp_path):
    nonexistent = str(tmp_path / "NoSuch.yaml")
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("sys.argv", ["pmca", nonexistent]):
            with pytest.raises(SystemExit) as exc_info:
                main()
    assert exc_info.value.code != 0


def test_nonexistent_config_prints_to_stderr(tmp_path, capsys):
    nonexistent = str(tmp_path / "NoSuch.yaml")
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("sys.argv", ["pmca", nonexistent]):
            with pytest.raises(SystemExit):
                main()
    assert capsys.readouterr().err.strip()


def test_missing_api_key_exits_nonzero(config_file):
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with patch("sys.argv", ["pmca", str(config_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
    assert exc_info.value.code != 0


def test_missing_api_key_prints_to_stderr(config_file, capsys):
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with patch("sys.argv", ["pmca", str(config_file)]):
            with pytest.raises(SystemExit):
                main()
    assert capsys.readouterr().err.strip()


# ---------------------------------------------------------------------------
# --resume flag
# ---------------------------------------------------------------------------

@pytest.fixture
def resume_log(tmp_path):
    p = tmp_path / "chat_2025-01-01_00-00-00.jsonl"
    p.write_text(
        json.dumps({"type": "system_prompt", "content": "You are helpful."}) + "\n"
        + json.dumps({"type": "exchange", "timestamp": "t", "role": "user", "content": "old q", "rag_chunks": [], "attachments": []}) + "\n"
        + json.dumps({"type": "exchange", "timestamp": "t", "role": "assistant", "content": "old answer"}) + "\n"
    )
    return p


def test_resume_flag_loads_history_into_session(config_file, resume_log):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            with patch("pmca.cli.ChatSession") as MockSession:
                _run(["pmca", str(config_file), "--resume", str(resume_log)])

    session = MockSession.return_value
    assert session.history == [
        {"role": "user", "content": "old q"},
        {"role": "assistant", "content": "old answer"},
    ]


def test_resume_flag_sets_session_attachments(config_file, resume_log):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            with patch("pmca.cli.ChatSession") as MockSession:
                _run(["pmca", str(config_file), "--resume", str(resume_log)])

    session = MockSession.return_value
    assert session.session_attachments == []


def test_resume_flag_uses_from_existing_logger(config_file, resume_log):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            with patch("pmca.cli.SessionLogger") as MockLogger:
                with patch("pmca.cli.load_resume") as mock_lr:
                    mock_lr.return_value = ResumedSession(
                        system_prompt="You are helpful.",
                        startup_docs=[],
                        history=[],
                        session_attachments=[],
                        last_assistant_message="hi",
                        jsonl_path=resume_log,
                        next_attachment_n=1,
                    )
                    _run(["pmca", str(config_file), "--resume", str(resume_log)])

    MockLogger.from_existing.assert_called_once_with(resume_log)


def test_resume_prints_startup_summary(config_file, resume_log, capsys):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            _run(["pmca", str(config_file), "--resume", str(resume_log)])

    out = capsys.readouterr().out
    assert "Resumed" in out
    assert "1" in out  # 1 turn


def test_resume_prints_last_assistant_message(config_file, resume_log, capsys):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            _run(["pmca", str(config_file), "--resume", str(resume_log)])

    out = capsys.readouterr().out
    assert "old answer" in out
    assert "[last response]" in out


def test_log_session_start_called_on_fresh_start(config_file):
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            with patch("pmca.cli.SessionLogger") as MockLogger:
                _run(["pmca", str(config_file)])

    MockLogger.return_value.log_session_start.assert_called_once()


def test_resume_warns_if_system_prompt_differs(config_file, tmp_path, capsys):
    log = tmp_path / "chat_diff.jsonl"
    log.write_text(
        json.dumps({"type": "system_prompt", "content": "DIFFERENT PROMPT"}) + "\n"
        + json.dumps({"type": "exchange", "timestamp": "t", "role": "user", "content": "q", "rag_chunks": [], "attachments": []}) + "\n"
        + json.dumps({"type": "exchange", "timestamp": "t", "role": "assistant", "content": "a"}) + "\n"
    )
    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            _run(["pmca", str(config_file), "--resume", str(log)])

    out = capsys.readouterr().out
    assert "Warning" in out
    assert "system_prompt" in out


def test_resume_exits_on_missing_file(config_file, tmp_path, capsys):
    missing = tmp_path / "ghost.jsonl"
    with patch("pmca.cli.VectorStore"):
        with pytest.raises(SystemExit) as exc_info:
            _run(["pmca", str(config_file), "--resume", str(missing)])

    assert exc_info.value.code != 0
    assert capsys.readouterr().err.strip()


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def test_log_folder_created_if_absent(config_file, tmp_path):
    log_folder = tmp_path / "logs"
    assert not log_folder.exists()

    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            _run(["pmca", str(config_file)])

    assert log_folder.is_dir()


def test_cache_dir_created_if_absent(config_file, tmp_path):
    cache_dir = tmp_path / "logs" / "cache"
    assert not cache_dir.exists()

    with patch("pmca.cli.VectorStore"):
        with patch("pmca.cli.run_repl"):
            _run(["pmca", str(config_file)])

    assert cache_dir.is_dir()
