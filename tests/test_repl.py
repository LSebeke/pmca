from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pmca.repl import handle_command, run_repl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session(**attrs):
    session = MagicMock()
    session.history_token_budget = 4000
    for k, v in attrs.items():
        setattr(session, k, v)
    return session


def _run(inputs: list, session=None):
    """Run the REPL with a fixed sequence of inputs, exiting on EOFError."""
    s = session or _session()
    if session is None:
        s.process.return_value = ("response", 0)
    mock_prompt = MagicMock()
    mock_prompt.prompt.side_effect = inputs + [EOFError()]
    with patch("pmca.repl.PromptSession", return_value=mock_prompt):
        run_repl(s)
    return s


# ---------------------------------------------------------------------------
# handle_command — /set
# ---------------------------------------------------------------------------

def test_set_history_token_budget():
    session = _session()
    handle_command("/set history_token_budget=2000", session)
    assert session.history_token_budget == 2000


def test_set_unknown_param_prints_error(capsys):
    session = _session()
    handle_command("/set foobar=1", session)
    out = capsys.readouterr().out
    assert out.strip()


def test_set_chunksize_prints_unknown_param_error(capsys):
    session = _session()
    handle_command("/set chunksize=5", session)
    out = capsys.readouterr().out
    assert out.strip()  # must print an error — chunksize is no longer a valid param


# ---------------------------------------------------------------------------
# handle_command — /help
# ---------------------------------------------------------------------------

def test_help_mentions_clear(capsys):
    handle_command("/help", _session())
    assert "/clear" in capsys.readouterr().out


def test_help_mentions_set(capsys):
    handle_command("/help", _session())
    assert "/set" in capsys.readouterr().out


def test_help_does_not_mention_rag(capsys):
    handle_command("/help", _session())
    assert "/rag" not in capsys.readouterr().out


def test_help_does_not_mention_chunksize(capsys):
    handle_command("/help", _session())
    assert "chunksize" not in capsys.readouterr().out


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
    session = _run(["what is python?"])
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
    session = _run(["", "  ", "hello"])
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
        run_repl(session)  # should not raise


# ---------------------------------------------------------------------------
# handle_command — /clear
# ---------------------------------------------------------------------------

def test_clear_prints_confirmation(capsys):
    session = _session(history=[{"role": "user", "content": "hi"}])
    handle_command("/clear", session)
    assert "Conversation history cleared." in capsys.readouterr().out


def test_clear_resets_history():
    session = _session(history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    handle_command("/clear", session)
    assert session.history == []


def test_clear_resets_session_attachments():
    from pmca.types import Attachment
    att = Attachment(path=Path("/f.py"), content="x", identifier="CONTEXT_1", size_warning=False)
    session = _session(session_attachments=[att])
    handle_command("/clear", session)
    # rotate_logger() (mocked) handles the actual reset; verify it was called
    session.rotate_logger.assert_called_once()


def test_clear_calls_rotate_logger():
    session = _session()
    handle_command("/clear", session)
    session.rotate_logger.assert_called_once()


def test_clear_prints_new_session_path(tmp_path, capsys):
    session = _session()
    log_path = tmp_path / "chat_2026-05-24_14-00-00.jsonl"
    session.rotate_logger.return_value = log_path
    handle_command("/clear", session)
    out = capsys.readouterr().out
    assert "New session:" in out
    assert str(log_path) in out


# ---------------------------------------------------------------------------
# handle_command — /extract
# ---------------------------------------------------------------------------

def test_extract_missing_path_prints_error(capsys):
    session = _session(history=[])
    handle_command("/extract", session)
    assert capsys.readouterr().out.strip()


def test_extract_no_history_prints_error(capsys):
    session = _session(history=[])
    handle_command("/extract /tmp/out.py", session)
    assert capsys.readouterr().out.strip()


def test_extract_creates_parent_directories(tmp_path):
    code = "x = 1"
    msg = f"```python\n{code}\n```"
    session = _session(history=[{"role": "assistant", "content": msg}])
    out = tmp_path / "a" / "b" / "out.py"
    handle_command(f"/extract {out}", session)
    assert out.read_text() == code


def test_extract_no_blocks_prints_error_and_no_file(tmp_path, capsys):
    session = _session(history=[{"role": "assistant", "content": "no code here"}])
    out = tmp_path / "out.py"
    handle_command(f"/extract {out}", session)
    assert capsys.readouterr().out.strip()
    assert not out.exists()


def test_extract_multiple_blocks_concatenated(tmp_path):
    msg = "```python\na = 1\n```\nsome prose\n```python\nb = 2\n```"
    session = _session(history=[{"role": "assistant", "content": msg}])
    out = tmp_path / "out.py"
    handle_command(f"/extract {out}", session)
    assert out.read_text() == "a = 1\n\nb = 2"


def test_extract_writes_python_block_to_file(tmp_path):
    code = "x = 1\nprint(x)"
    msg = f"Here you go:\n```python\n{code}\n```"
    session = _session(history=[{"role": "assistant", "content": msg}])
    out = tmp_path / "out.py"
    handle_command(f"/extract {out}", session)
    assert out.read_text() == code


def test_extract_yaml_block(tmp_path):
    content = "key: value\nother: 123"
    msg = f"```yaml\n{content}\n```"
    session = _session(history=[{"role": "assistant", "content": msg}])
    out = tmp_path / "config.yaml"
    handle_command(f"/extract {out}", session)
    assert out.read_text() == content


def test_extract_yml_extension_matches_yaml_fence(tmp_path):
    content = "key: value"
    msg = f"```yaml\n{content}\n```"
    session = _session(history=[{"role": "assistant", "content": msg}])
    out = tmp_path / "config.yml"
    handle_command(f"/extract {out}", session)
    assert out.read_text() == content


def test_extract_json_block(tmp_path):
    content = '{"key": "value"}'
    msg = f"```json\n{content}\n```"
    session = _session(history=[{"role": "assistant", "content": msg}])
    out = tmp_path / "data.json"
    handle_command(f"/extract {out}", session)
    assert out.read_text() == content


def test_extract_sh_block(tmp_path):
    content = "#!/bin/bash\necho hello"
    msg = f"```bash\n{content}\n```"
    session = _session(history=[{"role": "assistant", "content": msg}])
    out = tmp_path / "run.sh"
    handle_command(f"/extract {out}", session)
    assert out.read_text() == content


def test_extract_toml_block(tmp_path):
    content = "[tool]\nname = \"pmca\""
    msg = f"```toml\n{content}\n```"
    session = _session(history=[{"role": "assistant", "content": msg}])
    out = tmp_path / "pyproject.toml"
    handle_command(f"/extract {out}", session)
    assert out.read_text() == content


def test_extract_md_block(tmp_path):
    content = "# Title\n\nSome content."
    msg = f"```markdown\n{content}\n```"
    session = _session(history=[{"role": "assistant", "content": msg}])
    out = tmp_path / "notes.md"
    handle_command(f"/extract {out}", session)
    assert out.read_text() == content


# ---------------------------------------------------------------------------
# handle_command — /read
# ---------------------------------------------------------------------------

def test_read_add_appends_path_on_approval(tmp_path):
    session = _session()
    session.config = MagicMock()
    session.config.read_allowed_dirs = []
    with patch("builtins.input", return_value="y"):
        handle_command(f"/read add {tmp_path}", session)
    assert tmp_path in session.config.read_allowed_dirs


def test_read_add_does_not_append_on_denial(tmp_path):
    session = _session()
    session.config = MagicMock()
    session.config.read_allowed_dirs = []
    with patch("builtins.input", return_value="n"):
        handle_command(f"/read add {tmp_path}", session)
    assert session.config.read_allowed_dirs == []


def test_read_remove_removes_path_on_approval(tmp_path):
    session = _session()
    session.config = MagicMock()
    session.config.read_allowed_dirs = [tmp_path]
    with patch("builtins.input", return_value="y"):
        handle_command(f"/read remove {tmp_path}", session)
    assert tmp_path not in session.config.read_allowed_dirs


def test_read_remove_unknown_path_prints_message(tmp_path, capsys):
    session = _session()
    session.config = MagicMock()
    session.config.read_allowed_dirs = []
    handle_command(f"/read remove {tmp_path}", session)
    assert capsys.readouterr().out.strip()


def test_help_mentions_read(capsys):
    handle_command("/help", _session())
    assert "/read" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# handle_command — /set test_timeout
# ---------------------------------------------------------------------------

def test_set_test_timeout_updates_config(capsys):
    session = _session()
    session.config = MagicMock()
    session.config.test_timeout = 60
    handle_command("/set test_timeout=120", session)
    assert session.config.test_timeout == 120


def test_set_test_timeout_zero_prints_error_and_leaves_unchanged(capsys):
    session = _session()
    session.config = MagicMock()
    session.config.test_timeout = 60
    handle_command("/set test_timeout=0", session)
    assert session.config.test_timeout == 60
    assert capsys.readouterr().out.strip()


def test_set_test_timeout_negative_prints_error_and_leaves_unchanged(capsys):
    session = _session()
    session.config = MagicMock()
    session.config.test_timeout = 60
    handle_command("/set test_timeout=-5", session)
    assert session.config.test_timeout == 60
    assert capsys.readouterr().out.strip()


def test_help_mentions_test_timeout(capsys):
    handle_command("/help", _session())
    assert "test_timeout" in capsys.readouterr().out


def test_extract_unknown_extension_prints_error(tmp_path, capsys):
    session = _session(history=[{"role": "assistant", "content": "```text\nhello\n```"}])
    out = tmp_path / "out.txt"
    handle_command(f"/extract {out}", session)
    err = capsys.readouterr().out
    assert ".txt" in err or "unsupported" in err.lower() or "supported" in err.lower()
    assert not out.exists()


# ---------------------------------------------------------------------------
# handle_command — /scratchpad
# ---------------------------------------------------------------------------

def test_scratchpad_empty_prints_message(capsys):
    session = _session()
    session._scratchpad = []
    handle_command("/scratchpad", session)
    assert "empty" in capsys.readouterr().out.lower()


def test_scratchpad_shows_entry_title_and_content(capsys):
    from pmca.types import ScratchpadEntry
    session = _session()
    session._scratchpad = [
        ScratchpadEntry(title="my-note", content="important stuff"),
    ]
    handle_command("/scratchpad", session)
    out = capsys.readouterr().out
    assert "my-note" in out
    assert "important stuff" in out


def test_help_mentions_scratchpad(capsys):
    handle_command("/help", _session())
    assert "/scratchpad" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# handle_command — /skill
# ---------------------------------------------------------------------------

def _skill_session(skills_dir=None, active_skills=None):
    session = _session()
    session.config = MagicMock()
    session.config.skills_dir = skills_dir
    session._active_skills = active_skills if active_skills is not None else []
    return session


def test_skill_no_skills_dir_prints_error(capsys):
    session = _skill_session(skills_dir=None)
    handle_command("/skill", session)
    assert capsys.readouterr().out.strip()


def _make_skill_dir(parent, name):
    d = parent / name
    d.mkdir()
    (d / "SKILL.md").write_text(f"# {name} skill")
    return d


def test_skill_empty_dir_prints_no_skills_message(tmp_path, capsys):
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill", session)
    assert "No skills available" in capsys.readouterr().out


def test_skill_flat_md_file_not_listed(tmp_path, capsys):
    (tmp_path / "tdd.md").write_text("TDD content")
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill", session)
    assert "No skills available" in capsys.readouterr().out


def test_skill_lists_available_skills(tmp_path, capsys):
    _make_skill_dir(tmp_path, "tdd")
    _make_skill_dir(tmp_path, "security")
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill", session)
    out = capsys.readouterr().out
    assert "tdd" in out
    assert "security" in out


def test_skill_dir_without_skill_md_not_listed(tmp_path, capsys):
    (tmp_path / "empty-skill").mkdir()
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill", session)
    assert "No skills available" in capsys.readouterr().out


def test_skill_marks_active_skill_with_star(tmp_path, capsys):
    from pmca.types import ActiveSkill
    skill_dir = _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(
        skills_dir=tmp_path,
        active_skills=[ActiveSkill(name="tdd", content="TDD content", directory=skill_dir)],
    )
    handle_command("/skill", session)
    out = capsys.readouterr().out
    assert "*" in out
    assert "tdd" in out


def test_skill_list_includes_legend(tmp_path, capsys):
    _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill", session)
    out = capsys.readouterr().out
    assert "*" in out  # legend explains the symbol


def test_skill_activate_prints_confirmation(tmp_path, capsys):
    _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill tdd", session)
    out = capsys.readouterr().out
    assert "tdd" in out.lower()
    assert "activat" in out.lower()


def test_skill_activate_adds_to_active_skills(tmp_path):
    from pmca.types import ActiveSkill
    _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill tdd", session)
    assert any(isinstance(s, ActiveSkill) and s.name == "tdd" for s in session._active_skills)


def test_skill_activate_loads_skill_md_content(tmp_path):
    skill_dir = tmp_path / "tdd"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("## TDD\nWrite tests first.")
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill tdd", session)
    assert session._active_skills[0].content == "## TDD\nWrite tests first."


def test_skill_activate_sets_directory(tmp_path):
    from pmca.types import ActiveSkill
    skill_dir = _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill tdd", session)
    assert session._active_skills[0].directory == skill_dir


def test_skill_activate_adds_dir_to_read_allowed_dirs(tmp_path):
    skill_dir = _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(skills_dir=tmp_path)
    session.config.read_allowed_dirs = []
    handle_command("/skill tdd", session)
    assert skill_dir in session.config.read_allowed_dirs


def test_skill_activate_already_active_prints_message(tmp_path, capsys):
    from pmca.types import ActiveSkill
    skill_dir = _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(
        skills_dir=tmp_path,
        active_skills=[ActiveSkill(name="tdd", content="TDD content", directory=skill_dir)],
    )
    handle_command("/skill tdd", session)
    out = capsys.readouterr().out
    assert "already" in out.lower()


def test_skill_activate_already_active_does_not_duplicate(tmp_path):
    from pmca.types import ActiveSkill
    skill_dir = _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(
        skills_dir=tmp_path,
        active_skills=[ActiveSkill(name="tdd", content="TDD content", directory=skill_dir)],
    )
    handle_command("/skill tdd", session)
    assert len(session._active_skills) == 1


def test_skill_activate_not_found_prints_error(tmp_path, capsys):
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill nonexistent", session)
    assert capsys.readouterr().out.strip()


def test_skill_remove_deactivates_skill(tmp_path):
    from pmca.types import ActiveSkill
    skill_dir = _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(
        skills_dir=tmp_path,
        active_skills=[ActiveSkill(name="tdd", content="TDD content", directory=skill_dir)],
    )
    session.config.read_allowed_dirs = [skill_dir]
    handle_command("/skill remove tdd", session)
    assert session._active_skills == []


def test_skill_remove_prints_confirmation(tmp_path, capsys):
    from pmca.types import ActiveSkill
    skill_dir = _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(
        skills_dir=tmp_path,
        active_skills=[ActiveSkill(name="tdd", content="TDD content", directory=skill_dir)],
    )
    session.config.read_allowed_dirs = [skill_dir]
    handle_command("/skill remove tdd", session)
    out = capsys.readouterr().out
    assert "tdd" in out.lower()


def test_skill_remove_removes_dir_from_read_allowed_dirs(tmp_path):
    from pmca.types import ActiveSkill
    skill_dir = _make_skill_dir(tmp_path, "tdd")
    session = _skill_session(
        skills_dir=tmp_path,
        active_skills=[ActiveSkill(name="tdd", content="TDD content", directory=skill_dir)],
    )
    session.config.read_allowed_dirs = [skill_dir]
    handle_command("/skill remove tdd", session)
    assert skill_dir not in session.config.read_allowed_dirs


def test_skill_remove_not_active_prints_error(tmp_path, capsys):
    session = _skill_session(skills_dir=tmp_path)
    handle_command("/skill remove tdd", session)
    assert capsys.readouterr().out.strip()


def test_help_mentions_skill(capsys):
    handle_command("/help", _session())
    assert "/skill" in capsys.readouterr().out
