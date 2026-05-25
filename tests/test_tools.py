from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pmca.config import Config
from pmca.tools import execute_write_file, get_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides) -> Config:
    defaults = dict(
        name="test", model="gpt-4o-mini", system_prompt="You are helpful.",
        rag_files=[], top_k_chunks=3, log_folder=Path("/tmp/logs"),
        write_allowed_dirs=[],
    )
    defaults.update(overrides)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# get_tools
# ---------------------------------------------------------------------------

def test_get_tools_returns_none_when_no_allowed_dirs():
    cfg = _config(write_allowed_dirs=[])
    assert get_tools(cfg) is None


def test_get_tools_returns_list_when_dirs_configured(tmp_path):
    cfg = _config(write_allowed_dirs=[tmp_path])
    tools = get_tools(cfg)
    assert tools is not None
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "write_file"


def test_get_tools_description_lists_allowed_dirs(tmp_path):
    allowed = tmp_path / "output"
    cfg = _config(write_allowed_dirs=[allowed])
    tools = get_tools(cfg)
    description = tools[0]["function"]["description"]
    assert str(allowed) in description


def test_get_tools_schema_has_required_parameters(tmp_path):
    cfg = _config(write_allowed_dirs=[tmp_path])
    tools = get_tools(cfg)
    params = tools[0]["function"]["parameters"]
    required = params["required"]
    assert "path" in required
    assert "content" in required
    assert "description" in required


# ---------------------------------------------------------------------------
# execute_write_file — path outside allowed dirs
# ---------------------------------------------------------------------------

def test_execute_rejects_path_outside_allowed_dirs(tmp_path):
    allowed = tmp_path / "allowed"
    cfg = _config(write_allowed_dirs=[allowed])
    args = {"path": str(tmp_path / "sneaky" / "file.py"), "content": "x", "description": "test"}

    approved, result = execute_write_file(args, cfg)

    assert approved is False
    assert "outside allowed" in result.lower()


# ---------------------------------------------------------------------------
# execute_write_file — user denies
# ---------------------------------------------------------------------------

def test_execute_denies_when_user_inputs_not_y(tmp_path):
    allowed = tmp_path / "output"
    allowed.mkdir()
    target = allowed / "file.py"
    cfg = _config(write_allowed_dirs=[allowed])
    args = {"path": str(target), "content": "x = 1", "description": "Initial module"}

    with patch("builtins.input", return_value="n"):
        approved, result = execute_write_file(args, cfg)

    assert approved is False
    assert str(target.resolve()) in result


def test_execute_denial_message_contains_path(tmp_path):
    allowed = tmp_path / "output"
    allowed.mkdir()
    target = allowed / "file.py"
    cfg = _config(write_allowed_dirs=[allowed])
    args = {"path": str(target), "content": "x", "description": "test"}

    with patch("builtins.input", return_value=""):
        approved, result = execute_write_file(args, cfg)

    assert "Write denied by user" in result
    assert str(target.resolve()) in result


# ---------------------------------------------------------------------------
# execute_write_file — approval prompt format
# ---------------------------------------------------------------------------

def test_execute_prints_approval_prompt(tmp_path, capsys):
    allowed = tmp_path / "output"
    allowed.mkdir()
    target = allowed / "module.py"
    cfg = _config(write_allowed_dirs=[allowed])
    args = {"path": str(target), "content": "x = 1\n", "description": "My module"}

    with patch("builtins.input", return_value="n"):
        execute_write_file(args, cfg)

    out = capsys.readouterr().out
    assert str(target.resolve()) in out
    assert "My module" in out
    assert "File does not exist" in out


def test_execute_prompt_warns_when_file_exists(tmp_path, capsys):
    allowed = tmp_path / "output"
    allowed.mkdir()
    target = allowed / "existing.py"
    target.write_text("old content")
    cfg = _config(write_allowed_dirs=[allowed])
    args = {"path": str(target), "content": "new content", "description": "replace"}

    with patch("builtins.input", return_value="n"):
        execute_write_file(args, cfg)

    out = capsys.readouterr().out
    assert "will be overwritten" in out


# ---------------------------------------------------------------------------
# execute_write_file — successful write
# ---------------------------------------------------------------------------

def test_execute_writes_file_when_approved(tmp_path):
    allowed = tmp_path / "output"
    allowed.mkdir()
    target = allowed / "result.py"
    cfg = _config(write_allowed_dirs=[allowed])
    args = {"path": str(target), "content": "x = 42\n", "description": "Answer"}

    with patch("builtins.input", return_value="y"):
        approved, result = execute_write_file(args, cfg)

    assert approved is True
    assert target.read_text() == "x = 42\n"


def test_execute_success_result_contains_path_and_size(tmp_path):
    allowed = tmp_path / "output"
    allowed.mkdir()
    target = allowed / "result.py"
    content = "x = 42\n"
    cfg = _config(write_allowed_dirs=[allowed])
    args = {"path": str(target), "content": content, "description": "Answer"}

    with patch("builtins.input", return_value="y"):
        approved, result = execute_write_file(args, cfg)

    assert "Written:" in result
    assert str(target.resolve()) in result
    assert str(len(content.encode())) in result


def test_execute_creates_parent_directories(tmp_path):
    allowed = tmp_path / "output"
    cfg = _config(write_allowed_dirs=[allowed])
    target = allowed / "deep" / "nested" / "file.py"
    args = {"path": str(target), "content": "pass\n", "description": "nested"}

    with patch("builtins.input", return_value="y"):
        approved, _ = execute_write_file(args, cfg)

    assert approved is True
    assert target.exists()
