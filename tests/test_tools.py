from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pmca.config import Config
from pmca.tools import (
    execute_get_definition,
    execute_list_dir,
    execute_read_file,
    execute_search,
    execute_write_file,
    get_tools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides) -> Config:
    defaults = dict(
        name="test", model="gpt-4o-mini", system_prompt="You are helpful.",
        rag_files=[], top_k_chunks=3, log_folder=Path("/tmp/logs"),
        write_allowed_dirs=[], read_allowed_dirs=[],
    )
    defaults.update(overrides)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# get_tools
# ---------------------------------------------------------------------------

def test_get_tools_returns_none_when_no_allowed_dirs():
    cfg = _config(write_allowed_dirs=[], read_allowed_dirs=[])
    assert get_tools(cfg) is None


def test_get_tools_returns_read_tools_when_read_dirs_configured(tmp_path):
    cfg = _config(read_allowed_dirs=[tmp_path])
    tools = get_tools(cfg)
    assert tools is not None
    names = {t["function"]["name"] for t in tools}
    assert names == {"read_file", "list_dir", "search", "get_definition"}


def test_get_tools_returns_all_tools_when_both_configured(tmp_path):
    cfg = _config(write_allowed_dirs=[tmp_path], read_allowed_dirs=[tmp_path])
    tools = get_tools(cfg)
    assert tools is not None
    names = {t["function"]["name"] for t in tools}
    assert names == {"write_file", "read_file", "list_dir", "search", "get_definition"}


def test_get_tools_read_description_lists_allowed_dirs(tmp_path):
    cfg = _config(read_allowed_dirs=[tmp_path])
    tools = get_tools(cfg)
    descs = " ".join(t["function"]["description"] for t in tools)
    assert str(tmp_path) in descs


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
# execute_read_file
# ---------------------------------------------------------------------------

def test_read_file_returns_error_when_outside_allowed_dirs(tmp_path):
    allowed = tmp_path / "allowed"
    cfg = _config(read_allowed_dirs=[allowed])
    result = execute_read_file({"path": str(tmp_path / "secret.py")}, cfg)
    assert "outside allowed" in result.lower()


def test_read_file_returns_content_on_success(tmp_path):
    allowed = tmp_path / "src"
    allowed.mkdir()
    f = allowed / "foo.py"
    f.write_text("x = 1\n")
    cfg = _config(read_allowed_dirs=[allowed])
    assert execute_read_file({"path": str(f)}, cfg) == "x = 1\n"


def test_read_file_returns_error_when_file_not_found(tmp_path):
    allowed = tmp_path / "src"
    allowed.mkdir()
    cfg = _config(read_allowed_dirs=[allowed])
    result = execute_read_file({"path": str(allowed / "missing.py")}, cfg)
    assert "not found" in result.lower() or "error" in result.lower()


# ---------------------------------------------------------------------------
# execute_list_dir
# ---------------------------------------------------------------------------

def test_list_dir_returns_error_when_outside_allowed_dirs(tmp_path):
    allowed = tmp_path / "allowed"
    cfg = _config(read_allowed_dirs=[allowed])
    result = execute_list_dir({"path": str(tmp_path / "other"), "recursive": False}, cfg)
    assert "outside allowed" in result.lower()


def test_list_dir_returns_immediate_children(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("")
    (src / "b.py").write_text("")
    (src / "sub").mkdir()
    (src / "sub" / "c.py").write_text("")
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_list_dir({"path": str(src), "recursive": False}, cfg)
    listed = result.splitlines()
    assert any("a.py" in l for l in listed)
    assert any("b.py" in l for l in listed)
    assert not any("c.py" in l for l in listed)


def test_list_dir_returns_full_tree_when_recursive(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("")
    (src / "sub").mkdir()
    (src / "sub" / "c.py").write_text("")
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_list_dir({"path": str(src), "recursive": True}, cfg)
    assert "c.py" in result


def test_list_dir_returns_error_when_not_a_directory(tmp_path):
    f = tmp_path / "file.py"
    f.write_text("")
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_list_dir({"path": str(f), "recursive": False}, cfg)
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# execute_search
# ---------------------------------------------------------------------------

def test_search_returns_error_when_outside_allowed_dirs(tmp_path):
    allowed = tmp_path / "allowed"
    cfg = _config(read_allowed_dirs=[allowed])
    result = execute_search({"path": str(tmp_path / "other"), "pattern": "foo"}, cfg)
    assert "outside allowed" in result.lower()


def test_search_returns_matches_with_context(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("line1\nline2\nfoo = 1\nline4\nline5\n")
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_search({"path": str(f), "pattern": "foo", "context_lines": 1}, cfg)
    assert "foo" in result
    assert "line2" in result
    assert "line4" in result


def test_search_searches_directory_recursively(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.py").write_text("needle here\n")
    (tmp_path / "b.py").write_text("nothing\n")
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_search({"path": str(tmp_path), "pattern": "needle"}, cfg)
    assert "needle" in result


def test_search_returns_no_matches_message(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("hello world\n")
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_search({"path": str(f), "pattern": "zzznomatch"}, cfg)
    assert result == "No matches found."


def test_search_returns_error_on_invalid_regex(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x\n")
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_search({"path": str(f), "pattern": "["}, cfg)
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# execute_get_definition
# ---------------------------------------------------------------------------

_PY_SOURCE = """\
def standalone():
    return 42


class MyClass:
    def method(self):
        return "hello"

    def other(self):
        pass
"""


def test_get_definition_returns_error_when_outside_allowed_dirs(tmp_path):
    allowed = tmp_path / "allowed"
    cfg = _config(read_allowed_dirs=[allowed])
    result = execute_get_definition({"path": str(tmp_path / "x.py"), "symbol": "foo"}, cfg)
    assert "outside allowed" in result.lower()


def test_get_definition_returns_error_for_non_py_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello")
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_get_definition({"path": str(f), "symbol": "foo"}, cfg)
    assert "error" in result.lower()


def test_get_definition_returns_top_level_function(tmp_path):
    f = tmp_path / "code.py"
    f.write_text(_PY_SOURCE)
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_get_definition({"path": str(f), "symbol": "standalone"}, cfg)
    assert "def standalone" in result
    assert "return 42" in result


def test_get_definition_returns_class(tmp_path):
    f = tmp_path / "code.py"
    f.write_text(_PY_SOURCE)
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_get_definition({"path": str(f), "symbol": "MyClass"}, cfg)
    assert "class MyClass" in result
    assert "def method" in result


def test_get_definition_returns_method_via_dotted_symbol(tmp_path):
    f = tmp_path / "code.py"
    f.write_text(_PY_SOURCE)
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_get_definition({"path": str(f), "symbol": "MyClass.method"}, cfg)
    assert "def method" in result
    assert 'return "hello"' in result
    assert "class MyClass" not in result


def test_get_definition_returns_error_when_symbol_not_found(tmp_path):
    f = tmp_path / "code.py"
    f.write_text(_PY_SOURCE)
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_get_definition({"path": str(f), "symbol": "no_such_symbol"}, cfg)
    assert "not found" in result.lower() or "error" in result.lower()


def test_get_definition_includes_decorators(tmp_path):
    source = "@staticmethod\ndef decorated():\n    pass\n"
    f = tmp_path / "code.py"
    f.write_text(source)
    cfg = _config(read_allowed_dirs=[tmp_path])
    result = execute_get_definition({"path": str(f), "symbol": "decorated"}, cfg)
    assert "@staticmethod" in result


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
