from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from unittest.mock import MagicMock

from pmca.config import Config
from pmca.rag.store import VectorStore
from pmca.tools import (
    execute_edit_file,
    execute_get_definition,
    execute_list_dir,
    execute_rag_query,
    execute_read_file,
    execute_run_tests,
    execute_search,
    execute_write_file,
    get_tools,
)
from pmca.types import Chunk, ScratchpadEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(**overrides) -> Config:
    defaults = dict(
        name="test", model="gpt-4o-mini", system_prompt="You are helpful.",
        rag_files=[], log_folder=Path("/tmp/logs"),
        write_allowed_dirs=[], read_allowed_dirs=[],
    )
    defaults.update(overrides)
    return Config(**defaults)


def _empty_store() -> VectorStore:
    return VectorStore()


def _store_with_chunks(*labels: str) -> VectorStore:
    store = MagicMock(spec=VectorStore)
    chunks = [Chunk(content=f"content {l}", source_file=Path("/src/a.py"), label=l) for l in labels]
    store._chunks = chunks
    store.query.return_value = chunks
    return store


# ---------------------------------------------------------------------------
# get_tools
# ---------------------------------------------------------------------------

def test_get_tools_returns_none_when_no_allowed_dirs():
    cfg = _config(write_allowed_dirs=[], read_allowed_dirs=[])
    assert get_tools(cfg, _empty_store()) is None


def test_get_tools_returns_read_tools_when_read_dirs_configured(tmp_path):
    cfg = _config(read_allowed_dirs=[tmp_path])
    tools = get_tools(cfg, _empty_store())
    assert tools is not None
    names = {t["function"]["name"] for t in tools}
    assert {"read_file", "list_dir", "search", "get_definition"}.issubset(names)
    assert "save_to_scratchpad" in names


def test_get_tools_returns_all_tools_when_both_configured(tmp_path):
    cfg = _config(write_allowed_dirs=[tmp_path], read_allowed_dirs=[tmp_path])
    tools = get_tools(cfg, _empty_store())
    assert tools is not None
    names = {t["function"]["name"] for t in tools}
    assert {"write_file", "edit_file", "read_file", "list_dir", "search", "get_definition"}.issubset(names)
    assert "save_to_scratchpad" in names


def test_get_tools_read_description_lists_allowed_dirs(tmp_path):
    cfg = _config(read_allowed_dirs=[tmp_path])
    tools = get_tools(cfg, _empty_store())
    descs = " ".join(t["function"]["description"] for t in tools)
    assert str(tmp_path) in descs


def test_get_tools_returns_write_tools_when_write_dirs_configured(tmp_path):
    cfg = _config(write_allowed_dirs=[tmp_path])
    tools = get_tools(cfg, _empty_store())
    assert tools is not None
    names = {t["function"]["name"] for t in tools}
    assert {"write_file", "edit_file"}.issubset(names)
    assert "save_to_scratchpad" in names


def test_get_tools_includes_edit_file_when_write_dirs_configured(tmp_path):
    cfg = _config(write_allowed_dirs=[tmp_path])
    tools = get_tools(cfg, _empty_store())
    assert tools is not None
    names = {t["function"]["name"] for t in tools}
    assert "edit_file" in names


def test_get_tools_description_lists_allowed_dirs(tmp_path):
    allowed = tmp_path / "output"
    cfg = _config(write_allowed_dirs=[allowed])
    tools = get_tools(cfg, _empty_store())
    description = tools[0]["function"]["description"]
    assert str(allowed) in description


def test_get_tools_schema_has_required_parameters(tmp_path):
    cfg = _config(write_allowed_dirs=[tmp_path])
    tools = get_tools(cfg, _empty_store())
    params = tools[0]["function"]["parameters"]
    required = params["required"]
    assert "path" in required
    assert "content" in required
    assert "description" in required


def test_get_tools_includes_save_to_scratchpad_whenever_tools_registered(tmp_path):
    cfg = _config(read_allowed_dirs=[tmp_path])
    tools = get_tools(cfg, _empty_store())
    names = {t["function"]["name"] for t in tools}
    assert "save_to_scratchpad" in names


def test_get_tools_omits_save_to_scratchpad_when_no_tools_registered():
    cfg = _config(write_allowed_dirs=[], read_allowed_dirs=[])
    assert get_tools(cfg, _empty_store()) is None


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
# execute_edit_file
# ---------------------------------------------------------------------------

def test_edit_file_returns_error_when_outside_allowed_dirs(tmp_path):
    allowed = tmp_path / "allowed"
    cfg = _config(write_allowed_dirs=[allowed])
    args = {"path": str(tmp_path / "sneaky.py"), "old_string": "x", "new_string": "y", "description": "t"}
    ok, msg = execute_edit_file(args, cfg)
    assert ok is False
    assert "outside allowed" in msg.lower()


def test_edit_file_returns_error_when_file_not_found(tmp_path):
    cfg = _config(write_allowed_dirs=[tmp_path])
    args = {"path": str(tmp_path / "missing.py"), "old_string": "x", "new_string": "y", "description": "t"}
    ok, msg = execute_edit_file(args, cfg)
    assert ok is False
    assert "not found" in msg.lower()


def test_edit_file_returns_error_when_old_string_not_found(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    cfg = _config(write_allowed_dirs=[tmp_path])
    args = {"path": str(f), "old_string": "zzz_no_such_string", "new_string": "y", "description": "t"}
    ok, msg = execute_edit_file(args, cfg)
    assert ok is False
    assert "not found" in msg.lower()


def test_edit_file_returns_error_when_old_string_is_ambiguous(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\nx = 1\n")
    cfg = _config(write_allowed_dirs=[tmp_path])
    args = {"path": str(f), "old_string": "x = 1", "new_string": "y = 2", "description": "t"}
    ok, msg = execute_edit_file(args, cfg)
    assert ok is False
    assert "ambiguous" in msg.lower()
    assert "2" in msg


def test_edit_file_prints_approval_prompt(tmp_path, capsys):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    cfg = _config(write_allowed_dirs=[tmp_path])
    args = {"path": str(f), "old_string": "x = 1", "new_string": "x = 2", "description": "increment x"}

    with patch("builtins.input", return_value="n"):
        execute_edit_file(args, cfg)

    out = capsys.readouterr().out
    assert str(f.resolve()) in out
    assert "increment x" in out
    assert "x = 1" in out
    assert "x = 2" in out
    assert "--- remove ---" in out
    assert "--- insert ---" in out


def test_edit_file_returns_denial_when_user_denies(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\n")
    cfg = _config(write_allowed_dirs=[tmp_path])
    args = {"path": str(f), "old_string": "x = 1", "new_string": "x = 2", "description": "t"}

    with patch("builtins.input", return_value="n"):
        ok, msg = execute_edit_file(args, cfg)

    assert ok is False
    assert "Edit denied by user" in msg
    assert str(f.resolve()) in msg


def test_edit_file_replaces_and_writes_on_approval(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("x = 1\ny = 2\n")
    cfg = _config(write_allowed_dirs=[tmp_path])
    args = {"path": str(f), "old_string": "x = 1", "new_string": "x = 99", "description": "t"}

    with patch("builtins.input", return_value="y"):
        ok, msg = execute_edit_file(args, cfg)

    assert ok is True
    assert "Edited:" in msg
    assert str(f.resolve()) in msg
    assert f.read_text() == "x = 99\ny = 2\n"


def test_edit_file_replaces_only_first_when_one_occurrence(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("a = 1\nb = 2\n")
    cfg = _config(write_allowed_dirs=[tmp_path])
    args = {"path": str(f), "old_string": "a = 1", "new_string": "a = 42", "description": "t"}

    with patch("builtins.input", return_value="y"):
        execute_edit_file(args, cfg)

    assert f.read_text() == "a = 42\nb = 2\n"


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


# ---------------------------------------------------------------------------
# get_tools — run_tests registration
# ---------------------------------------------------------------------------

def test_get_tools_includes_run_tests_when_test_dir_configured(tmp_path):
    cfg = _config(test_dir=tmp_path)
    tools = get_tools(cfg, _empty_store())
    assert tools is not None
    names = {t["function"]["name"] for t in tools}
    assert "run_tests" in names


def test_get_tools_excludes_run_tests_when_test_dir_is_none():
    cfg = _config()
    assert cfg.test_dir is None
    tools = get_tools(cfg, _empty_store())
    assert tools is None  # no tools at all when nothing configured


def test_get_tools_run_tests_has_optional_filter_parameter(tmp_path):
    cfg = _config(test_dir=tmp_path)
    tools = get_tools(cfg, _empty_store())
    schema = next(t for t in tools if t["function"]["name"] == "run_tests")
    params = schema["function"]["parameters"]
    assert "filter" in params["properties"]
    assert "filter" not in params.get("required", [])


# ---------------------------------------------------------------------------
# get_tools — query_knowledge_base registration
# ---------------------------------------------------------------------------

def test_get_tools_includes_rag_when_store_has_content(tmp_path):
    cfg = _config()
    store = _store_with_chunks("fn `foo`")
    tools = get_tools(cfg, store)
    assert tools is not None
    names = {t["function"]["name"] for t in tools}
    assert "query_knowledge_base" in names


def test_get_tools_excludes_rag_when_store_is_empty():
    cfg = _config()
    tools = get_tools(cfg, _empty_store())
    assert tools is None


def test_get_tools_rag_schema_has_query_and_depth(tmp_path):
    cfg = _config()
    store = _store_with_chunks("fn `foo`")
    tools = get_tools(cfg, store)
    schema = next(t for t in tools if t["function"]["name"] == "query_knowledge_base")
    props = schema["function"]["parameters"]["properties"]
    assert "query" in props
    assert "depth" in props
    assert schema["function"]["parameters"]["required"] == ["query", "depth"]


# ---------------------------------------------------------------------------
# execute_rag_query
# ---------------------------------------------------------------------------

def test_execute_rag_query_shallow_returns_at_most_shallow_k():
    cfg = _config(rag_shallow_k=2)
    store = _store_with_chunks("fn `a`", "fn `b`", "fn `c`")
    store.query.return_value = [
        Chunk(content="a", source_file=Path("/a.py"), label="fn `a`"),
        Chunk(content="b", source_file=Path("/a.py"), label="fn `b`"),
    ]
    turn_seen: set = set()
    result = execute_rag_query({"query": "foo", "depth": "shallow"}, cfg, store, turn_seen)
    store.query.assert_called_once_with("foo", 2)
    assert "[RAG_1]" in result
    assert "[RAG_2]" in result
    assert "[RAG_3]" not in result


def test_execute_rag_query_deep_uses_deep_k():
    cfg = _config(rag_deep_k=15)
    store = _store_with_chunks("fn `x`")
    store.query.return_value = [Chunk(content="x", source_file=Path("/a.py"), label="fn `x`")]
    turn_seen: set = set()
    execute_rag_query({"query": "q", "depth": "deep"}, cfg, store, turn_seen)
    store.query.assert_called_once_with("q", 15)


def test_execute_rag_query_adds_chunks_to_turn_seen():
    cfg = _config()
    chunk = Chunk(content="c", source_file=Path("/a.py"), label="fn `foo`")
    store = MagicMock(spec=VectorStore)
    store._chunks = [chunk]
    store.query.return_value = [chunk]
    turn_seen: set = set()
    execute_rag_query({"query": "q", "depth": "shallow"}, cfg, store, turn_seen)
    assert (Path("/a.py"), "fn `foo`") in turn_seen


def test_execute_rag_query_second_call_returns_only_new_chunks():
    cfg = _config(rag_shallow_k=3)
    chunk_a = Chunk(content="a", source_file=Path("/a.py"), label="fn `a`")
    chunk_b = Chunk(content="b", source_file=Path("/a.py"), label="fn `b`")
    store = MagicMock(spec=VectorStore)
    store._chunks = [chunk_a, chunk_b]
    store.query.return_value = [chunk_a, chunk_b]
    turn_seen: set = set()

    execute_rag_query({"query": "q", "depth": "shallow"}, cfg, store, turn_seen)
    result2 = execute_rag_query({"query": "q", "depth": "shallow"}, cfg, store, turn_seen)

    assert "fn `a`" not in result2
    assert "fn `b`" not in result2
    assert "No results found." in result2


def test_execute_rag_query_returns_no_results_when_all_seen():
    cfg = _config()
    chunk = Chunk(content="c", source_file=Path("/a.py"), label="fn `foo`")
    store = MagicMock(spec=VectorStore)
    store._chunks = [chunk]
    store.query.return_value = [chunk]
    turn_seen = {(Path("/a.py"), "fn `foo`")}

    result = execute_rag_query({"query": "q", "depth": "shallow"}, cfg, store, turn_seen)
    assert result == "No results found."


# ---------------------------------------------------------------------------
# execute_run_tests — command detection
# ---------------------------------------------------------------------------

def test_run_tests_uses_pixi_when_pixi_toml_present(tmp_path, capsys):
    (tmp_path / "pixi.toml").write_text("")
    cfg = _config(test_dir=tmp_path)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "1 passed"
        mock_run.return_value.returncode = 0
        execute_run_tests({}, cfg)
    cmd = mock_run.call_args[0][0]
    assert cmd[:3] == ["pixi", "run", "pytest"]


def test_run_tests_uses_pytest_when_no_pixi_toml(tmp_path, capsys):
    cfg = _config(test_dir=tmp_path)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "1 passed"
        mock_run.return_value.returncode = 0
        execute_run_tests({}, cfg)
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "pytest"
    assert "pixi" not in cmd


# ---------------------------------------------------------------------------
# execute_run_tests — filter appended
# ---------------------------------------------------------------------------

def test_run_tests_appends_filter_tokens(tmp_path):
    cfg = _config(test_dir=tmp_path)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "1 passed"
        mock_run.return_value.returncode = 0
        execute_run_tests({"filter": "tests/test_foo.py -k bar"}, cfg)
    cmd = mock_run.call_args[0][0]
    assert "tests/test_foo.py" in cmd
    assert "-k" in cmd
    assert "bar" in cmd


def test_run_tests_no_extra_tokens_when_filter_absent(tmp_path):
    cfg = _config(test_dir=tmp_path)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "passed"
        mock_run.return_value.returncode = 0
        execute_run_tests({}, cfg)
    cmd = mock_run.call_args[0][0]
    assert cmd == ["pytest"]


# ---------------------------------------------------------------------------
# execute_run_tests — prints command, returns output
# ---------------------------------------------------------------------------

def test_run_tests_prints_command_before_running(tmp_path, capsys):
    cfg = _config(test_dir=tmp_path)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "passed"
        mock_run.return_value.returncode = 0
        execute_run_tests({}, cfg)
    out = capsys.readouterr().out
    assert "[run_tests]" in out
    assert "pytest" in out


def test_run_tests_returns_true_and_output_on_pass(tmp_path):
    cfg = _config(test_dir=tmp_path)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "3 passed"
        mock_run.return_value.returncode = 0
        ok, result = execute_run_tests({}, cfg)
    assert ok is True
    assert "3 passed" in result


def test_run_tests_returns_true_and_output_on_test_failure(tmp_path):
    cfg = _config(test_dir=tmp_path)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "1 failed, 2 passed"
        mock_run.return_value.returncode = 1
        ok, result = execute_run_tests({}, cfg)
    assert ok is True
    assert "1 failed" in result


# ---------------------------------------------------------------------------
# execute_run_tests — timeout and OSError
# ---------------------------------------------------------------------------

def test_run_tests_returns_false_on_timeout(tmp_path):
    import subprocess
    cfg = _config(test_dir=tmp_path, test_timeout=5)
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pytest", timeout=5)):
        ok, result = execute_run_tests({}, cfg)
    assert ok is False
    assert "timed out" in result.lower()
    assert "5" in result


def test_run_tests_returns_false_on_oserror(tmp_path):
    cfg = _config(test_dir=tmp_path)
    with patch("subprocess.run", side_effect=OSError("no such file")):
        ok, result = execute_run_tests({}, cfg)
    assert ok is False
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# ScratchpadEntry
# ---------------------------------------------------------------------------

def test_scratchpad_entry_has_title_and_content():
    entry = ScratchpadEntry(title="read_file: src/foo.py — init", content="def __init__(): pass")
    assert entry.title == "read_file: src/foo.py — init"
    assert entry.content == "def __init__(): pass"


# ---------------------------------------------------------------------------
# execute_save_to_scratchpad
# ---------------------------------------------------------------------------

def test_scratchpad_import():
    from pmca.tools import execute_save_to_scratchpad  # noqa: F401


def test_scratchpad_delete_removes_entry_by_title():
    from pmca.tools import execute_save_to_scratchpad
    cfg = _config(max_scratchpad_entries=20)
    scratchpad = [ScratchpadEntry(title="foo", content="bar")]
    execute_save_to_scratchpad({"delete": ["foo"]}, cfg, scratchpad)
    assert scratchpad == []


def test_scratchpad_delete_unknown_title_is_silently_ignored():
    from pmca.tools import execute_save_to_scratchpad
    cfg = _config(max_scratchpad_entries=20)
    scratchpad = [ScratchpadEntry(title="foo", content="bar")]
    execute_save_to_scratchpad({"delete": ["nonexistent"]}, cfg, scratchpad)
    assert len(scratchpad) == 1


def test_scratchpad_add_new_entry():
    from pmca.tools import execute_save_to_scratchpad
    cfg = _config(max_scratchpad_entries=20)
    scratchpad = []
    execute_save_to_scratchpad({"entries": [{"title": "t1", "content": "c1"}]}, cfg, scratchpad)
    assert len(scratchpad) == 1
    assert scratchpad[0].title == "t1"
    assert scratchpad[0].content == "c1"


def test_scratchpad_overwrite_existing_title():
    from pmca.tools import execute_save_to_scratchpad
    cfg = _config(max_scratchpad_entries=20)
    scratchpad = [ScratchpadEntry(title="t1", content="old")]
    execute_save_to_scratchpad({"entries": [{"title": "t1", "content": "new"}]}, cfg, scratchpad)
    assert len(scratchpad) == 1
    assert scratchpad[0].content == "new"


def test_scratchpad_overwrite_does_not_count_against_cap():
    from pmca.tools import execute_save_to_scratchpad
    cfg = _config(max_scratchpad_entries=1)
    scratchpad = [ScratchpadEntry(title="t1", content="old")]
    result = execute_save_to_scratchpad({"entries": [{"title": "t1", "content": "new"}]}, cfg, scratchpad)
    assert "Error" not in result
    assert scratchpad[0].content == "new"


def test_scratchpad_delete_then_add_in_one_call():
    from pmca.tools import execute_save_to_scratchpad
    cfg = _config(max_scratchpad_entries=1)
    scratchpad = [ScratchpadEntry(title="old", content="x")]
    result = execute_save_to_scratchpad(
        {"delete": ["old"], "entries": [{"title": "new", "content": "y"}]},
        cfg, scratchpad,
    )
    assert "Error" not in result
    assert len(scratchpad) == 1
    assert scratchpad[0].title == "new"


def test_scratchpad_cap_exceeded_returns_error_and_does_not_apply():
    from pmca.tools import execute_save_to_scratchpad
    cfg = _config(max_scratchpad_entries=1)
    scratchpad = [ScratchpadEntry(title="existing", content="x")]
    result = execute_save_to_scratchpad(
        {"entries": [{"title": "new1", "content": "a"}, {"title": "new2", "content": "b"}]},
        cfg, scratchpad,
    )
    assert "Error" in result
    assert len(scratchpad) == 1  # not partially applied


def test_scratchpad_summary_string_format():
    from pmca.tools import execute_save_to_scratchpad
    cfg = _config(max_scratchpad_entries=20)
    scratchpad = [ScratchpadEntry(title="old", content="x")]
    result = execute_save_to_scratchpad(
        {"delete": ["old"], "entries": [{"title": "n1", "content": "a"}, {"title": "n2", "content": "b"}]},
        cfg, scratchpad,
    )
    assert "Deleted 1" in result
    assert "Saved 2" in result
    assert "[Scratchpad: 2 entries]" in result
