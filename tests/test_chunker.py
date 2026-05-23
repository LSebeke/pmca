from pathlib import Path

import pytest

from pmca.rag.chunker import chunk_file
from pmca.types import Chunk


# ---------------------------------------------------------------------------
# Python chunking
# ---------------------------------------------------------------------------

def test_python_single_function(tmp_path):
    src = "def greet(name):\n    return f'Hello, {name}'\n"
    f = (tmp_path / "foo.py")
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 1
    assert chunks[0].label == "function `greet` (lines 1–2)"
    assert chunks[0].source_file == f


def test_python_single_class(tmp_path):
    src = "class Foo:\n    def bar(self):\n        pass\n"
    f = tmp_path / "foo.py"
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 1
    assert chunks[0].label == "class `Foo` (lines 1–3)"


def test_python_function_and_class(tmp_path):
    src = (
        "def helper():\n"
        "    pass\n"
        "\n"
        "class MyClass:\n"
        "    x = 1\n"
    )
    f = tmp_path / "foo.py"
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 2
    labels = [c.label for c in chunks]
    assert any("helper" in l for l in labels)
    assert any("MyClass" in l for l in labels)


def test_python_module_level_only(tmp_path):
    src = "x = 1\ny = 2\n"
    f = tmp_path / "foo.py"
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 1
    assert chunks[0].label == "module-level"


def test_python_module_level_plus_function(tmp_path):
    src = "CONSTANT = 42\n\ndef compute():\n    return CONSTANT * 2\n"
    f = tmp_path / "foo.py"
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 2
    labels = {c.label for c in chunks}
    assert "module-level" in labels
    assert any("compute" in l for l in labels)


def test_python_chunk_content_includes_source(tmp_path):
    src = "def add(a, b):\n    return a + b\n"
    f = tmp_path / "foo.py"
    f.write_text(src)
    chunks = chunk_file(f)
    assert "def add" in chunks[0].content
    assert "return a + b" in chunks[0].content


def test_python_label_line_range_is_accurate(tmp_path):
    src = "x = 1\n\ndef foo():\n    pass\n"
    # foo() is at lines 3–4
    f = tmp_path / "foo.py"
    f.write_text(src)
    chunks = chunk_file(f)
    func_chunk = next(c for c in chunks if "foo" in c.label)
    assert "lines 3–4" in func_chunk.label


def test_python_async_function(tmp_path):
    src = "async def fetch():\n    pass\n"
    f = tmp_path / "foo.py"
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 1
    assert "fetch" in chunks[0].label
    assert "function" in chunks[0].label


def test_python_empty_file(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("")
    chunks = chunk_file(f)
    assert chunks == []


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------

def test_md_splits_on_atx_headings(tmp_path):
    src = "# Introduction\nSome intro text.\n\n# Usage\nUsage details here.\n"
    f = tmp_path / "doc.md"
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 2
    assert chunks[0].label == "Introduction"
    assert chunks[1].label == "Usage"
    assert chunks[0].source_file == f


def test_md_chunk_content_includes_heading_and_body(tmp_path):
    src = "# Setup\nRun `pip install`.\n"
    f = tmp_path / "doc.md"
    f.write_text(src)
    chunks = chunk_file(f)
    assert "# Setup" in chunks[0].content
    assert "pip install" in chunks[0].content


def test_md_no_headings_returns_one_chunk(tmp_path):
    src = "Just a paragraph with no headings.\n"
    f = tmp_path / "doc.md"
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 1
    assert chunks[0].label == ""


def test_md_empty_file(tmp_path):
    f = tmp_path / "empty.md"
    f.write_text("")
    chunks = chunk_file(f)
    assert chunks == []


# ---------------------------------------------------------------------------
# Plain text chunking
# ---------------------------------------------------------------------------

def test_txt_splits_on_double_newlines(tmp_path):
    src = "Para one.\n\nPara two.\n\nPara three.\n"
    f = tmp_path / "notes.txt"
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 3


def test_txt_single_paragraph(tmp_path):
    src = "Just one paragraph with no blank lines.\n"
    f = tmp_path / "notes.txt"
    f.write_text(src)
    chunks = chunk_file(f)
    assert len(chunks) == 1


def test_txt_chunk_label_is_empty(tmp_path):
    src = "First.\n\nSecond.\n"
    f = tmp_path / "notes.txt"
    f.write_text(src)
    chunks = chunk_file(f)
    assert all(c.label == "" for c in chunks)


def test_txt_source_file_set_on_all_chunks(tmp_path):
    src = "First.\n\nSecond.\n"
    f = tmp_path / "notes.txt"
    f.write_text(src)
    chunks = chunk_file(f)
    assert all(c.source_file == f for c in chunks)


def test_txt_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    chunks = chunk_file(f)
    assert chunks == []
