from __future__ import annotations

import ast
import re
from pathlib import Path

from pmca.types import Chunk

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)


def chunk_file(path: Path) -> list[Chunk]:
    if path.suffix == ".py":
        return _chunk_python(path)
    return _chunk_prose(path)


def _chunk_python(path: Path) -> list[Chunk]:
    source = path.read_text(encoding="utf-8")
    if not source.strip():
        return []

    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=str(path))

    top_level = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    chunks: list[Chunk] = []
    covered: set[int] = set()

    for node in top_level:
        start = node.lineno
        end = node.end_lineno
        covered.update(range(start, end + 1))
        content = "".join(lines[start - 1:end])
        label = _python_label(node)
        chunks.append(Chunk(content=content, source_file=path, label=label))

    module_lines = [
        line for i, line in enumerate(lines, start=1)
        if i not in covered
    ]
    module_content = "".join(module_lines).strip()
    if module_content:
        chunks.insert(0, Chunk(content=module_content, source_file=path, label="module-level"))

    return chunks


def _python_label(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    return f"{kind} `{node.name}` (lines {node.lineno}–{node.end_lineno})"


def _chunk_prose(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    if path.suffix == ".md":
        return _chunk_markdown(text, path)
    return _chunk_paragraphs(text, path)


def _chunk_markdown(text: str, path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_label = ""
    current_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        m = re.match(r"^(#{1,6})\s+(.+)", line)
        if m:
            if current_lines or current_label:
                content = "".join(current_lines).strip()
                if content:
                    chunks.append(Chunk(content=content, source_file=path, label=current_label))
            current_label = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines or current_label:
        content = "".join(current_lines).strip()
        if content:
            chunks.append(Chunk(content=content, source_file=path, label=current_label))

    return chunks


def _chunk_paragraphs(text: str, path: Path) -> list[Chunk]:
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[Chunk] = []
    for para in paragraphs:
        content = para.strip()
        if content:
            chunks.append(Chunk(content=content, source_file=path, label=""))
    return chunks
