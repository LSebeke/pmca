from __future__ import annotations

import ast
import re
from pathlib import Path

from pmca.config import Config

_WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the file to write."},
                "content": {"type": "string", "description": "Full content to write to the file (UTF-8)."},
                "description": {"type": "string", "description": "Short human-readable explanation of what is being written and why."},
            },
            "required": ["path", "content", "description"],
            "additionalProperties": False,
        },
    },
}

_READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the file to read."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

_LIST_DIR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the directory to list."},
                "recursive": {"type": "boolean", "description": "If true, list all descendants; if false, immediate children only."},
            },
            "required": ["path", "recursive"],
            "additionalProperties": False,
        },
    },
}

_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of file or directory to search."},
                "pattern": {"type": "string", "description": "Regex pattern to search for."},
                "context_lines": {"type": "integer", "description": "Number of lines of context to include before and after each match.", "default": 3},
            },
            "required": ["path", "pattern"],
            "additionalProperties": False,
        },
    },
}

_GET_DEFINITION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_definition",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the .py file."},
                "symbol": {"type": "string", "description": "Symbol name, e.g. 'my_func' or 'MyClass.my_method'."},
            },
            "required": ["path", "symbol"],
            "additionalProperties": False,
        },
    },
}


def get_tools(config: Config) -> list[dict] | None:
    tools = []

    if config.write_allowed_dirs:
        dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
        schema = {
            **_WRITE_FILE_SCHEMA,
            "function": {
                **_WRITE_FILE_SCHEMA["function"],
                "description": f"Write a file to disk. Allowed directories: {dirs_str}",
            },
        }
        tools.append(schema)

    if config.read_allowed_dirs:
        dirs_str = ", ".join(str(d) for d in config.read_allowed_dirs)
        desc_suffix = f" Allowed directories: {dirs_str}"

        for base_schema, desc in [
            (_READ_FILE_SCHEMA, "Read a file from disk." + desc_suffix),
            (_LIST_DIR_SCHEMA, "List directory contents." + desc_suffix),
            (_SEARCH_SCHEMA, "Search for a regex pattern in a file or directory tree." + desc_suffix),
            (_GET_DEFINITION_SCHEMA, "Get the full source of a Python function or class." + desc_suffix),
        ]:
            tools.append({
                **base_schema,
                "function": {**base_schema["function"], "description": desc},
            })

    return tools if tools else None


def execute_write_file(arguments: dict, config: Config) -> tuple[bool, str]:
    raw_path = arguments["path"]
    content = arguments["content"]
    reason = arguments.get("description", "")

    target = Path(raw_path).resolve()

    if not _is_allowed(target, config.write_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
        return False, f"Error: path {target} is outside allowed directories: {dirs_str}"

    size = len(content.encode())
    exists_msg = "File exists — will be overwritten." if target.exists() else "File does not exist."

    print(f"[write_file] {target} ({size} bytes)")
    print(f"Reason: {reason}")
    print(f"{exists_msg} Approve? [y/N] ", end="", flush=True)
    answer = input()

    if answer.strip().lower() != "y":
        return False, f"Write denied by user. Path: {target}"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return True, f"Written: {target} ({size} bytes)"


def execute_read_file(arguments: dict, config: Config) -> str:
    target = Path(arguments["path"]).resolve()

    if not _is_allowed(target, config.read_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.read_allowed_dirs)
        return f"Error: path {target} is outside allowed directories: {dirs_str}"

    try:
        return target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {target}"
    except OSError as e:
        return f"Error reading {target}: {e}"


def execute_list_dir(arguments: dict, config: Config) -> str:
    target = Path(arguments["path"]).resolve()
    recursive = arguments.get("recursive", False)

    if not _is_allowed(target, config.read_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.read_allowed_dirs)
        return f"Error: path {target} is outside allowed directories: {dirs_str}"

    if not target.exists():
        return f"Error: path not found: {target}"
    if not target.is_dir():
        return f"Error: not a directory: {target}"

    if recursive:
        paths = sorted(p for p in target.rglob("*"))
    else:
        paths = sorted(target.iterdir())

    return "\n".join(str(p) for p in paths) if paths else ""


def execute_search(arguments: dict, config: Config) -> str:
    target = Path(arguments["path"]).resolve()
    pattern = arguments["pattern"]
    context_lines = int(arguments.get("context_lines", 3))

    if not _is_allowed(target, config.read_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.read_allowed_dirs)
        return f"Error: path {target} is outside allowed directories: {dirs_str}"

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    if not target.exists():
        return f"Error: path not found: {target}"

    files = sorted(target.rglob("*") if target.is_dir() else [target])
    files = [f for f in files if f.is_file()]

    all_groups: list[str] = []
    for fpath in files:
        try:
            lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        groups = _search_file(fpath, lines, regex, context_lines)
        all_groups.extend(groups)

    if not all_groups:
        return "No matches found."
    return "\n--\n".join(all_groups)


def execute_get_definition(arguments: dict, config: Config) -> str:
    target = Path(arguments["path"]).resolve()
    symbol = arguments["symbol"]

    if not _is_allowed(target, config.read_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.read_allowed_dirs)
        return f"Error: path {target} is outside allowed directories: {dirs_str}"

    if target.suffix != ".py":
        return f"Error: get_definition requires a .py file, got: {target}"

    try:
        source = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found: {target}"
    except OSError as e:
        return f"Error reading {target}: {e}"

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"Error: syntax error in {target}: {e}"

    lines = source.splitlines(keepends=True)
    parts = symbol.split(".", 1)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == parts[0]:
                if len(parts) == 1:
                    return _extract_node_source(node, lines)
                # looking for a method inside this class
                if isinstance(node, ast.ClassDef):
                    for child in ast.walk(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name == parts[1]:
                                return _extract_node_source(child, lines)
                    return f"Error: symbol '{symbol}' not found in {target}"

    return f"Error: symbol '{parts[0]}' not found in {target}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_allowed(target: Path, allowed_dirs: list[Path]) -> bool:
    for d in allowed_dirs:
        try:
            target.relative_to(d)
            return True
        except ValueError:
            continue
    return False


def _search_file(fpath: Path, lines: list[str], regex: re.Pattern, context_lines: int) -> list[str]:
    groups: list[str] = []
    i = 0
    while i < len(lines):
        if regex.search(lines[i]):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            block_lines = []
            for j in range(start, end):
                prefix = ">" if j == i else " "
                block_lines.append(f"{fpath}:{j + 1}{prefix} {lines[j]}")
            groups.append("\n".join(block_lines))
            i = end
        else:
            i += 1
    return groups


def _extract_node_source(node: ast.AST, lines: list[str]) -> str:
    start = node.lineno - 1
    end = node.end_lineno
    # include decorators
    if hasattr(node, "decorator_list") and node.decorator_list:
        start = node.decorator_list[0].lineno - 1
    return "".join(lines[start:end])
