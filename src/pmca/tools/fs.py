from __future__ import annotations

import ast
import difflib
import re
from pathlib import Path

from pmca.config import Config


def execute_write_file(arguments: dict, config: Config, turn_read_files: set[Path]) -> tuple[bool, str]:
    raw_path = arguments["path"]
    content = arguments["content"]
    reason = arguments.get("description", "")

    target = Path(raw_path).resolve()

    if not _is_allowed(target, config.write_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
        return False, f"Error: path {target} is outside allowed directories: {dirs_str}"

    if target.exists() and target not in turn_read_files:
        return False, f"Error: {target} has not been read this turn. Call read_file first."

    size = len(content.encode())
    exists_msg = "File exists — will be overwritten." if target.exists() else "File does not exist."

    if not config.auto_approve_writes:
        print(f"[write_file] {target} ({size} bytes)")
        print(f"Reason: {reason}")
        if target.exists():
            old = target.read_text(encoding="utf-8")
            _print_unified_diff(old, content, target)
        print(f"{exists_msg} Approve? [y/N] ", end="", flush=True)
        answer = input()
        if answer.strip().lower() != "y":
            return False, f"Write denied by user. Path: {target}"
    elif config.show_diff_on_auto_approve and target.exists():
        old = target.read_text(encoding="utf-8")
        _print_unified_diff(old, content, target)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    turn_read_files.discard(target)
    return True, f"Written: {target} ({size} bytes). Re-read required before next edit."


def execute_edit_file(arguments: dict, config: Config, turn_read_files: set[Path]) -> tuple[bool, str]:
    raw_path = arguments["path"]
    old_string = arguments["old_string"]
    new_string = arguments["new_string"]
    reason = arguments.get("description", "")

    target = Path(raw_path).resolve()

    if not _is_allowed(target, config.write_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
        return False, f"Error: path {target} is outside allowed directories: {dirs_str}"

    if not target.exists():
        return False, f"Error: file not found: {target}"

    if target not in turn_read_files:
        return False, f"Error: {target} has not been read this turn. Call read_file first."

    try:
        content = target.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"Error reading {target}: {e}"

    count = content.count(old_string)
    if count == 0:
        return False, f"Error: old_string not found in {target}"
    if count > 1:
        return False, f"Error: old_string is ambiguous ({count} occurrences) in {target}; provide more context"

    new_content = content.replace(old_string, new_string, 1)

    if not config.auto_approve_writes:
        print(f"[edit_file] {target}")
        print(f"Reason: {reason}")
        _print_unified_diff(content, new_content, target)
        print("Approve? [y/N] ", end="", flush=True)
        answer = input()
        if answer.strip().lower() != "y":
            return False, f"Edit denied by user. Path: {target}"
    elif config.show_diff_on_auto_approve:
        _print_unified_diff(content, new_content, target)
    try:
        target.write_text(new_content, encoding="utf-8")
    except OSError as e:
        return False, f"Error writing {target}: {e}"

    turn_read_files.discard(target)
    return True, f"Edited: {target}. Re-read required before next edit."


def execute_insert_at_line(arguments: dict, config: Config, turn_read_files: set[Path]) -> tuple[bool, str]:
    raw_path = arguments["path"]
    line_number = int(arguments["line_number"])
    content = arguments["content"]
    mode = arguments["mode"]
    reason = arguments.get("description", "")

    target = Path(raw_path).resolve()

    if not _is_allowed(target, config.write_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
        return False, f"Error: path {target} is outside allowed directories: {dirs_str}"

    if not target.exists():
        return False, f"Error: file not found: {target}"

    if target not in turn_read_files:
        return False, f"Error: {target} has not been read this turn. Call read_file first."

    try:
        lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as e:
        return False, f"Error reading {target}: {e}"

    if line_number < 1 or line_number > len(lines):
        return False, f"Error: line_number {line_number} is out of range (file has {len(lines)} lines)"

    idx = line_number - 1

    new_lines = list(lines)
    if mode == "before":
        new_lines.insert(idx, content if content.endswith("\n") else content + "\n")
    elif mode == "after":
        new_lines.insert(idx + 1, content if content.endswith("\n") else content + "\n")
    elif mode == "replace":
        new_lines[idx] = content if content.endswith("\n") else content + "\n"

    if not config.auto_approve_writes:
        print(f"[insert_at_line] {target} (line {line_number}, mode={mode})")
        print(f"Reason: {reason}")
        _print_unified_diff("".join(lines), "".join(new_lines), target)
        print("Approve? [y/N] ", end="", flush=True)
        answer = input()
        if answer.strip().lower() != "y":
            return False, f"Edit denied by user. Path: {target}"
    elif config.show_diff_on_auto_approve:
        _print_unified_diff("".join(lines), "".join(new_lines), target)

    if mode == "before":
        lines.insert(idx, content if content.endswith("\n") else content + "\n")
    elif mode == "after":
        lines.insert(idx + 1, content if content.endswith("\n") else content + "\n")
    elif mode == "replace":
        lines[idx] = content if content.endswith("\n") else content + "\n"

    try:
        target.write_text("".join(lines), encoding="utf-8")
    except OSError as e:
        return False, f"Error writing {target}: {e}"

    turn_read_files.discard(target)
    return True, f"Edited: {target}. Re-read required before next edit."


def execute_delete_file(arguments: dict, config: Config, turn_read_files: set[Path]) -> tuple[bool, str]:
    raw_path = arguments["path"]
    reason = arguments.get("description", "")

    target = Path(raw_path).resolve()

    if not _is_allowed(target, config.write_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
        return False, f"Error: path {target} is outside allowed directories: {dirs_str}"

    if not target.exists():
        return False, f"Error: file not found: {target}"

    if target not in turn_read_files:
        return False, f"Error: {target} has not been read this turn. Call read_file first."

    if not config.auto_approve_writes:
        print(f"[delete_file] {target}")
        print(f"Reason: {reason}")
        print("Approve? [y/N] ", end="", flush=True)
        answer = input()
        if answer.strip().lower() != "y":
            return False, f"Delete denied by user. Path: {target}"

    try:
        target.unlink()
    except OSError as e:
        return False, f"Error deleting {target}: {e}"

    turn_read_files.discard(target)
    return True, f"Deleted: {target}"


def execute_move_file(arguments: dict, config: Config, turn_read_files: set[Path]) -> tuple[bool, str]:
    src = Path(arguments["src"]).resolve()
    dst = Path(arguments["dst"]).resolve()
    reason = arguments.get("description", "")

    if not _is_allowed(src, config.write_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
        return False, f"Error: src {src} is outside allowed directories: {dirs_str}"

    if not _is_allowed(dst, config.write_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
        return False, f"Error: dst {dst} is outside allowed directories: {dirs_str}"

    if not src.exists():
        return False, f"Error: source file not found: {src}"

    if src not in turn_read_files:
        return False, f"Error: {src} has not been read this turn. Call read_file first."

    if not config.auto_approve_writes:
        print(f"[move_file] {src} → {dst}")
        print(f"Reason: {reason}")
        print("Approve? [y/N] ", end="", flush=True)
        answer = input()
        if answer.strip().lower() != "y":
            return False, f"Move denied by user. src: {src}"

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    except OSError as e:
        return False, f"Error moving {src} → {dst}: {e}"

    turn_read_files.discard(src)
    return True, f"Moved: {src} → {dst}"


def execute_read_file(arguments: dict, config: Config, turn_read_files: set[Path]) -> str:
    paths = arguments["paths"]
    sections: list[str] = []
    for raw in paths:
        target = Path(raw).resolve()
        if not _is_allowed(target, config.read_allowed_dirs):
            dirs_str = ", ".join(str(d) for d in config.read_allowed_dirs)
            content = f"Error: path {target} is outside allowed directories: {dirs_str}"
        else:
            try:
                content = target.read_text(encoding="utf-8")
                turn_read_files.add(target)
            except FileNotFoundError:
                content = f"Error: file not found: {target}"
            except OSError as e:
                content = f"Error reading {target}: {e}"
        sections.append(f"=== {target} ===\n{content}")
    return "\n".join(sections)


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


def execute_find_files(arguments: dict, config: Config) -> str:
    target = Path(arguments["path"]).resolve()
    pattern = arguments["pattern"]

    if not _is_allowed(target, config.read_allowed_dirs):
        dirs_str = ", ".join(str(d) for d in config.read_allowed_dirs)
        return f"Error: path {target} is outside allowed directories: {dirs_str}"

    if not target.exists():
        return f"Error: path not found: {target}"
    if not target.is_dir():
        return f"Error: not a directory: {target}"

    matches = sorted(target.rglob(pattern))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return "No matches found."
    return "\n".join(str(p) for p in matches)


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

def _print_unified_diff(old: str, new: str, path: Path) -> None:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=str(path), tofile=str(path))
    print("".join(diff), end="")


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
    if hasattr(node, "decorator_list") and node.decorator_list:
        start = node.decorator_list[0].lineno - 1
    return "".join(lines[start:end])
