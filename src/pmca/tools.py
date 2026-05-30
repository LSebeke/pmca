from __future__ import annotations

import ast
import difflib
import re
import subprocess
from pathlib import Path

import git as gitlib

from pmca.config import Config
from pmca.rag.store import VectorStore
from pmca.types import Chunk, ScratchpadEntry

class SafeGitOps:
    def __init__(self, repo_path: Path, read_allowed_dirs: list[Path]) -> None:
        self._repo = gitlib.Repo(repo_path)
        self._root = Path(self._repo.working_dir)
        self._read_allowed_dirs = read_allowed_dirs

    def _validate_path(self, path: str) -> Path | str:
        resolved = (self._root / path).resolve()
        if not _is_allowed(resolved, self._read_allowed_dirs):
            dirs_str = ", ".join(str(d) for d in self._read_allowed_dirs)
            return f"Error: path {resolved} is outside allowed directories: {dirs_str}"
        return resolved

    def _resolve_ref(self, ref: str):
        try:
            return self._repo.commit(ref)
        except (gitlib.BadName, gitlib.BadObject, ValueError):
            return None

    def status(self) -> dict:
        return {
            "dirty": self._repo.is_dirty(),
            "untracked": self._repo.untracked_files,
            "staged": [d.a_path for d in self._repo.index.diff("HEAD")],
            "unstaged": [d.a_path for d in self._repo.index.diff(None)],
        }

    def log(self, max_count: int = 20) -> list[dict]:
        commits = list(self._repo.iter_commits(max_count=max_count))
        return [
            {
                "sha": c.hexsha[:8],
                "message": c.message.strip(),
                "author": str(c.author),
                "date": c.committed_datetime.isoformat(),
            }
            for c in commits
        ]

    def diff(self, ref: str = "HEAD", path: str | None = None, staged: bool = False) -> str:
        commit = self._resolve_ref(ref)
        if commit is None:
            return f"Error: invalid ref '{ref}'"

        if path is not None:
            result = self._validate_path(path)
            if isinstance(result, str):
                return result
            path_posix = Path(path).as_posix()
        else:
            path_posix = None

        try:
            if staged:
                diff_args = [commit.hexsha]
                if path_posix:
                    diff_args += ["--", path_posix]
                return self._repo.git.diff("--cached", *diff_args)
            else:
                diff_args = [commit.hexsha]
                if path_posix:
                    diff_args += ["--", path_posix]
                return self._repo.git.diff(*diff_args)
        except gitlib.GitCommandError as e:
            return f"Error: {e}"

    def blame(self, ref: str = "HEAD", path: str = "") -> str:
        result = self._validate_path(path)
        if isinstance(result, str):
            return result

        commit = self._resolve_ref(ref)
        if commit is None:
            return f"Error: invalid ref '{ref}'"

        try:
            blame = self._repo.blame(commit.hexsha, path)
        except gitlib.GitCommandError as e:
            return f"Error: {e}"

        lines = []
        for entry_commit, entry_lines in blame:
            sha = entry_commit.hexsha[:8]
            author = str(entry_commit.author)
            for line in entry_lines:
                text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
                lines.append(f"{sha} {author}: {text}")
        return "".join(lines)

    def show_file(self, ref: str, path: str) -> str:
        result = self._validate_path(path)
        if isinstance(result, str):
            return result

        commit = self._resolve_ref(ref)
        if commit is None:
            return f"Error: invalid ref '{ref}'"

        try:
            blob = commit.tree / Path(path).as_posix()
            return blob.data_stream.read().decode("utf-8", errors="replace")
        except (KeyError, gitlib.GitCommandError) as e:
            return f"Error: {e}"

    def branches(self) -> list[str]:
        return [b.name for b in self._repo.branches]

    def current_branch(self) -> str:
        try:
            return self._repo.active_branch.name
        except TypeError:
            return "(detached HEAD)"


_GIT_STATUS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "git_status",
        "description": "",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}

_GIT_LOG_SCHEMA = {
    "type": "function",
    "function": {
        "name": "git_log",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "max_count": {"type": "integer", "description": "Maximum number of commits to return (default 20)."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

_GIT_DIFF_SCHEMA = {
    "type": "function",
    "function": {
        "name": "git_diff",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Commit/branch to diff against (default HEAD)."},
                "path": {"type": "string", "description": "Optional path to scope the diff to a specific file."},
                "staged": {"type": "boolean", "description": "If true, show staged (index) diff; otherwise show working-tree diff."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

_GIT_BLAME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "git_blame",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative path of the file to blame."},
                "ref": {"type": "string", "description": "Commit/branch to blame against (default HEAD)."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

_GIT_SHOW_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "git_show_file",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Commit/branch/tag to retrieve the file from."},
                "path": {"type": "string", "description": "Repo-relative path of the file."},
            },
            "required": ["ref", "path"],
            "additionalProperties": False,
        },
    },
}

_GIT_BRANCHES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "git_branches",
        "description": "",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}

_GIT_CURRENT_BRANCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "git_current_branch",
        "description": "",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
}

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

_EDIT_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the file to edit."},
                "old_string": {"type": "string", "description": "Exact text to find and replace. Must appear exactly once in the file."},
                "new_string": {"type": "string", "description": "Text to replace old_string with."},
                "description": {"type": "string", "description": "Short human-readable explanation of what is being changed and why."},
            },
            "required": ["path", "old_string", "new_string", "description"],
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
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Absolute paths of the files to read.",
                },
            },
            "required": ["paths"],
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

_RUN_TESTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "filter": {"type": "string", "description": "Optional pytest filter: a test file path, a -k expression, or both (e.g. 'tests/test_foo.py -k bar')."},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

_FIND_FILES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_files",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the directory to search."},
                "pattern": {"type": "string", "description": "Glob pattern matched against filenames, e.g. '*.py' or 'test_*.py'."},
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


_RAG_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_knowledge_base",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query to find relevant code or documentation chunks."},
                "depth": {
                    "type": "string",
                    "enum": ["shallow", "medium", "deep"],
                    "description": "How many results to retrieve: shallow (few), medium, or deep (many). Subsequent calls with the same query return only new results not already retrieved.",
                },
            },
            "required": ["query", "depth"],
            "additionalProperties": False,
        },
    },
}


_INSERT_AT_LINE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "insert_at_line",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the file to edit."},
                "line_number": {"type": "integer", "description": "1-indexed line number to target."},
                "content": {"type": "string", "description": "Content to insert or use as replacement."},
                "mode": {"type": "string", "enum": ["before", "after", "replace"], "description": "before: insert before the line; after: insert after; replace: substitute the line."},
                "description": {"type": "string", "description": "Short human-readable explanation of what is being changed and why."},
            },
            "required": ["path", "line_number", "content", "mode", "description"],
            "additionalProperties": False,
        },
    },
}

_DELETE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the file to delete."},
                "description": {"type": "string", "description": "Short human-readable explanation of what is being deleted and why."},
            },
            "required": ["path", "description"],
            "additionalProperties": False,
        },
    },
}

_MOVE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "move_file",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "Absolute path of the source file."},
                "dst": {"type": "string", "description": "Absolute path of the destination."},
                "description": {"type": "string", "description": "Short human-readable explanation of the move and why."},
            },
            "required": ["src", "dst", "description"],
            "additionalProperties": False,
        },
    },
}

_SAVE_TO_SCRATCHPAD_SCHEMA = {
    "type": "function",
    "function": {
        "name": "save_to_scratchpad",
        "description": (
            "Save excerpts from tool call returns to the scratchpad so they persist across turns. "
            "Only save information that would otherwise be lost (tool call returns are not stored in history). "
            "Each entry must have a title that makes its origin clear "
            "(e.g. 'read_file: src/pmca/config.py — load_config body'). "
            "Use 'entries' to upsert (add or overwrite by title) and 'delete' to remove entries by title."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Short label making the origin clear."},
                            "content": {"type": "string", "description": "Excerpt to save."},
                        },
                        "required": ["title", "content"],
                        "additionalProperties": False,
                    },
                    "description": "Entries to upsert (add new or overwrite existing by title).",
                    "default": [],
                },
                "delete": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Titles of entries to delete. Unknown titles are silently ignored.",
                    "default": [],
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def get_tools(config: Config, store: VectorStore) -> list[dict] | None:
    tools = []

    if store._chunks:
        tools.append({
            **_RAG_SCHEMA,
            "function": {
                **_RAG_SCHEMA["function"],
                "description": "Search the project knowledge base for relevant code or documentation. Use depth='shallow' first; call again with 'medium' or 'deep' to retrieve additional results.",
            },
        })

    if config.write_allowed_dirs:
        dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
        tools.append({
            **_WRITE_FILE_SCHEMA,
            "function": {
                **_WRITE_FILE_SCHEMA["function"],
                "description": f"Write a file to disk. Allowed directories: {dirs_str}",
            },
        })
        tools.append({
            **_EDIT_FILE_SCHEMA,
            "function": {
                **_EDIT_FILE_SCHEMA["function"],
                "description": f"Edit a file by replacing an exact string. old_string must appear exactly once. Allowed directories: {dirs_str}",
            },
        })
        tools.append({
            **_INSERT_AT_LINE_SCHEMA,
            "function": {
                **_INSERT_AT_LINE_SCHEMA["function"],
                "description": f"Insert content before/after a line or replace a line by number. Allowed directories: {dirs_str}",
            },
        })
        tools.append({
            **_DELETE_FILE_SCHEMA,
            "function": {
                **_DELETE_FILE_SCHEMA["function"],
                "description": f"Delete a file. Requires prior read_file this turn. Allowed directories: {dirs_str}",
            },
        })
        tools.append({
            **_MOVE_FILE_SCHEMA,
            "function": {
                **_MOVE_FILE_SCHEMA["function"],
                "description": f"Move/rename a file. src must have been read this turn. Allowed directories: {dirs_str}",
            },
        })

    if config.read_allowed_dirs:
        dirs_str = ", ".join(str(d) for d in config.read_allowed_dirs)
        desc_suffix = f" Allowed directories: {dirs_str}"

        for base_schema, desc in [
            (_READ_FILE_SCHEMA, "Read a file from disk." + desc_suffix),
            (_LIST_DIR_SCHEMA, "List directory contents." + desc_suffix),
            (_SEARCH_SCHEMA, "Search for a regex pattern in a file or directory tree." + desc_suffix),
            (_GET_DEFINITION_SCHEMA, "Get the full source of a Python function or class." + desc_suffix),
            (_FIND_FILES_SCHEMA, "Find files matching a glob pattern." + desc_suffix),
        ]:
            tools.append({
                **base_schema,
                "function": {**base_schema["function"], "description": desc},
            })

    if config.git_root is not None:
        for base_schema, desc in [
            (_GIT_STATUS_SCHEMA, "Show git working tree status (dirty, staged, unstaged, untracked)."),
            (_GIT_LOG_SCHEMA, "Show recent git commit history."),
            (_GIT_DIFF_SCHEMA, "Show git diff against a ref. Optional path filter and staged flag."),
            (_GIT_BLAME_SCHEMA, "Show per-line git blame for a file."),
            (_GIT_SHOW_FILE_SCHEMA, "Show file content at a specific git ref."),
            (_GIT_BRANCHES_SCHEMA, "List all local git branches."),
            (_GIT_CURRENT_BRANCH_SCHEMA, "Show the current git branch."),
        ]:
            tools.append({**base_schema, "function": {**base_schema["function"], "description": desc}})

    if config.test_dir is not None:
        tools.append({
            **_RUN_TESTS_SCHEMA,
            "function": {
                **_RUN_TESTS_SCHEMA["function"],
                "description": f"Run the test suite in {config.test_dir}. Pass an optional filter (file path or -k expression).",
            },
        })

    if tools:
        tools.append(_SAVE_TO_SCRATCHPAD_SCHEMA)

    return tools if tools else None


def execute_rag_query(
    arguments: dict,
    config: Config,
    store: VectorStore,
    turn_seen: set[tuple[Path, str]],
) -> str:
    query = arguments["query"]
    depth = arguments.get("depth", "shallow")
    k = {"shallow": config.rag_shallow_k, "medium": config.rag_medium_k, "deep": config.rag_deep_k}.get(depth, config.rag_shallow_k)

    candidates = store.query(query, k)
    new_chunks = [c for c in candidates if (c.source_file, c.label) not in turn_seen]

    if not new_chunks:
        return "No results found."

    for c in new_chunks:
        turn_seen.add((c.source_file, c.label))

    return _format_rag_chunks(new_chunks)


def _format_rag_chunks(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[RAG_{i}]\nFile: {chunk.source_file}\nChunk: {chunk.label}\n---\n{chunk.content}\n---"
        )
    return "\n\n".join(parts)


def execute_save_to_scratchpad(
    arguments: dict,
    config: Config,
    scratchpad: list[ScratchpadEntry],
) -> str:
    delete_titles = set(arguments.get("delete", []))
    upsert_entries = arguments.get("entries", [])

    # 1. Deletes first
    deleted_count = 0
    if delete_titles:
        before = len(scratchpad)
        scratchpad[:] = [e for e in scratchpad if e.title not in delete_titles]
        deleted_count = before - len(scratchpad)

    # 2. Split upserts into overwrites vs new additions
    existing_titles = {e.title for e in scratchpad}
    overwrites = [e for e in upsert_entries if e["title"] in existing_titles]
    new_additions = [e for e in upsert_entries if e["title"] not in existing_titles]

    # 3. Cap check on new additions only
    if len(scratchpad) + len(new_additions) > config.max_scratchpad_entries:
        free = config.max_scratchpad_entries - len(scratchpad)
        return (
            f"Error: cap is {config.max_scratchpad_entries}; "
            f"{len(scratchpad)} slot(s) used, {free} free — "
            f"cannot add {len(new_additions)} new entry/entries. Delete some first."
        )

    # 4. Apply overwrites in-place
    overwrite_map = {e["title"]: e["content"] for e in overwrites}
    for entry in scratchpad:
        if entry.title in overwrite_map:
            entry.content = overwrite_map[entry.title]

    # 5. Append new additions
    for e in new_additions:
        scratchpad.append(ScratchpadEntry(title=e["title"], content=e["content"]))

    # 6. Summary
    def _n(count: int, noun: str) -> str:
        return f"{count} {noun}" + ("" if count == 1 else "s")

    parts = []
    if deleted_count:
        parts.append(f"Deleted {_n(deleted_count, 'entry')}.")
    if overwrites:
        parts.append(f"Updated {_n(len(overwrites), 'entry')}.")
    if new_additions:
        parts.append(f"Saved {_n(len(new_additions), 'new entry')}.")
    parts.append(f"[Scratchpad: {len(scratchpad)} entries]")
    return " ".join(parts)


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
        if config.show_diff_on_approve and target.exists():
            old = target.read_text(encoding="utf-8")
            _print_unified_diff(old, content, target)
        print(f"{exists_msg} Approve? [y/N] ", end="", flush=True)
        answer = input()
        if answer.strip().lower() != "y":
            return False, f"Write denied by user. Path: {target}"

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
        if config.show_diff_on_approve:
            _print_unified_diff(content, new_content, target)
        print("Approve? [y/N] ", end="", flush=True)
        answer = input()
        if answer.strip().lower() != "y":
            return False, f"Edit denied by user. Path: {target}"
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
    target_line = lines[idx].rstrip("\n")

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
        if config.show_diff_on_approve:
            _print_unified_diff("".join(lines), "".join(new_lines), target)
        print("Approve? [y/N] ", end="", flush=True)
        answer = input()
        if answer.strip().lower() != "y":
            return False, f"Edit denied by user. Path: {target}"

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
                # looking for a method inside this class
                if isinstance(node, ast.ClassDef):
                    for child in ast.walk(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name == parts[1]:
                                return _extract_node_source(child, lines)
                    return f"Error: symbol '{symbol}' not found in {target}"

    return f"Error: symbol '{parts[0]}' not found in {target}"


def execute_git_status(config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    status = ops.status()
    lines = [f"dirty: {status['dirty']}"]
    if status["staged"]:
        lines.append("staged: " + ", ".join(status["staged"]))
    if status["unstaged"]:
        lines.append("unstaged: " + ", ".join(status["unstaged"]))
    if status["untracked"]:
        lines.append("untracked: " + ", ".join(status["untracked"]))
    return "\n".join(lines)


def execute_git_log(arguments: dict, config: Config) -> str:
    max_count = int(arguments.get("max_count", 20))
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    commits = ops.log(max_count=max_count)
    lines = [f"{c['sha']} {c['date'][:10]} {c['author']}: {c['message']}" for c in commits]
    return "\n".join(lines) if lines else "No commits found."


def execute_git_diff(arguments: dict, config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return ops.diff(
        ref=arguments.get("ref", "HEAD"),
        path=arguments.get("path"),
        staged=bool(arguments.get("staged", False)),
    )


def execute_git_blame(arguments: dict, config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return ops.blame(ref=arguments.get("ref", "HEAD"), path=arguments["path"])


def execute_git_show_file(arguments: dict, config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return ops.show_file(ref=arguments["ref"], path=arguments["path"])


def execute_git_branches(config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return "\n".join(ops.branches())


def execute_git_current_branch(config: Config) -> str:
    ops = SafeGitOps(config.git_root, config.read_allowed_dirs)
    return ops.current_branch()


def execute_run_tests(arguments: dict, config: Config) -> tuple[bool, str]:
    test_dir = config.test_dir
    use_pixi = (test_dir / "pixi.toml").exists()
    cmd = ["pixi", "run", "pytest"] if use_pixi else ["pytest"]

    raw_filter = arguments.get("filter", "").strip()
    if raw_filter:
        cmd.extend(raw_filter.split())

    print(f"[run_tests] {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=test_dir,
            capture_output=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=config.test_timeout,
        )
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, f"Error: run_tests timed out after {config.test_timeout} seconds"
    except OSError as e:
        return False, f"Error: {e}"


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
    # include decorators
    if hasattr(node, "decorator_list") and node.decorator_list:
        start = node.decorator_list[0].lineno - 1
    return "".join(lines[start:end])
