from __future__ import annotations

from pmca.config import Config
from pmca.rag.store import VectorStore

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
