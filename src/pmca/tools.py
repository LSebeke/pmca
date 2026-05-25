from __future__ import annotations

from pathlib import Path

from pmca.config import Config

_WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "",  # filled in by get_tools()
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Full content to write to the file (UTF-8).",
                },
                "description": {
                    "type": "string",
                    "description": "Short human-readable explanation of what is being written and why.",
                },
            },
            "required": ["path", "content", "description"],
            "additionalProperties": False,
        },
    },
}


def get_tools(config: Config) -> list[dict] | None:
    if not config.write_allowed_dirs:
        return None
    dirs_str = ", ".join(str(d) for d in config.write_allowed_dirs)
    schema = {
        **_WRITE_FILE_SCHEMA,
        "function": {
            **_WRITE_FILE_SCHEMA["function"],
            "description": (
                f"Write a file to disk. "
                f"Allowed directories: {dirs_str}"
            ),
        },
    }
    return [schema]


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


def _is_allowed(target: Path, allowed_dirs: list[Path]) -> bool:
    for d in allowed_dirs:
        try:
            target.relative_to(d)
            return True
        except ValueError:
            continue
    return False
