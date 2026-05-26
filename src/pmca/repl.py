from __future__ import annotations

import re
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from pmca.chat import ChatSession

_HELP = """\
Commands:
  /set chunksize=N            Set top-k RAG retrieval count for this session
  /set history_token_budget=N Set history token budget for this session
  /read add <path>            Add a directory to read_allowed_dirs for this session
  /read remove <path>         Remove a directory from read_allowed_dirs for this session
  /rag                        Print RAG chunks retrieved for the last query
  /extract <path>             Extract code blocks from last response into <path> (type inferred from extension)
  /clear                      Clear conversation history
  /help                       Print this help message
  /exit                       End session

Key bindings:
  Up arrow  Recall previous input for editing
  Esc       Clear current input
"""

_SETTABLE = {
    "chunksize": "top_k",
    "history_token_budget": "history_token_budget",
}


def run_repl(session: ChatSession) -> None:
    bindings = KeyBindings()

    @bindings.add("escape")
    def _clear(event):
        event.app.current_buffer.reset()

    prompt = PromptSession(history=InMemoryHistory(), key_bindings=bindings)

    while True:
        try:
            user_input = prompt.prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            try:
                handle_command(user_input, session)
            except SystemExit:
                break
        else:
            response, turns_dropped = session.process(user_input)
            if response is not None:
                print(response)
                if turns_dropped > 0:
                    print(_trim_notice(turns_dropped))


def handle_command(cmd: str, session: ChatSession) -> None:
    parts = cmd.strip().split(None, 1)
    name = parts[0].lower()

    if name == "/exit":
        raise SystemExit(0)

    if name == "/help":
        print(_HELP, end="")
        return

    if name == "/rag":
        if not session._last_rag_chunks:
            print("No RAG data yet — send a message first.")
        else:
            for i, chunk in enumerate(session._last_rag_chunks, start=1):
                print(f"[RAG_{i}] {chunk.source_file}  {chunk.label}")
                print(chunk.content)
                print()
        return

    if name == "/set":
        _handle_set(parts[1] if len(parts) > 1 else "", session)
        return

    if name == "/extract":
        _extract(parts[1] if len(parts) > 1 else "", session)
        return

    if name == "/clear":
        session.history = []
        session._last_rag_chunks = []
        new_path = session.rotate_logger()
        print(f"Conversation history cleared. New session: {new_path}")
        return

    if name == "/read":
        _handle_read(parts[1] if len(parts) > 1 else "", session)
        return

    print(f"Unknown command: {name}")


def _handle_set(arg: str, session: ChatSession) -> None:
    if "=" not in arg:
        print(f"Error: expected /set <param>=<value>")
        return

    key, _, raw_value = arg.partition("=")
    key = key.strip()
    attr = _SETTABLE.get(key)

    if attr is None:
        print(f"Error: unknown parameter '{key}'. Valid: {', '.join(_SETTABLE)}")
        return

    try:
        value = int(raw_value.strip())
    except ValueError:
        print(f"Error: value for '{key}' must be an integer, got: {raw_value.strip()!r}")
        return

    if value <= 0:
        print(f"Error: '{key}' must be a positive integer, got {value}")
        return

    setattr(session, attr, value)
    print(f"{key} = {value}")


_EXT_TO_FENCE: dict[str, str] = {
    ".py": "python",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sh": "bash",
    ".md": "markdown",
}


def _extract(arg: str, session: ChatSession) -> None:
    arg = arg.strip()
    if not arg:
        print("Error: usage: /extract <absolute-path>")
        return

    path = Path(arg)
    fence = _EXT_TO_FENCE.get(path.suffix)
    if fence is None:
        supported = ", ".join(sorted(_EXT_TO_FENCE))
        print(f"Error: unsupported extension '{path.suffix}'. Supported: {supported}")
        return

    if not session.history:
        print("Error: no assistant response yet.")
        return

    last = session.history[-1]["content"]
    blocks = re.findall(rf"```{fence}\n(.*?)```", last, re.DOTALL)

    if not blocks:
        print(f"No {fence} code blocks found in last response.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(b.rstrip("\n") for b in blocks))
    print(f"Wrote {len(blocks)} block(s) to {arg}")


def _handle_read(arg: str, session) -> None:
    parts = arg.strip().split(None, 1)
    if len(parts) < 2 or parts[0] not in ("add", "remove"):
        print("Error: usage: /read add <path> | /read remove <path>")
        return

    subcommand = parts[0]
    target = Path(parts[1].strip()).resolve()

    if subcommand == "add":
        print(f"Add {target} to read_allowed_dirs? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Cancelled.")
            return
        session.config.read_allowed_dirs.append(target)
        print(f"Added: {target}")

    elif subcommand == "remove":
        if target not in session.config.read_allowed_dirs:
            print(f"Not in read_allowed_dirs: {target}")
            return
        print(f"Remove {target} from read_allowed_dirs? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Cancelled.")
            return
        session.config.read_allowed_dirs.remove(target)
        print(f"Removed: {target}")


def _trim_notice(n: int) -> str:
    return f"[{n} earlier turn(s) omitted from context]"
