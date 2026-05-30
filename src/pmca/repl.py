from __future__ import annotations

import re
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

from pmca.chat import ChatSession, _format_scratchpad_entry

_HELP = """\
Commands:
  /set history_token_budget=N      Set history token budget for this session
  /set test_timeout=N              Set test run timeout in seconds for this session
  /set auto_approve_writes=true|false  Skip write approval prompts for this session
  /set show_diff_on_approve=true|false Show unified diff before approval prompt
  /read add <path>            Add a directory to read_allowed_dirs for this session
  /read remove <path>         Remove a directory from read_allowed_dirs for this session
  /extract <path>             Extract code blocks from last response into <path> (type inferred from extension)
  /scratchpad                 Show all scratchpad entries
  /skill                      List available skills (* = active)
  /skill <name>               Activate a skill
  /skill remove <name>        Deactivate a skill
  /clear                      Clear conversation history
  /help                       Print this help message
  /exit                       End session

Key bindings:
  Up arrow  Recall previous input for editing
  Esc       Clear current input
"""

_SETTABLE = {
    "history_token_budget": "history_token_budget",
}

_CONFIG_BOOL_SETTABLE = {"auto_approve_writes", "show_diff_on_approve"}


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

    if name == "/set":
        _handle_set(parts[1] if len(parts) > 1 else "", session)
        return

    if name == "/extract":
        _extract(parts[1] if len(parts) > 1 else "", session)
        return

    if name == "/clear":
        session.history = []
        new_path = session.rotate_logger()
        print(f"Conversation history cleared. New session: {new_path}")
        return

    if name == "/read":
        _handle_read(parts[1] if len(parts) > 1 else "", session)
        return

    if name == "/scratchpad":
        entries = session._scratchpad
        if not entries:
            print("Scratchpad is empty.")
        else:
            for i, entry in enumerate(entries, start=1):
                print(_format_scratchpad_entry(i, entry))
        return

    if name == "/skill":
        _handle_skill(parts[1] if len(parts) > 1 else "", session)
        return

    print(f"Unknown command: {name}")


def _handle_set(arg: str, session: ChatSession) -> None:
    if "=" not in arg:
        print(f"Error: expected /set <param>=<value>")
        return

    key, _, raw_value = arg.partition("=")
    key = key.strip()
    attr = _SETTABLE.get(key)

    _CONFIG_SETTABLE = {"test_timeout"}

    if key in _CONFIG_BOOL_SETTABLE:
        normalized = raw_value.strip().lower()
        if normalized not in ("true", "false"):
            print(f"Error: value for '{key}' must be true or false, got: {raw_value.strip()!r}")
            return
        value = normalized == "true"
        setattr(session.config, key, value)
        print(f"{key} = {value}")
        return

    if attr is None and key not in _CONFIG_SETTABLE:
        valid = ", ".join(list(_SETTABLE) + sorted(_CONFIG_BOOL_SETTABLE) + sorted(_CONFIG_SETTABLE))
        print(f"Error: unknown parameter '{key}'. Valid: {valid}")
        return

    try:
        value = int(raw_value.strip())
    except ValueError:
        print(f"Error: value for '{key}' must be an integer, got: {raw_value.strip()!r}")
        return

    if value <= 0:
        print(f"Error: '{key}' must be a positive integer, got {value}")
        return

    if key in _CONFIG_SETTABLE:
        setattr(session.config, key, value)
    else:
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


def _handle_skill(arg: str, session) -> None:
    from pmca.types import ActiveSkill

    skills_dir = session.config.skills_dir
    if skills_dir is None:
        print("Error: skills_dir not configured.")
        return

    arg = arg.strip()

    if arg.startswith("remove "):
        name = arg[len("remove "):].strip()
        for i, skill in enumerate(session._active_skills):
            if skill.name == name:
                session._active_skills.pop(i)
                session.config.read_allowed_dirs.remove(skill.directory)
                print(f"Skill '{name}' deactivated.")
                return
        print(f"Skill '{name}' is not active.")
        return

    if not arg:
        skill_dirs = sorted(d for d in skills_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists())
        if not skill_dirs:
            print("No skills available.")
            return
        active_names = {s.name for s in session._active_skills}
        print("Available skills (* = active):")
        for d in skill_dirs:
            marker = "*" if d.name in active_names else " "
            print(f"  {marker} {d.name}")
        return

    # Activate by name
    name = arg
    if any(s.name == name for s in session._active_skills):
        print(f"Skill '{name}' is already active.")
        return
    skill_dir = skills_dir / name
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        print(f"Error: skill '{name}' not found in {skills_dir}.")
        return
    content = skill_md.read_text(encoding="utf-8")
    skill = ActiveSkill(name=name, content=content, directory=skill_dir)
    session._active_skills.append(skill)
    session.config.read_allowed_dirs.append(skill_dir)
    print(f"Skill '{name}' activated.")


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
