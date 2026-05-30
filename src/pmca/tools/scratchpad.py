from __future__ import annotations

from pmca.config import Config
from pmca.types import ScratchpadEntry


def execute_save_to_scratchpad(
    arguments: dict,
    config: Config,
    scratchpad: list[ScratchpadEntry],
) -> str:
    delete_titles = set(arguments.get("delete", []))
    upsert_entries = arguments.get("entries", [])

    deleted_count = 0
    if delete_titles:
        before = len(scratchpad)
        scratchpad[:] = [e for e in scratchpad if e.title not in delete_titles]
        deleted_count = before - len(scratchpad)

    existing_titles = {e.title for e in scratchpad}
    overwrites = [e for e in upsert_entries if e["title"] in existing_titles]
    new_additions = [e for e in upsert_entries if e["title"] not in existing_titles]

    if len(scratchpad) + len(new_additions) > config.max_scratchpad_entries:
        free = config.max_scratchpad_entries - len(scratchpad)
        return (
            f"Error: cap is {config.max_scratchpad_entries}; "
            f"{len(scratchpad)} slot(s) used, {free} free — "
            f"cannot add {len(new_additions)} new entry/entries. Delete some first."
        )

    overwrite_map = {e["title"]: e["content"] for e in overwrites}
    for entry in scratchpad:
        if entry.title in overwrite_map:
            entry.content = overwrite_map[entry.title]

    for e in new_additions:
        scratchpad.append(ScratchpadEntry(title=e["title"], content=e["content"]))

    parts = []
    if deleted_count:
        parts.append(f"Deleted {_pluralize(deleted_count, 'entry')}.")
    if overwrites:
        parts.append(f"Updated {_pluralize(len(overwrites), 'entry')}.")
    if new_additions:
        parts.append(f"Saved {_pluralize(len(new_additions), 'new entry')}.")
    parts.append(f"[Scratchpad: {len(scratchpad)} entries]")
    return " ".join(parts)


def _pluralize(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")
