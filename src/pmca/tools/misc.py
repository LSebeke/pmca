from __future__ import annotations

import subprocess
from pathlib import Path

from pmca.config import Config
from pmca.rag.store import VectorStore
from pmca.types import Chunk, ScratchpadEntry


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
