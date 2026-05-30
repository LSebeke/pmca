from __future__ import annotations

from pathlib import Path

from pmca.config import Config
from pmca.rag.store import VectorStore
from pmca.types import Chunk


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
