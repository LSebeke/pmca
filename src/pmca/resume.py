from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class ResumeError(Exception):
    pass


@dataclass
class ResumedSession:
    history: list[dict]
    resumed_context: str
    last_assistant_message: str
    jsonl_path: Path


def load_resume(path: Path) -> ResumedSession:
    if not path.exists():
        raise ResumeError(f"Resume file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    bad_lines: list[int] = []
    entries: list[dict] = []

    for i, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines.append(i)

    if bad_lines:
        nums = ", ".join(f"line {n}" for n in bad_lines)
        raise ResumeError(f"Malformed JSON in {path}: {nums}")

    history = [
        {"role": e["role"], "content": e["content"]}
        for e in entries
        if e.get("role") in ("user", "assistant")
    ]

    if not history:
        raise ResumeError(f"No valid user/assistant turns found in {path}")

    last_assistant = next(
        (e["content"] for e in reversed(entries) if e.get("role") == "assistant"),
        "",
    )

    resumed_context = _build_resumed_context(entries)

    return ResumedSession(
        history=history,
        resumed_context=resumed_context,
        last_assistant_message=last_assistant,
        jsonl_path=path,
    )


def _build_resumed_context(entries: list[dict]) -> str:
    seen_identifiers: set[str] = set()
    att_blocks: list[str] = []
    rag_blocks: list[str] = []

    for entry in entries:
        if entry.get("role") != "user":
            continue

        for att in entry.get("attachments", []):
            ident = att.get("identifier", "")
            if ident in seen_identifiers:
                continue
            seen_identifiers.add(ident)
            suffix = Path(att.get("path", "")).suffix.lstrip(".")
            att_blocks.append(
                f"[{ident}]\nFile: {att.get('path', '')}\nType: {suffix}\n---\n{att.get('content', '')}\n---"
            )

        for chunk in entry.get("rag_chunks", []):
            rag_blocks.append(
                f"[RAG]\nFile: {chunk.get('source', '')}\nChunk: {chunk.get('label', '')}\n---\n{chunk.get('content', '')}\n---"
            )

    if not att_blocks and not rag_blocks:
        return ""

    parts = ["[RESUMED_CONTEXT]", "The following attachments and RAG chunks were present in the resumed session."]
    parts.extend(att_blocks)
    parts.extend(rag_blocks)
    return "\n\n".join(parts)
