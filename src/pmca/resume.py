from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pmca.types import Attachment


class ResumeError(Exception):
    pass


@dataclass
class ResumedSession:
    system_prompt: str
    startup_docs: list[tuple[Path, str]]
    history: list[dict]
    session_attachments: list[Attachment]
    last_assistant_message: str
    jsonl_path: Path
    next_attachment_n: int


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

    system_prompt_entries = [e for e in entries if e.get("type") == "system_prompt"]
    if not system_prompt_entries:
        raise ResumeError(f"No system_prompt entry found in {path}")

    system_prompt = system_prompt_entries[0]["content"]

    startup_docs = [
        (Path(e["path"]), e["content"])
        for e in entries
        if e.get("type") == "startup_doc"
    ]

    exchange_entries = [e for e in entries if e.get("type") == "exchange"]

    history = [
        {"role": e["role"], "content": e["content"]}
        for e in exchange_entries
        if e.get("role") in ("user", "assistant")
    ]

    if not history:
        raise ResumeError(f"No valid user/assistant turns found in {path}")

    last_assistant = next(
        (e["content"] for e in reversed(exchange_entries) if e.get("role") == "assistant"),
        "",
    )

    session_attachments = _collect_attachments(exchange_entries)
    next_attachment_n = _compute_next_attachment_n(exchange_entries)

    return ResumedSession(
        system_prompt=system_prompt,
        startup_docs=startup_docs,
        history=history,
        session_attachments=session_attachments,
        last_assistant_message=last_assistant,
        jsonl_path=path,
        next_attachment_n=next_attachment_n,
    )


def _collect_attachments(exchange_entries: list[dict]) -> list[Attachment]:
    seen: set[str] = set()
    result: list[Attachment] = []
    for entry in exchange_entries:
        if entry.get("role") != "user":
            continue
        for att in entry.get("attachments", []):
            ident = att.get("identifier", "")
            if ident in seen:
                continue
            seen.add(ident)
            result.append(Attachment(
                path=Path(att["path"]),
                content=att["content"],
                identifier=ident,
                size_warning=att.get("size_warning", False),
            ))
    return result


def _compute_next_attachment_n(exchange_entries: list[dict]) -> int:
    max_n = 0
    for entry in exchange_entries:
        for att in entry.get("attachments", []):
            ident = att.get("identifier", "")
            if ident.startswith("CONTEXT_"):
                try:
                    n = int(ident[len("CONTEXT_"):])
                    max_n = max(max_n, n)
                except ValueError:
                    pass
    return max_n + 1
