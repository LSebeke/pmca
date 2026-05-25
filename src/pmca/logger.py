from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from pmca.types import Attachment, Chunk


class SessionLogger:
    def __init__(self, log_folder: Path, timestamp: str) -> None:
        self._jsonl: IO[str] = open(log_folder / f"chat_{timestamp}.jsonl", "a", encoding="utf-8")
        self._log: IO[str] = open(log_folder / f"debug_{timestamp}.log", "a", encoding="utf-8")

    @classmethod
    def from_existing(cls, jsonl_path: Path) -> "SessionLogger":
        stem = jsonl_path.stem  # e.g. "chat_2025-01-01_12-00-00"
        debug_stem = stem.replace("chat_", "debug_", 1)
        debug_path = jsonl_path.parent / f"{debug_stem}.log"
        instance = cls.__new__(cls)
        instance._jsonl = open(jsonl_path, "a", encoding="utf-8")
        instance._log = open(debug_path, "a", encoding="utf-8")
        return instance

    def log_session_start(
        self,
        system_prompt: str,
        startup_docs: list[tuple[Path, str]],
    ) -> None:
        self._jsonl.write(json.dumps({"type": "system_prompt", "content": system_prompt}) + "\n")
        for path, content in startup_docs:
            self._jsonl.write(json.dumps({"type": "startup_doc", "path": str(path), "content": content}) + "\n")
        self._jsonl.flush()

    def log_exchange(
        self,
        user_message: str,
        assistant_message: str,
        rag_chunks: list[Chunk],
        attachments: list[Attachment],
    ) -> None:
        now = _utcnow()
        user_entry = {
            "type": "exchange",
            "timestamp": now,
            "role": "user",
            "content": user_message,
            "rag_chunks": [_chunk_dict(c) for c in rag_chunks],
            "attachments": [_attachment_dict(a) for a in attachments],
        }
        asst_entry = {
            "type": "exchange",
            "timestamp": now,
            "role": "assistant",
            "content": assistant_message,
        }
        self._jsonl.write(json.dumps(user_entry) + "\n")
        self._jsonl.write(json.dumps(asst_entry) + "\n")
        self._jsonl.flush()

    def log_debug(self, message: str) -> None:
        self._log.write(f"[{_utcnow()}] {message}\n")
        self._log.flush()

    def close(self) -> None:
        self._jsonl.close()
        self._log.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunk_dict(chunk: Chunk) -> dict:
    return {"label": chunk.label, "source": str(chunk.source_file), "content": chunk.content}


def _attachment_dict(attachment: Attachment) -> dict:
    return {
        "identifier": attachment.identifier,
        "path": str(attachment.path),
        "content": attachment.content,
        "size_warning": attachment.size_warning,
    }
