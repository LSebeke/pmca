from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from pmca.types import Attachment


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
        attachments: list[Attachment],
    ) -> None:
        now = _utcnow()
        user_entry = {
            "type": "exchange",
            "timestamp": now,
            "role": "user",
            "content": user_message,
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

    def log_tool_call(
        self,
        tool_call_id: str,
        name: str,
        arguments: dict,
        approved: bool,
        result: str,
    ) -> None:
        entry = {
            "type": "tool_call",
            "timestamp": _utcnow(),
            "tool_call_id": tool_call_id,
            "name": name,
            "arguments": arguments,
            "approved": approved,
            "result": result,
        }
        self._jsonl.write(json.dumps(entry) + "\n")
        self._jsonl.flush()

    def log_debug(self, message: str) -> None:
        self._log.write(f"[{_utcnow()}] {message}\n")
        self._log.flush()

    def close(self) -> None:
        self._jsonl.close()
        self._log.close()


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _attachment_dict(attachment: Attachment) -> dict:
    return {
        "identifier": attachment.identifier,
        "path": str(attachment.path),
        "content": attachment.content,
        "size_warning": attachment.size_warning,
    }
