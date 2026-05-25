from __future__ import annotations

import getpass
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from pmca.attachments import AttachmentAborted, parse_attachment_paths, resolve_attachments, substitute_identifiers
from pmca.config import Config
from pmca.logger import SessionLogger
from pmca.openai_client import chat_completion
from pmca.rag.store import VectorStore
from pmca.tools import execute_write_file, get_tools
from pmca.types import Attachment, Chunk, ToolCallRequest


class ChatSession:
    def __init__(
        self,
        config: Config,
        store: VectorStore,
        logger: SessionLogger,
        unsafe: bool = False,
    ) -> None:
        self.config = config
        self.store = store
        self.logger = logger
        self.unsafe = unsafe
        self.system_prompt: str = config.system_prompt
        self.startup_docs: list[tuple] = list(getattr(config, "startup_docs", []))
        self.history: list[dict] = []
        self.top_k: int = config.top_k_chunks
        self.history_token_budget: int = config.history_token_budget
        self._last_rag_chunks: list[Chunk] = []
        self._next_attachment_n: int = 1
        self.session_attachments: list[Attachment] = []
        self.session_rag_chunks: list[Chunk] = []
        self._system_context: str = _build_system_context()

    def process(self, user_input: str) -> tuple[str | None, int]:
        # 1. Attachments
        try:
            paths = parse_attachment_paths(user_input)
            attachments = resolve_attachments(paths, self.config.max_attachment_kb, self.unsafe, start_n=self._next_attachment_n)
        except AttachmentAborted:
            print("[message cancelled]")
            return None, 0

        self._next_attachment_n += len(attachments)
        self.session_attachments.extend(attachments)
        message = substitute_identifiers(user_input, attachments)

        # 2. Trim history
        turns_dropped = self._trim_history()

        # 3. RAG
        rag_chunks = self.store.query(user_input, self.top_k)
        self._last_rag_chunks = rag_chunks
        self._merge_rag_chunks(rag_chunks)

        # 4. Assemble and call (with tool loop)
        messages = self._build_messages(message, attachments)
        tools = get_tools(self.config)
        response = chat_completion(messages, self.config, tools=tools)

        while isinstance(response, ToolCallRequest):
            approved, result = execute_write_file(response.arguments, self.config)
            self.logger.log_tool_call(
                tool_call_id=response.tool_call_id,
                name=response.name,
                arguments=response.arguments,
                approved=approved,
                result=result,
            )
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": response.tool_call_id,
                    "type": "function",
                    "function": {"name": response.name, "arguments": str(response.arguments)},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": response.tool_call_id,
                "content": result,
            })
            response = chat_completion(messages, self.config, tools=tools)

        # 5. Update history and log
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        self.logger.log_exchange(message, response, rag_chunks, attachments)

        return response, turns_dropped

    def rotate_logger(self) -> Path:
        self.logger.close()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logger = SessionLogger(self.config.log_folder, timestamp)
        self.logger.log_session_start(self.system_prompt, self.startup_docs)
        self._next_attachment_n = 1
        self.session_attachments = []
        self.session_rag_chunks = []
        return self.config.log_folder / f"chat_{timestamp}.jsonl"

    def _trim_history(self) -> int:
        dropped = 0
        while True:
            total = sum(len(m["content"]) // 4 for m in self.history)
            if total <= self.history_token_budget or len(self.history) < 2:
                break
            self.history.pop(0)
            self.history.pop(0)
            dropped += 1
        return dropped

    def _build_messages(
        self,
        user_message: str,
        turn_attachments: list[Attachment],
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "system", "content": self._system_context})

        for path, content in self.startup_docs:
            messages.append({"role": "system", "content": _format_startup_doc(path, content)})

        for att in self.session_attachments:
            messages.append({"role": "system", "content": _format_attachment(att)})

        if self.session_rag_chunks:
            messages.append({"role": "system", "content": _format_rag(self.session_rag_chunks)})

        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        return messages

    def _merge_rag_chunks(self, new_chunks: list[Chunk]) -> None:
        seen = {(c.source_file, c.label) for c in self.session_rag_chunks}
        for chunk in new_chunks:
            key = (chunk.source_file, chunk.label)
            if key not in seen:
                self.session_rag_chunks.append(chunk)
                seen.add(key)


def _format_startup_doc(path: Path, content: str) -> str:
    return f"[STARTUP_DOC]\nFile: {path}\n---\n{content}\n---"


def _format_rag(chunks: list[Chunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[RAG_{i}]\nFile: {chunk.source_file}\nChunk: {chunk.label}\n---\n{chunk.content}\n---"
        )
    return "\n\n".join(parts)


def _format_attachment(att: Attachment) -> str:
    suffix = att.path.suffix.lstrip(".")
    return f"[{att.identifier}]\nFile: {att.path}\nType: {suffix}\n---\n{att.content}\n---"


def _build_system_context() -> str:
    now = datetime.now(timezone.utc).astimezone()
    return (
        f"Session started: {now.strftime('%Y-%m-%d %H:%M:%S %z')}\n"
        f"OS: {platform.system()} {platform.version()}\n"
        f"Host: {platform.node()}\n"
        f"User: {os.environ.get('USER') or getpass.getuser()}\n"
        f"Shell: {os.environ.get('SHELL', 'unknown')}"
    )
