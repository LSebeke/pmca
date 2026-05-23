from __future__ import annotations

from pathlib import Path

from pmca.attachments import AttachmentAborted, parse_attachment_paths, resolve_attachments, substitute_identifiers
from pmca.config import Config
from pmca.logger import SessionLogger
from pmca.openai_client import chat_completion
from pmca.rag.store import VectorStore
from pmca.types import Attachment, Chunk


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
        self.history: list[dict] = []
        self.top_k: int = config.top_k_chunks
        self.history_token_budget: int = config.history_token_budget
        self._last_rag_chunks: list[Chunk] = []

    def process(self, user_input: str) -> tuple[str | None, int]:
        # 1. Attachments
        try:
            paths = parse_attachment_paths(user_input)
            attachments = resolve_attachments(paths, self.config.max_attachment_kb, self.unsafe)
        except AttachmentAborted:
            print("[message cancelled]")
            return None, 0

        message = substitute_identifiers(user_input, attachments)

        # 2. Trim history
        turns_dropped = self._trim_history()

        # 3. RAG
        rag_chunks = self.store.query(user_input, self.top_k)
        self._last_rag_chunks = rag_chunks

        # 4. Assemble and call
        messages = self._build_messages(message, rag_chunks, attachments)
        response = chat_completion(messages, self.config)

        # 5. Update history and log
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        self.logger.log_exchange(message, response, rag_chunks, attachments)

        return response, turns_dropped

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
        rag_chunks: list[Chunk],
        attachments: list[Attachment],
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.config.system_prompt}]

        if rag_chunks:
            messages.append({"role": "system", "content": _format_rag(rag_chunks)})

        for att in attachments:
            messages.append({"role": "system", "content": _format_attachment(att)})

        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        return messages


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
