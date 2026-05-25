from __future__ import annotations

import re
from pathlib import Path

from pmca.types import Attachment

_TOKEN_RE = re.compile(r"\[\[([^\]]+)\]\]")

SECURITY_PROMPT = "You are about to attach {path}. Have you reviewed it for secrets? (y/n) "


class AttachmentError(Exception):
    pass


class AttachmentAborted(Exception):
    pass


def parse_attachment_paths(message: str) -> list[Path]:
    paths = []
    for match in _TOKEN_RE.finditer(message):
        raw = match.group(1).strip('"')
        path = Path(raw)
        if not path.is_absolute():
            raise AttachmentError(f"Attachment path must be absolute, got: {raw}")
        paths.append(path)
    return paths


def resolve_attachments(
    paths: list[Path],
    max_attachment_kb: int,
    unsafe: bool,
) -> list[Attachment]:
    attachments: list[Attachment] = []

    for i, path in enumerate(paths, start=1):
        if not path.exists():
            raise AttachmentError(f"Attachment not found: {path}")

        size_kb = path.stat().st_size / 1024
        size_warning = size_kb > max_attachment_kb
        if size_warning:
            print(f"Warning: {path} is {size_kb:.0f} KB (limit {max_attachment_kb} KB)")

        if not unsafe:
            answer = input(SECURITY_PROMPT.format(path=path))
            if answer.strip().lower() != "y":
                raise AttachmentAborted(f"Attachment of {path} cancelled by user")

        content = path.read_text(encoding="utf-8", errors="replace")
        attachments.append(Attachment(
            path=path,
            content=content,
            identifier=f"CONTEXT_{i}",
            size_warning=size_warning,
        ))

    return attachments


def substitute_identifiers(message: str, attachments: list[Attachment]) -> str:
    path_to_id = {a.path: a.identifier for a in attachments}

    def _replace(match: re.Match) -> str:
        raw = match.group(1).strip('"')
        return path_to_id.get(Path(raw), match.group(0))

    return _TOKEN_RE.sub(_replace, message)
