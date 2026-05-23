from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    content: str
    source_file: Path
    label: str


@dataclass
class Attachment:
    path: Path
    content: str
    identifier: str
    size_warning: bool
