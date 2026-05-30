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


@dataclass
class ScratchpadEntry:
    title: str    # short label making the origin of the information clear
    content: str  # arbitrary excerpt from a tool call return


@dataclass(frozen=True)
class ActiveSkill:
    name: str
    content: str
    directory: Path


@dataclass
class ToolCallRequest:
    tool_call_id: str
    name: str
    arguments: dict
