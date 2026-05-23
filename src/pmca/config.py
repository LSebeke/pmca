from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIGS_DIR: Path = Path(__file__).parent / "configs"

REQUIRED_FIELDS = ("name", "model", "system_prompt", "top_k_chunks", "log_folder")


class ConfigError(Exception):
    pass


@dataclass
class Config:
    name: str
    model: str
    system_prompt: str
    rag_files: list[Path]
    top_k_chunks: int
    log_folder: Path
    max_attachment_kb: int = 500
    history_token_budget: int = 4000
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


def load_config(config_name: str) -> Config:
    path = _resolve_path(config_name)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    _validate_required(data)
    _validate_log_folder(data["log_folder"])
    _validate_rag_files(data.get("rag_files") or [])

    return Config(
        name=data["name"],
        model=data["model"],
        system_prompt=data["system_prompt"],
        rag_files=[Path(p) for p in (data.get("rag_files") or [])],
        top_k_chunks=data["top_k_chunks"],
        log_folder=Path(data["log_folder"]),
        max_attachment_kb=data.get("max_attachment_kb", 500),
        history_token_budget=data.get("history_token_budget", 4000),
        temperature=data.get("temperature"),
        max_tokens=data.get("max_tokens"),
        top_p=data.get("top_p"),
        frequency_penalty=data.get("frequency_penalty"),
        presence_penalty=data.get("presence_penalty"),
    )


def _resolve_path(config_name: str) -> Path:
    if "/" in config_name or "\\" in config_name or config_name.endswith(".yaml"):
        return Path(config_name)
    return _CONFIGS_DIR / f"{config_name}.yaml"


def _validate_required(data: dict) -> None:
    for field_name in REQUIRED_FIELDS:
        if field_name not in data:
            raise ConfigError(f"Missing required field: {field_name}")


def _validate_log_folder(value: str) -> None:
    if not Path(value).is_absolute():
        raise ConfigError(f"log_folder must be an absolute path, got: {value}")


def _validate_rag_files(paths: list[str]) -> None:
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            raise ConfigError(f"rag_files paths must be absolute, got: {raw}")
        if not p.exists():
            raise ConfigError(f"rag_files path not found: {raw}")
