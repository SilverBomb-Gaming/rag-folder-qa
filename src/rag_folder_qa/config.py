"""Settings from the environment and an optional ``.env`` file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rag_folder_qa.chunking import validate_chunk_settings


class ConfigError(ValueError):
    """A setting is missing or not usable."""


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    chat_model: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def load_dotenv(path: Path | None = None) -> None:
    """Load ``KEY=value`` lines into the environment without overriding it.

    Existing variables win, so a shell export beats the file. Missing files
    are ignored.
    """

    env_path = path if path is not None else Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def settings_from_env(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file)
    chunk_size = _positive_int("CHUNK_SIZE", 800)
    chunk_overlap = _non_negative_int("CHUNK_OVERLAP", 120)
    try:
        validate_chunk_settings(chunk_size, chunk_overlap)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
    if not base_url:
        raise ConfigError("OLLAMA_BASE_URL is empty.")
    chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2").strip()
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
    if not chat_model:
        raise ConfigError("OLLAMA_CHAT_MODEL is empty.")
    if not embed_model:
        raise ConfigError("OLLAMA_EMBED_MODEL is empty.")
    return Settings(
        ollama_base_url=base_url.rstrip("/"),
        chat_model=chat_model,
        embed_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=_positive_int("TOP_K", 4),
    )


def _positive_int(name: str, default: int) -> int:
    value = _read_int(name, default)
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}.")
    return value


def _non_negative_int(name: str, default: int) -> int:
    value = _read_int(name, default)
    if value < 0:
        raise ConfigError(f"{name} must be >= 0, got {value}.")
    return value


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc
