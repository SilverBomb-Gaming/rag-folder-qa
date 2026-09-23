"""Deterministic stand-ins so pipeline tests never call Ollama."""

from __future__ import annotations

import hashlib


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([byte / 255.0 for byte in digest[:8]])
    return vectors
