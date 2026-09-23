"""Split document text into overlapping character windows."""

from __future__ import annotations

from dataclasses import dataclass

_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    """A slice of normalized document text.

    ``start`` and ``end`` are offsets into the normalized text (newlines as
    ``\\n``), not necessarily the raw file bytes.
    """

    text: str
    index: int
    start: int
    end: int


def validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size "
            f"(got size={chunk_size}, overlap={chunk_overlap})."
        )


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    Breaks prefer a paragraph, then a line, then a sentence, then a space,
    as long as that break keeps the chunk reasonably large. Adjacent chunks
    share ``chunk_overlap`` characters. Whitespace-only input yields no chunks.
    """

    validate_chunk_settings(chunk_size, chunk_overlap)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    length = len(normalized)
    index = 0
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _snap_end(normalized, start, end, chunk_size)
        raw = normalized[start:end]
        left = len(raw) - len(raw.lstrip())
        trimmed = raw.strip()
        if trimmed:
            content_end = start + len(raw.rstrip())
            chunks.append(
                Chunk(
                    text=trimmed,
                    index=index,
                    start=start + left,
                    end=content_end,
                )
            )
            index += 1
        if end >= length:
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def _snap_end(text: str, start: int, end: int, chunk_size: int) -> int:
    window = text[start:end]
    min_break = max(1, int(chunk_size * 0.4))
    for separator in _SEPARATORS:
        found = window.rfind(separator)
        if found >= min_break:
            return start + found + len(separator)
    return end
