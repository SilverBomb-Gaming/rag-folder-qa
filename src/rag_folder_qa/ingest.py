"""Walk a folder, chunk it, embed it, and replace the local index."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rag_folder_qa.chunking import chunk_text, validate_chunk_settings
from rag_folder_qa.paths import relative_source, scan_documents
from rag_folder_qa.store import StoredChunk, VectorStore


class IngestError(RuntimeError):
    """The folder could not be indexed."""


@dataclass(frozen=True)
class IngestReport:
    files: tuple[str, ...]
    chunks: int
    skipped: tuple[str, ...]
    store_path: Path


def ingest_folder(
    folder: Path,
    store_path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Callable[[list[str]], list[list[float]]],
    embed_model: str,
    chat_model: str,
) -> IngestReport:
    """Index ``folder`` into ``store_path``, replacing any previous index."""

    validate_chunk_settings(chunk_size, chunk_overlap)
    root = folder.resolve()
    scan = scan_documents(root)
    if not scan.files:
        raise IngestError(f"No .md or .txt files under {root}.")

    prepared: list[tuple[str, str, int, int, int]] = []
    sources: list[str] = []
    for path in scan.files:
        source = relative_source(root, path)
        sources.append(source)
        text = path.read_text(encoding="utf-8", errors="replace")
        for chunk in chunk_text(text, chunk_size, chunk_overlap):
            prepared.append((source, chunk.text, chunk.index, chunk.start, chunk.end))
    if not prepared:
        raise IngestError(f"Files under {root} had no text to index.")

    vectors = embedder([item[1] for item in prepared])
    if len(vectors) != len(prepared):
        raise IngestError(
            f"Embedder returned {len(vectors)} vectors for {len(prepared)} chunks."
        )

    stored: list[StoredChunk] = []
    for (source, text, index, start, end), vector in zip(prepared, vectors, strict=True):
        if not vector:
            raise IngestError("Embedder returned an empty vector.")
        stored.append(
            StoredChunk(
                id=f"{source}#{index}",
                text=text,
                source=source,
                chunk_index=index,
                start=start,
                end=end,
                embedding=[float(value) for value in vector],
            )
        )

    manifest = {
        "docs_root": str(root),
        "embed_model": embed_model,
        "chat_model": chat_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": len(sources),
        "chunks": len(stored),
    }
    VectorStore(store_path).replace(stored, manifest)
    return IngestReport(
        files=tuple(sources),
        chunks=len(stored),
        skipped=scan.skipped,
        store_path=store_path,
    )
