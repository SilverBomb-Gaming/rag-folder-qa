"""Persist chunk vectors in a local Chroma database under ``.rag_store/``."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COLLECTION_NAME = "documents"
MANIFEST_NAME = "manifest.json"
_ADD_BATCH = 256

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")


class StoreError(RuntimeError):
    """The local index is missing or could not be read."""


class _ExplicitEmbeddings:
    """Stand-in so Chroma does not download its default embedding model.

    Vectors are produced by Ollama and passed into ``add`` / ``query``.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "This index stores Ollama embeddings. Pass vectors in; do not embed inside Chroma."
        )

    def embed_query(self, input: str) -> list[float]:
        raise RuntimeError(
            "This index stores Ollama embeddings. Pass vectors in; do not embed inside Chroma."
        )

    @staticmethod
    def name() -> str:
        return "ollama-explicit"


@dataclass(frozen=True)
class StoredChunk:
    id: str
    text: str
    source: str
    chunk_index: int
    start: int
    end: int
    embedding: list[float]


class VectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def exists(self) -> bool:
        if not self.path.is_dir():
            return False
        try:
            client = self._client()
        except StoreError:
            return False
        return COLLECTION_NAME in _collection_names(client)

    def replace(self, chunks: list[StoredChunk], manifest: dict[str, Any]) -> None:
        client = self._client()
        if COLLECTION_NAME in _collection_names(client):
            client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ExplicitEmbeddings(),
        )
        for start in range(0, len(chunks), _ADD_BATCH):
            batch = chunks[start : start + _ADD_BATCH]
            collection.add(
                ids=[item.id for item in batch],
                documents=[item.text for item in batch],
                embeddings=[item.embedding for item in batch],
                metadatas=[
                    {
                        "source": item.source,
                        "chunk_index": item.chunk_index,
                        "start": item.start,
                        "end": item.end,
                    }
                    for item in batch
                ],
            )
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    def read_manifest(self) -> dict[str, Any] | None:
        path = self.path / MANIFEST_NAME
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise StoreError(f"Manifest at {path} is not a JSON object.")
        return payload

    def query(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not self.exists():
            raise StoreError(
                f"No index at {self.path}. Index a folder first: rag-qa ingest <folder>"
            )
        client = self._client()
        collection = client.get_collection(
            COLLECTION_NAME,
            embedding_function=_ExplicitEmbeddings(),
        )
        count = collection.count()
        if count == 0:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = _first_row(result.get("documents"))
        metadatas = _first_row(result.get("metadatas"))
        distances = _first_row(result.get("distances"))
        hits: list[dict[str, Any]] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            meta = metadata or {}
            source = meta.get("source")
            if not isinstance(source, str) or not isinstance(document, str):
                continue
            hits.append(
                {
                    "source": source,
                    "text": document,
                    "score": distance_to_score(distance),
                }
            )
        return hits

    def _client(self):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise StoreError(
                "chromadb is not installed. Install the project with `pip install -e .`."
            ) from exc
        self.path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False),
        )


def distance_to_score(distance: float | None) -> float:
    """Map Chroma cosine distance (``1 - cosine similarity``) to a 0–1 score."""

    if distance is None:
        return 0.0
    score = 1.0 - float(distance)
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _collection_names(client: Any) -> set[str]:
    names: set[str] = set()
    for item in client.list_collections():
        if isinstance(item, str):
            names.add(item)
        else:
            name = getattr(item, "name", None)
            if isinstance(name, str):
                names.add(name)
    return names


def _first_row(value: Any) -> list[Any]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        return list(value[0])
    if isinstance(value, list):
        return list(value)
    return []
