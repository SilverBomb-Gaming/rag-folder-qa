"""Chroma round-trip with vectors supplied by the test, not by Ollama."""

from rag_folder_qa.store import StoredChunk, VectorStore, distance_to_score


def test_distance_to_score_clamps_cosine_distance():
    assert distance_to_score(0.0) == 1.0
    assert distance_to_score(0.25) == 0.75
    assert distance_to_score(2.0) == 0.0
    assert distance_to_score(None) == 0.0


def test_query_returns_nearest_source(tmp_path):
    store = VectorStore(tmp_path / ".rag_store")
    store.replace(
        [
            StoredChunk(
                id="handbook.md#0",
                text="Ship day is Thursday.",
                source="handbook.md",
                chunk_index=0,
                start=0,
                end=22,
                embedding=[1.0, 0.0, 0.0],
            ),
            StoredChunk(
                id="faq.md#0",
                text="Full-time teammates get 20 days.",
                source="faq.md",
                chunk_index=0,
                start=0,
                end=30,
                embedding=[0.0, 1.0, 0.0],
            ),
        ],
        {"docs_root": str(tmp_path), "embed_model": "fake", "files": 2, "chunks": 2},
    )
    hits = store.query([1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0]["source"] == "handbook.md"
    assert hits[0]["text"] == "Ship day is Thursday."
    assert hits[0]["score"] > 0.99
    manifest = store.read_manifest()
    assert manifest is not None
    assert manifest["embed_model"] == "fake"
