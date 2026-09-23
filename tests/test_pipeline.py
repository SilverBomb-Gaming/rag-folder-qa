"""Ingest and ask, with a fake embedder and a fake chat model."""

from pathlib import Path

import pytest

from rag_folder_qa.ask import REFUSAL, answer_question, build_messages
from rag_folder_qa.chunking import chunk_text
from rag_folder_qa.citations import Citation
from rag_folder_qa.ingest import ingest_folder
from rag_folder_qa.store import StoreError, VectorStore
from tests.fakes import fake_embed

SAMPLE_DOCS = Path(__file__).resolve().parents[1] / "sample-docs"


def _ingest(folder: Path, store: Path, embed_model: str = "fake-embed"):
    return ingest_folder(
        folder,
        store,
        chunk_size=800,
        chunk_overlap=120,
        embedder=fake_embed,
        embed_model=embed_model,
        chat_model="fake-chat",
    )


def test_prompt_requires_not_found_phrase():
    messages = build_messages(
        "What was revenue?",
        [Citation(1, "handbook.md", 0.4, "Ship day is Thursday.")],
    )
    assert messages[0]["role"] == "system"
    assert REFUSAL in messages[0]["content"]
    assert "handbook.md" in messages[1]["content"]
    assert "What was revenue?" in messages[1]["content"]


def test_ingest_skips_symlink_and_keeps_relative_paths(tmp_path):
    docs = tmp_path / "docs"
    (docs / "team").mkdir(parents=True)
    (docs / "handbook.md").write_text(
        "Ship day is Thursday at 16:00 Pacific.\n", encoding="utf-8"
    )
    (docs / "team" / "faq.md").write_text(
        "Full-time teammates get 20 days of paid time off.\n", encoding="utf-8"
    )
    outside = tmp_path / "secret.txt"
    outside.write_text("The revenue was 999999.\n", encoding="utf-8")
    (docs / "leak.txt").symlink_to(outside)

    report = _ingest(docs, tmp_path / ".rag_store")
    assert report.files == ("handbook.md", "team/faq.md")
    assert report.skipped

    hits = VectorStore(tmp_path / ".rag_store").query(fake_embed(["ship day"])[0], top_k=4)
    blob = " ".join(hit["text"] for hit in hits)
    sources = {hit["source"] for hit in hits}
    assert sources == {"handbook.md", "team/faq.md"}
    assert "999999" not in blob
    assert all(not source.startswith("/") and ".." not in source for source in sources)


def test_reingest_replaces_previous_chunks(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    notes = docs / "notes.md"
    notes.write_text("The office plant is named Fern.\n", encoding="utf-8")
    store = tmp_path / ".rag_store"
    _ingest(docs, store)
    notes.write_text("The office plant is named Moss.\n", encoding="utf-8")
    _ingest(docs, store)
    hits = VectorStore(store).query(fake_embed(["plant"])[0], top_k=4)
    blob = " ".join(hit["text"] for hit in hits)
    assert "Moss" in blob
    assert "Fern" not in blob


def test_sample_docs_ingest_and_refusal_answer(tmp_path):
    store = tmp_path / ".rag_store"
    report = _ingest(SAMPLE_DOCS, store)
    assert report.files == (
        "faq.md",
        "handbook.md",
        "onboarding.txt",
        "release-checklist.md",
        "values.md",
    )
    assert report.chunks >= len(report.files)
    handbook = (SAMPLE_DOCS / "handbook.md").read_text(encoding="utf-8")
    assert any(
        "Thursday" in chunk.text and "16:00" in chunk.text
        for chunk in chunk_text(handbook, 800, 120)
    )

    def chatter(messages):
        assert REFUSAL in messages[0]["content"]
        joined = messages[1]["content"]
        assert "handbook.md" in joined
        assert "faq.md" in joined
        return f"{REFUSAL} Revenue is not discussed in the excerpts."

    text, warning = answer_question(
        "What was Lumen Finch's revenue last year?",
        store,
        top_k=20,
        embed_model="fake-embed",
        embedder=fake_embed,
        chatter=chatter,
        show_context=True,
    )
    assert text.startswith(f"Answer\n{REFUSAL}")
    assert "handbook.md" in text
    assert "(rank 1, score" in text
    assert "Retrieved context" in text
    assert str(SAMPLE_DOCS) not in text
    assert warning is None


def test_embed_model_mismatch_warns(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "handbook.md").write_text("Ship day is Thursday.\n", encoding="utf-8")
    store = tmp_path / ".rag_store"
    _ingest(docs, store, embed_model="nomic-embed-text")
    text, warning = answer_question(
        "When is ship day?",
        store,
        top_k=2,
        embed_model="other-embed",
        embedder=fake_embed,
        chatter=lambda messages: "Thursday. [1]",
        show_context=False,
    )
    assert "Thursday" in text
    assert "handbook.md" in text
    assert warning is not None
    assert "nomic-embed-text" in warning
    assert "other-embed" in warning


def test_ask_without_index_raises(tmp_path):
    with pytest.raises(StoreError, match="ingest"):
        answer_question(
            "hello",
            tmp_path / ".rag_store",
            top_k=2,
            embed_model="fake-embed",
            embedder=fake_embed,
            chatter=lambda messages: "nope",
            show_context=False,
        )
