"""Chunking does not need Ollama or a vector store."""

import pytest

from rag_folder_qa.chunking import chunk_text, validate_chunk_settings


def test_short_text_is_one_chunk():
    chunks = chunk_text("Hello there.", chunk_size=800, chunk_overlap=120)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello there."
    assert chunks[0].index == 0
    assert chunks[0].start == 0
    assert chunks[0].end == len("Hello there.")


def test_empty_and_whitespace_yield_nothing():
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   \n\n\t", 100, 10) == []


def test_crlf_is_normalized():
    chunks = chunk_text("alpha\r\nbeta", chunk_size=800, chunk_overlap=10)
    assert chunks[0].text == "alpha\nbeta"


def test_long_text_overlaps_and_covers_every_character():
    text = "abcdefghijklmnopqrstuvwxyz" * 20
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert chunks[1].text[:20] == chunks[0].text[-20:]
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    covered = [False] * len(text)
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        for index in range(chunk.start, chunk.end):
            covered[index] = True
    assert all(covered)


def test_prefers_a_paragraph_break():
    text = ("A" * 50) + "\n\n" + ("B" * 500)
    chunks = chunk_text(text, chunk_size=80, chunk_overlap=10)
    assert chunks[0].text == "A" * 50
    assert "B" in chunks[1].text


def test_mixed_document_covers_non_whitespace():
    text = (
        "Ship day is Thursday.\n\n"
        "The cutoff is 16:00 Pacific.\n"
        + ("notes " * 80)
    )
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=15)
    assert len(chunks) > 1
    for index, char in enumerate(text):
        if char.isspace():
            continue
        assert any(chunk.start <= index < chunk.end for chunk in chunks)


def test_large_overlap_still_advances():
    text = "word " * 400
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=49)
    assert len(chunks) > 1
    starts = [chunk.start for chunk in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


def test_invalid_chunk_settings():
    with pytest.raises(ValueError):
        validate_chunk_settings(0, 0)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, chunk_overlap=-1)
