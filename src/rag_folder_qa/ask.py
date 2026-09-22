"""Retrieve chunks and ask the local chat model to answer from them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag_folder_qa.citations import Citation, format_citations
from rag_folder_qa.store import StoreError, VectorStore

REFUSAL = "Not found in the documents."

SYSTEM_PROMPT = (
    "You answer questions using only the document excerpts in the user message. "
    "The excerpts come from a local folder the user indexed. "
    "Do not use outside knowledge, and do not invent details. "
    "If the excerpts contain the answer, reply in a short paragraph and cite "
    "the excerpt numbers you used, like [1]. "
    f"If the excerpts do not contain the answer, start your reply with exactly: {REFUSAL} "
    "You may add one short sentence about what the excerpts do cover, but do not guess."
)


def build_messages(question: str, citations: list[Citation]) -> list[dict[str, str]]:
    if citations:
        blocks = [
            f"[{item.rank}] source: {item.source}\n{item.quote}" for item in citations
        ]
        excerpts = "\n\n".join(blocks)
    else:
        excerpts = "(no excerpts were retrieved)"
    user = f"Excerpts:\n\n{excerpts}\n\nQuestion: {question.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def format_context(citations: list[Citation]) -> str:
    if not citations:
        return "(no chunks retrieved)"
    blocks = [
        f"[{item.rank}] {item.source}  (score {item.score:.3f})\n{item.quote}"
        for item in citations
    ]
    return "\n\n".join(blocks)


def render_output(
    answer: str,
    citations: list[Citation],
    *,
    show_context: bool,
    question: str = "",
) -> str:
    lines = [
        "Answer",
        answer.strip(),
        "",
        "Citations",
        format_citations(citations, focus=question),
    ]
    if show_context:
        lines.extend(["", "Retrieved context", format_context(citations)])
    return "\n".join(lines)


def answer_question(
    question: str,
    store_path: Path,
    *,
    top_k: int,
    embed_model: str,
    embedder: Callable[[list[str]], list[list[float]]],
    chatter: Callable[[list[dict[str, str]]], str],
    show_context: bool,
) -> tuple[str, str | None]:
    """Return ``(stdout, warning)``.

    ``warning`` is set when the index was built with a different embedding
    model than the one used for this question. The caller should print it
    on stderr.
    """

    if not question or not question.strip():
        raise ValueError("Question is empty.")
    store = VectorStore(store_path)
    if not store.exists():
        raise StoreError(
            f"No index at {store_path}. Index a folder first: rag-qa ingest <folder>"
        )
    manifest = store.read_manifest()
    warning = _model_warning(manifest, embed_model)
    vectors = embedder([question.strip()])
    if len(vectors) != 1 or not vectors[0]:
        raise ValueError("Embedder did not return a vector for the question.")
    raw_hits = store.query(vectors[0], top_k)
    citations = [
        Citation(
            rank=index,
            source=str(hit["source"]),
            score=float(hit["score"]),
            quote=str(hit["text"]),
        )
        for index, hit in enumerate(raw_hits, start=1)
    ]
    answer = chatter(build_messages(question, citations))
    return (
        render_output(
            answer,
            citations,
            show_context=show_context,
            question=question,
        ),
        warning,
    )


def _model_warning(manifest: dict | None, embed_model: str) -> str | None:
    if not manifest:
        return None
    indexed = manifest.get("embed_model")
    if isinstance(indexed, str) and indexed and indexed != embed_model:
        return (
            f"Index was built with embed model {indexed}, "
            f"but this question uses {embed_model}. Re-run ingest so scores match."
        )
    return None


