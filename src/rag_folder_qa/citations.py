"""Format retrieval hits as citations a reader can check."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "many",
        "does",
        "did",
        "was",
        "were",
        "are",
        "have",
        "has",
        "into",
        "about",
        "last",
        "year",
        "your",
        "their",
    }
)


@dataclass(frozen=True)
class Citation:
    rank: int
    source: str
    score: float
    quote: str


def assert_citation_source(source: str) -> str:
    """Reject absolute paths and any ``..`` segment.

    Citations are always stored relative to the ingested folder. Refusing
    anything else keeps a bad metadata row from printing a path outside that
    folder.
    """

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Citation source is empty.")
    path = Path(source)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "Citation source must stay under the docs root "
            f"(relative path, no '..'): {source}"
        )
    return path.as_posix()


def best_quote(text: str, focus: str = "", max_chars: int = 180) -> str:
    """Pick a short quote, preferring sentences that overlap the question.

    With no focus string, this is the start of the chunk. With a question,
    every run of sentences that fits in ``max_chars`` is scored by how many
    question terms it contains. The highest-scoring run is the quote.
    """

    if max_chars < 1:
        raise ValueError("max_chars must be positive.")
    collapsed = _collapse(text)
    if not focus.strip() or len(collapsed) <= max_chars:
        return shorten_quote(collapsed, max_chars)
    terms = _focus_terms(focus)
    sentences = [sentence for sentence in _sentences(collapsed) if len(sentence.split()) >= 4]
    if not sentences:
        sentences = _sentences(collapsed)
    if not terms or not sentences:
        return shorten_quote(collapsed, max_chars)

    best = ""
    best_key: tuple[int, int, int] | None = None
    for start in range(len(sentences)):
        chosen = ""
        for end in range(start, len(sentences)):
            candidate = sentences[end] if not chosen else f"{chosen} {sentences[end]}"
            if len(candidate) > max_chars:
                break
            chosen = candidate
            # More matched terms, then fewer sentences, then the earlier span.
            key = (_term_hits(chosen, terms), -(end - start), -start)
            if best_key is None or key > best_key:
                best_key = key
                best = chosen
    if not best or best_key is None or best_key[0] == 0:
        return shorten_quote(collapsed, max_chars)
    return best


def _collapse(text: str) -> str:
    # Headings become their own sentences so a title does not glue onto the next line.
    headed = re.sub(r"(?m)^#{1,6}\s*(.+?)\s*$", r"\1.", text)
    return " ".join(headed.split())


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _focus_terms(focus: str) -> set[str]:
    return {
        token
        for token in (match.group(0).lower() for match in _WORD.finditer(focus))
        if len(token) > 2 and token not in _STOPWORDS
    }


def _term_hits(window: str, terms: set[str]) -> int:
    return len(_matched_terms(window, terms))


def _matched_terms(window: str, terms: set[str]) -> set[str]:
    words = {match.group(0).lower() for match in _WORD.finditer(window)}
    matched: set[str] = set()
    for term in terms:
        if term in words or any(
            min(len(word), len(term)) >= 4 and (word.startswith(term) or term.startswith(word))
            for word in words
        ):
            matched.add(term)
    return matched


def shorten_quote(text: str, max_chars: int = 180) -> str:
    """Collapse whitespace and trim a quote to ``max_chars``."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive.")
    snippet = " ".join(text.split())
    if len(snippet) <= max_chars:
        return snippet
    if max_chars == 1:
        return "…"
    return snippet[: max_chars - 1].rstrip() + "…"


def format_citation(
    citation: Citation,
    *,
    max_chars: int = 180,
    focus: str = "",
) -> str:
    source = assert_citation_source(citation.source)
    snippet = best_quote(citation.quote, focus, max_chars)
    return (
        f"[{citation.rank}] {source}  "
        f"(rank {citation.rank}, score {citation.score:.3f})\n"
        f'    "{snippet}"'
    )


def format_citations(
    citations: list[Citation],
    *,
    max_chars: int = 180,
    focus: str = "",
) -> str:
    if not citations:
        return "(no citations)"
    return "\n".join(
        format_citation(citation, max_chars=max_chars, focus=focus)
        for citation in citations
    )
