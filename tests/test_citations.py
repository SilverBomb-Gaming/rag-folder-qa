"""Citation lines name the source, rank, score, and a short quote."""

from pathlib import Path

import pytest

from rag_folder_qa.citations import Citation, format_citation, format_citations


def test_citation_includes_path_rank_score_and_quote():
    text = format_citation(
        Citation(rank=2, source="faq.md", score=0.5, quote="Twenty days of paid time off.")
    )
    assert text == '[2] faq.md  (rank 2, score 0.500)\n    "Twenty days of paid time off."'


def test_nested_relative_path_is_kept():
    text = format_citation(
        Citation(rank=1, source="team/faq.md", score=1, quote="Priya handles laptops.")
    )
    assert text.startswith("[1] team/faq.md  (rank 1, score 1.000)")


def test_whitespace_is_collapsed_and_quote_is_trimmed():
    quote = "Ship day\n\nis   Thursday. " + ("word " * 80)
    text = format_citation(
        Citation(rank=1, source="handbook.md", score=0.25, quote=quote),
        max_chars=40,
    )
    quote_line = text.splitlines()[1].strip()
    assert quote_line.startswith('"Ship day is Thursday.')
    assert "\n" not in quote_line
    assert "  " not in quote_line
    assert quote_line.endswith('…"')
    assert len(quote_line.strip('"')) <= 40


def test_multiple_citations_keep_rank_order():
    rendered = format_citations(
        [
            Citation(1, "handbook.md", 0.9, "Thursday"),
            Citation(2, "faq.md", 0.4, "20 days"),
        ]
    )
    assert rendered.index("[1]") < rendered.index("[2]")
    assert "handbook.md" in rendered
    assert "faq.md" in rendered


def test_quote_prefers_the_sentence_that_matches_the_question():
    handbook = (
        Path(__file__).resolve().parents[1] / "sample-docs" / "handbook.md"
    ).read_text(encoding="utf-8")
    text = format_citation(
        Citation(rank=1, source="handbook.md", score=0.71, quote=handbook),
        focus="What day does Lumen Finch ship builds, and what is the cutoff?",
    )
    assert (
        '"We ship the weekly playable build on Thursday. The cutoff is 16:00 Pacific."'
        in text
    )


def test_no_citations_placeholder():
    assert format_citations([]) == "(no citations)"


def test_absolute_path_is_rejected():
    with pytest.raises(ValueError, match="docs root"):
        format_citation(Citation(1, "/etc/passwd", 0.2, "secret"))


def test_parent_segment_is_rejected():
    with pytest.raises(ValueError, match="docs root"):
        format_citation(Citation(1, "notes/../../secret.md", 0.2, "secret"))


def test_empty_source_is_rejected():
    with pytest.raises(ValueError):
        format_citation(Citation(1, "  ", 0.2, "secret"))
