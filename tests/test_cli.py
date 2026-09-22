"""CLI help and a full ingest → ask pass with Ollama replaced by a fake."""

from typer.testing import CliRunner

from rag_folder_qa.cli import app
from tests.fakes import fake_embed

runner = CliRunner()


class _FakeOllama:
    def __init__(self, base_url: str, **kwargs) -> None:
        self.base_url = base_url

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return fake_embed(texts)

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        question = messages[1]["content"].rsplit("Question:", 1)[-1].lower()
        if "revenue" in question:
            return "Not found in the documents. The excerpts do not mention revenue."
        return "The studio ships on Thursday at 16:00 Pacific. [1]"


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = result.output
    assert "ingest" in output
    assert "ask" in output


def test_ask_help_shows_context_flag():
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--show-context" in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_ask_without_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["ask", "When is ship day?"])
    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "ingest" in combined


def test_cli_ingest_and_ask_cites_file(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "handbook.md").write_text(
        "Ship day is Thursday. The cutoff is 16:00 Pacific.\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("rag_folder_qa.cli.OllamaClient", _FakeOllama)

    ingested = runner.invoke(app, ["ingest", str(docs)])
    assert ingested.exit_code == 0, ingested.output
    assert "handbook.md" in ingested.output
    assert (tmp_path / ".rag_store").is_dir()

    asked = runner.invoke(
        app,
        ["ask", "What day does the studio ship, and what is the cutoff?", "--show-context"],
    )
    assert asked.exit_code == 0, asked.output
    assert "Thursday" in asked.output
    assert "handbook.md" in asked.output
    assert "Retrieved context" in asked.output
    assert "(rank 1, score" in asked.output

    missing = runner.invoke(app, ["ask", "What was the studio revenue last year?"])
    assert missing.exit_code == 0, missing.output
    assert missing.output.startswith("Answer\nNot found in the documents.")
    assert "handbook.md" in missing.output
