"""Command-line entry points: ``ingest`` a folder, then ``ask`` questions."""

from __future__ import annotations

from pathlib import Path

import typer

from rag_folder_qa import __version__
from rag_folder_qa.ask import answer_question
from rag_folder_qa.config import ConfigError, settings_from_env
from rag_folder_qa.ingest import IngestError, ingest_folder
from rag_folder_qa.ollama import OllamaClient, OllamaError
from rag_folder_qa.store import StoreError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Ask questions about a local folder of Markdown and text files.",
)


def _version(value: bool) -> bool:
    if value:
        typer.echo(__version__)
        raise typer.Exit()
    return value


@app.callback()
def _callback(
    version: bool = typer.Option(
        False,
        "--version",
        help="Print the version and exit.",
        callback=_version,
        is_eager=True,
    ),
) -> None:
    """Ask questions about a local folder of Markdown and text files."""


@app.command()
def ingest(
    folder: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Folder of .md and .txt files to index.",
    ),
    store: Path = typer.Option(
        Path(".rag_store"),
        "--store",
        help="Directory for the local Chroma index.",
    ),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        min=1,
        help="Chunk length in characters. Defaults to CHUNK_SIZE.",
    ),
    chunk_overlap: int | None = typer.Option(
        None,
        "--chunk-overlap",
        min=0,
        help="Overlap in characters. Defaults to CHUNK_OVERLAP.",
    ),
) -> None:
    """Chunk a folder, embed it with Ollama, and save the index locally."""

    settings = _settings()
    size = settings.chunk_size if chunk_size is None else chunk_size
    overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap
    client = OllamaClient(settings.ollama_base_url)
    typer.echo(f"Indexing {folder.resolve()}")
    typer.echo(
        f"Embedding with {settings.embed_model} via {settings.ollama_base_url} "
        f"(chunk size {size}, overlap {overlap})"
    )
    try:
        report = ingest_folder(
            folder,
            store,
            chunk_size=size,
            chunk_overlap=overlap,
            embedder=lambda texts: client.embed(settings.embed_model, texts),
            embed_model=settings.embed_model,
            chat_model=settings.chat_model,
        )
    except (IngestError, OllamaError, ConfigError, ValueError) as exc:
        _fail(str(exc))
    for skipped in report.skipped:
        typer.echo(f"Skipped: {skipped}", err=True)
    listed = "\n".join(f"  {name}" for name in report.files)
    typer.echo(listed)
    typer.echo(
        f"Stored {report.chunks} chunks from {len(report.files)} files in {report.store_path}."
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from the indexed documents."),
    store: Path = typer.Option(
        Path(".rag_store"),
        "--store",
        help="Directory of an existing Chroma index.",
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        help="How many chunks to retrieve. Defaults to TOP_K.",
    ),
    show_context: bool = typer.Option(
        False,
        "--show-context",
        help="Print the retrieved chunks in full, after the citations.",
    ),
) -> None:
    """Retrieve the closest chunks and answer with citations."""

    settings = _settings()
    k = settings.top_k if top_k is None else top_k
    client = OllamaClient(settings.ollama_base_url)
    try:
        text, warning = answer_question(
            question,
            store,
            top_k=k,
            embed_model=settings.embed_model,
            embedder=lambda texts: client.embed(settings.embed_model, texts),
            chatter=lambda messages: client.chat(settings.chat_model, messages),
            show_context=show_context,
        )
    except (StoreError, OllamaError, ConfigError, ValueError) as exc:
        _fail(str(exc))
    if warning:
        typer.echo(f"Warning: {warning}", err=True)
    typer.echo(text)


def _settings():
    try:
        return settings_from_env()
    except ConfigError as exc:
        _fail(str(exc))


def _fail(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def main() -> None:
    app()
