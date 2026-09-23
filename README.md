# rag-folder-qa

Ask a folder of notes a question and get an answer with citations. The index stays on your machine: [Ollama](https://ollama.com) embeds and chats, Chroma stores the vectors under `.rag_store/`, and the happy path does not use a cloud API key.

This is a portfolio demo of a small retrieval-augmented generation (RAG) pipeline. It is meant for a local handbook, a design folder, or a personal set of notes — the kind of corpus you can read end to end.

The sample corpus is fiction: a tiny studio called Lumen Finch Games and their puzzle game, Paper Lanterns. Nothing in `sample-docs/` is a real company, a real password, or someone else's game.

Author: Alfredo Cardona ([SilverBomb-Gaming](https://github.com/SilverBomb-Gaming)).

## What it is

- A command-line tool that indexes `.md` and `.txt` files in a directory you choose.
- An `ask` command that retrieves the closest chunks and asks a local model to answer from those chunks.
- Citations on every answer: source path, rank, similarity score, and a short quote you can check against the file.

## What it is not

- A hosted search product, a chat UI, or a multi-user service.
- A crawler. It does not read PDF, Office, or HTML files (yet).
- A client of OpenAI or any other cloud model API.

## Demo (about ten minutes, plus model downloads)

You need Python 3.11+ and Ollama.

```bash
# Pull the two local models. llama3.2 is the chat model; nomic-embed-text embeds chunks.
ollama pull llama3.2
ollama pull nomic-embed-text

# From a fresh clone
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# Defaults match .env.example. Copy it only if you want to change a model or the chunk size.
cp .env.example .env

rag-qa ingest sample-docs

rag-qa ask "What day does Lumen Finch ship builds, and what is the cutoff?"

rag-qa ask "What was Lumen Finch's revenue last year?"
```

The first question is in the handbook: the studio ships on **Thursday**, and the cutoff is **16:00 Pacific**. The answer should cite `handbook.md`.

The second question is not in the sample corpus. The prompt tells the model to start that reply with `Not found in the documents.`

Print the retrieved chunks in full:

```bash
rag-qa ask --show-context "How many paid time off days do full-time teammates get?"
```

That one is in `faq.md`: 20 days.

Run the tests without Ollama:

```bash
pip install -e ".[dev]"
pytest
```

`python -m rag_folder_qa` is the same CLI as `rag-qa`.

## How the pipeline works

**Ingest.** `rag-qa ingest <folder>` walks the folder and keeps `.md` and `.txt` files. A path that resolves outside that folder is skipped, including a symlink whose target lives elsewhere. Each file is split into overlapping character windows. Ollama's embedding model (`nomic-embed-text` by default) turns each chunk into a vector. Chroma writes the vectors, the text, and the source path to `.rag_store/`. Running ingest again replaces that index.

**Retrieve.** `rag-qa ask` embeds the question with the same model and asks Chroma for the top-k nearest chunks (cosine similarity).

**Generate.** Those chunks are placed in a prompt that tells the chat model to answer only from the excerpts. If the excerpts do not contain the answer, the model is told to say `Not found in the documents.` The CLI then prints the reply and a citation for each retrieved chunk, whether or not the excerpt actually answered the question — so you can see what the model was looking at.

```
folder/*.md, *.txt
        │
        ▼
     chunk
        │
        ▼
  Ollama embed ──► .rag_store/  (Chroma)
                         │
           question ──► embed ──► top-k
                         │
                         ▼
                   Ollama chat
                         │
                         ▼
               answer + citations
```

Example citation:

```
Answer
We ship the weekly playable build on Thursday. The cutoff is 16:00 Pacific. [1]

Citations
[1] handbook.md  (rank 1, score 0.710)
    "We ship the weekly playable build on Thursday. The cutoff is 16:00 Pacific."
```

The quoted line is the part of the chunk that overlaps the question. The score is cosine similarity from Chroma (`1` is the closest). Rank `1` is the nearest chunk. The number moves a little between machines; the source file should not.

## Configuration

Environment variables already set in the shell win over `.env`.

| Variable | Default | Role |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | Model that writes the answer |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model that embeds chunks and questions |
| `CHUNK_SIZE` | `800` | Chunk length in characters |
| `CHUNK_OVERLAP` | `120` | Characters shared by neighboring chunks |
| `TOP_K` | `4` | Chunks retrieved for each question |

One-off overrides: `rag-qa ingest ./notes --chunk-size 500` and `rag-qa ask --top-k 6 "..."`.

The index defaults to `.rag_store/` in the current working directory. Pass `--store` when you keep more than one corpus. If `ask` uses a different embedding model than the one recorded at ingest, the CLI warns you to re-index.

If Ollama is not running, the CLI tells you to start `ollama serve` and pull the two models. The first request after a pull can take a while while the model loads.

## Scope

In scope: one person, one machine, plain text and Markdown, local models, and citations you can check by opening the file.

Out of scope:

- Accounts, auth, or more than one user
- Tools that write, delete, or send documents anywhere but local Ollama
- Production search (no hybrid keyword index, no reranker, no evaluation harness)
- Formats other than `.md` and `.txt`
- Watching the folder and updating the index for you — run `ingest` again when the files change

## Layout

```
.
├── pyproject.toml
├── LICENSE
├── .env.example
├── .gitignore
├── sample-docs/                  fictional Lumen Finch handbook
│   ├── handbook.md
│   ├── faq.md
│   ├── onboarding.txt
│   ├── values.md
│   └── release-checklist.md
├── src/rag_folder_qa/
│   ├── cli.py                    ingest and ask
│   ├── chunking.py               size, overlap, paragraph breaks
│   ├── paths.py                  stay inside the docs root
│   ├── ollama.py                 HTTP client for /api/embed and /api/chat
│   ├── store.py                  Chroma under .rag_store/
│   ├── ingest.py
│   ├── ask.py                    prompt, refusal rule, output
│   ├── citations.py
│   └── config.py                 .env and defaults
└── tests/                        no Ollama process required
```

## License

MIT © 2026 Alfredo Cardona
