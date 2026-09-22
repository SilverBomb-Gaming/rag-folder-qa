"""Defaults match .env.example and a shell export beats the file."""

from rag_folder_qa.config import ConfigError, settings_from_env

_KEYS = (
    "OLLAMA_BASE_URL",
    "OLLAMA_CHAT_MODEL",
    "OLLAMA_EMBED_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
)


def _clear(monkeypatch):
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_match_env_example(monkeypatch, tmp_path):
    _clear(monkeypatch)
    settings = settings_from_env(tmp_path / "missing.env")
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.chat_model == "llama3.2"
    assert settings.embed_model == "nomic-embed-text"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.top_k == 4


def test_dotenv_does_not_override_existing_env(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("TOP_K", "2")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TOP_K=9\nOLLAMA_CHAT_MODEL=from-file\nexport OLLAMA_EMBED_MODEL='custom-embed'\n",
        encoding="utf-8",
    )
    settings = settings_from_env(env_file)
    assert settings.top_k == 2
    assert settings.chat_model == "from-file"
    assert settings.embed_model == "custom-embed"


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("CHUNK_SIZE", "50")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    try:
        settings_from_env(tmp_path / "missing.env")
    except ConfigError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("expected ConfigError")
