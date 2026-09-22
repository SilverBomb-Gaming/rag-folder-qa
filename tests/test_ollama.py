"""The HTTP client is tested with a fake transport, not a live Ollama."""

import json

import httpx
import pytest

from rag_folder_qa.ollama import OllamaClient, OllamaError


def _client(handler) -> OllamaClient:
    return OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))


def test_embed_uses_api_embed():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        body = json.loads(request.content)
        seen["model"] = body["model"]
        vectors = [[0.1, 0.2] for _ in body["input"]]
        return httpx.Response(200, json={"embeddings": vectors})

    vectors = _client(handler).embed("nomic-embed-text", ["one", "two"])
    assert seen == {"path": "/api/embed", "model": "nomic-embed-text"}
    assert vectors == [[0.1, 0.2], [0.1, 0.2]]


def test_embed_falls_back_to_legacy_endpoint():
    prompts = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            return httpx.Response(404, json={"error": "missing"})
        body = json.loads(request.content)
        prompts.append(body["prompt"])
        return httpx.Response(200, json={"embedding": [0.25, 0.5]})

    vectors = _client(handler).embed("nomic-embed-text", ["alpha", "beta"])
    assert prompts == ["alpha", "beta"]
    assert vectors == [[0.25, 0.5], [0.25, 0.5]]


def test_embed_batches_requests():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(len(body["input"]))
        return httpx.Response(
            200, json={"embeddings": [[1.0] for _ in body["input"]]}
        )

    vectors = _client(handler).embed("nomic-embed-text", [str(i) for i in range(20)])
    assert calls == [16, 4]
    assert len(vectors) == 20


def test_chat_returns_message_text():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0
        assert body["model"] == "llama3.2"
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "  Thursday.  "}}
        )

    assert _client(handler).chat("llama3.2", [{"role": "user", "content": "when?"}]) == "Thursday."


def test_empty_chat_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "  "}})

    with pytest.raises(OllamaError, match="empty"):
        _client(handler).chat("llama3.2", [])


def test_http_error_includes_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="model missing")

    with pytest.raises(OllamaError, match="HTTP 500"):
        _client(handler).embed("missing", ["hello"])


def test_connection_error_mentions_how_to_start():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(OllamaError, match="ollama serve"):
        _client(handler).chat("llama3.2", [{"role": "user", "content": "hi"}])
