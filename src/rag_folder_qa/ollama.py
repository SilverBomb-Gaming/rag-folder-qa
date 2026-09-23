"""Call a local Ollama server over HTTP. No cloud SDK."""

from __future__ import annotations

import httpx

EMBED_BATCH = 16
DEFAULT_TIMEOUT = 180.0


class OllamaError(RuntimeError):
    """Ollama could not be reached or returned an error."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout, connect=5.0)
        self._transport = transport

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        with self._client() as client:
            for start in range(0, len(texts), EMBED_BATCH):
                batch = texts[start : start + EMBED_BATCH]
                vectors.extend(self._embed_batch(client, model, batch))
        return vectors

    def chat(self, model: str, messages: list[dict[str, str]]) -> str:
        try:
            with self._client() as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                )
        except httpx.HTTPError as exc:
            raise _connection_error(self.base_url, exc) from exc
        payload = _json_or_raise(response)
        message = payload.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama /api/chat returned an empty message.")
        return content.strip()

    def _embed_batch(
        self,
        client: httpx.Client,
        model: str,
        texts: list[str],
    ) -> list[list[float]]:
        try:
            response = client.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
            )
        except httpx.HTTPError as exc:
            raise _connection_error(self.base_url, exc) from exc
        if response.status_code == 404:
            return [self._embed_legacy(client, model, text) for text in texts]
        payload = _json_or_raise(response)
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            count = len(embeddings) if isinstance(embeddings, list) else 0
            raise OllamaError(
                f"Ollama /api/embed returned {count} vectors for {len(texts)} inputs."
            )
        return [_as_vector(item) for item in embeddings]

    def _embed_legacy(self, client: httpx.Client, model: str, text: str) -> list[float]:
        try:
            response = client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
        except httpx.HTTPError as exc:
            raise _connection_error(self.base_url, exc) from exc
        payload = _json_or_raise(response)
        return _as_vector(payload.get("embedding"))

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self._timeout, transport=self._transport)


def _as_vector(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise OllamaError("Ollama returned an embedding that is not a vector.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise OllamaError("Ollama returned a non-numeric embedding.") from exc


def _json_or_raise(response: httpx.Response) -> dict:
    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")
        if len(detail) > 400:
            detail = detail[:400] + "…"
        raise OllamaError(f"Ollama returned HTTP {response.status_code}: {detail}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise OllamaError("Ollama returned a non-JSON response.") from exc
    if not isinstance(payload, dict):
        raise OllamaError("Ollama returned a JSON value that is not an object.")
    return payload


def _connection_error(base_url: str, exc: httpx.HTTPError) -> OllamaError:
    return OllamaError(
        f"Could not reach Ollama at {base_url} ({exc.__class__.__name__}: {exc}). "
        "Start it with `ollama serve`, then pull the chat and embed models."
    )
