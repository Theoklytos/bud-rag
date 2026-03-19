"""Embedding client for Bud RAG Pipeline."""

import requests

from bud.lib.errors import EmbeddingError


class EmbeddingClient:
    """Client for generating embeddings.

    Supports both the current Ollama API (``/api/embed``, ``input=``,
    ``embeddings[0]``) and the legacy format (``/api/embeddings``,
    ``prompt=``, ``embedding``).  The new endpoint is tried first; if it
    returns a non-200 status the client falls back to the legacy endpoint.
    Both endpoints are retried at most once before raising
    :class:`~bud.lib.errors.EmbeddingError`.
    """

    def __init__(self, config: dict):
        self._cfg = config["embeddings"]
        self._timeout = config["llm"].get("timeout_seconds", 60)
        self._dim = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            EmbeddingError: If every attempted API call fails.
        """
        provider = self._cfg.get("provider", "ollama")
        if provider == "openai":
            return self._embed_openai(text)
        return self._embed_ollama(text)

    @property
    def dimension(self) -> int | None:
        """Return the embedding dimension (None until the first successful embed)."""
        return self._dim

    # ------------------------------------------------------------------
    # Provider-specific helpers
    # ------------------------------------------------------------------

    def _embed_ollama(self, text: str) -> list[float]:
        """Embed using Ollama, trying the current API then the legacy API."""
        base = self._cfg["base_url"].rstrip("/")

        # ── Current Ollama API (>=0.1.26): POST /api/embed ───────────────
        # Request body: {"model": "...", "input": "..."}
        # Response:     {"embeddings": [[...]]}
        try:
            resp = requests.post(
                f"{base}/api/embed",
                json={"model": self._cfg["model"], "input": text},
                timeout=self._timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "embeddings" in data and data["embeddings"]:
                    result = data["embeddings"][0]
                    if self._dim is None:
                        self._dim = len(result)
                    return result
        except requests.exceptions.Timeout:
            raise EmbeddingError("Embedding request timed out")
        except requests.exceptions.RequestException:
            pass  # fall through to legacy endpoint

        # ── Legacy Ollama API: POST /api/embeddings ───────────────────────
        # Request body: {"model": "...", "prompt": "..."}
        # Response:     {"embedding": [...]}
        try:
            resp = requests.post(
                f"{base}/api/embeddings",
                json={"model": self._cfg["model"], "prompt": text},
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout:
            raise EmbeddingError("Embedding request timed out")
        except requests.exceptions.RequestException as e:
            raise EmbeddingError(f"Embedding connection error: {e}")

        if resp.status_code != 200:
            raise EmbeddingError(
                f"Embedding API returned {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        if "embedding" not in data:
            raise EmbeddingError(
                f"Unexpected embedding API response (missing 'embedding' key): "
                f"{list(data.keys())}"
            )

        result = data["embedding"]
        if self._dim is None:
            self._dim = len(result)
        return result

    def _embed_openai(self, text: str) -> list[float]:
        """Embed using the OpenAI-compatible embeddings endpoint."""
        base = self._cfg["base_url"].rstrip("/")
        try:
            resp = requests.post(
                f"{base}/v1/embeddings",
                json={"model": self._cfg["model"], "input": text},
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {self._cfg.get('api_key', 'NONE')}"},
            )
        except requests.exceptions.Timeout:
            raise EmbeddingError("Embedding request timed out")
        except requests.exceptions.RequestException as e:
            raise EmbeddingError(f"Embedding connection error: {e}")

        if resp.status_code != 200:
            raise EmbeddingError(
                f"Embedding API returned {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        try:
            result = data["data"][0]["embedding"]
        except (KeyError, IndexError) as exc:
            raise EmbeddingError(
                f"Unexpected OpenAI embedding response: {list(data.keys())}"
            ) from exc

        if self._dim is None:
            self._dim = len(result)
        return result
