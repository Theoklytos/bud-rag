"""Embedding client for Bud RAG Pipeline."""

import requests

from bud.lib.errors import EmbeddingError


class EmbeddingClient:
    """Client for generating embeddings."""

    def __init__(self, config: dict):
        self._cfg = config["embeddings"]
        self._timeout = config["llm"].get("timeout_seconds", 60)
        self._dim = None

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats

        Raises:
            EmbeddingError: If the API call fails
        """
        url = f"{self._cfg['base_url'].rstrip('/')}/api/embeddings"
        payload = {"model": self._cfg["model"], "prompt": text}
        try:
            resp = requests.post(url, json=payload, timeout=self._timeout)
        except requests.exceptions.Timeout:
            raise EmbeddingError("Embedding request timed out")
        except requests.exceptions.RequestException as e:
            raise EmbeddingError(f"Embedding connection error: {e}")
        if resp.status_code != 200:
            raise EmbeddingError(f"Embedding API returned {resp.status_code}: {resp.text}")
        result = resp.json()["embedding"]
        if self._dim is None:
            self._dim = len(result)
        return result

    @property
    def dimension(self) -> int | None:
        """Return the embedding dimension (None until first embed call)."""
        return self._dim
