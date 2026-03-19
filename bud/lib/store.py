"""Vector store using FAISS for Bud RAG Pipeline."""

import json
import os
import shutil

import faiss
import numpy as np

from bud.lib.errors import StoreError


class VectorStore:
    """FAISS-based vector store for embeddings."""

    def __init__(self, index_dir: str, dim: int = 768):
        self._dir = index_dir
        self._dim = dim
        self._index_path = f"{index_dir}.faiss"
        self._meta_path = f"{index_dir}_metadata.jsonl"
        self._bak_path = f"{index_dir}.bak.faiss"
        self._bak_meta = f"{index_dir}.bak_metadata.jsonl"
        self._index = None
        self._metadata: list[dict] = []

    def _create_index(self) -> None:
        """Create a new FAISS index with the configured dimension."""
        self._index = faiss.IndexFlatIP(self._dim)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dim

    def add(self, vectors: list[list[float]], metadata: list[dict]) -> None:
        """Add vectors to the store.

        Args:
            vectors: List of embedding vectors
            metadata: List of metadata dicts (one per vector)
        """
        if self._index is None:
            # Infer dimension from first vector and create index
            self._dim = len(vectors[0])
            self._create_index()
        arr = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(arr)
        self._index.add(arr)
        self._metadata.extend(metadata)

    def search(self, query: list[float], k: int = 5) -> list[dict]:
        """Search for similar vectors.

        Args:
            query: Query embedding vector
            k: Number of results to return

        Returns:
            List of matching metadata dicts (with 'score' added from cosine similarity)
        """
        arr = np.array([query], dtype=np.float32)
        faiss.normalize_L2(arr)
        distances, indices = self._index.search(arr, min(k, max(1, self._index.ntotal)))
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self._metadata):
                meta = dict(self._metadata[idx])
                meta['score'] = float(dist)  # Cosine similarity score
                results.append(meta)
        return results

    def get_by_id(self, chunk_id: str) -> dict | None:
        """Get metadata by chunk_id.

        Args:
            chunk_id: The chunk identifier to look up

        Returns:
            Metadata dict or None if not found
        """
        for m in self._metadata:
            if m.get("chunk_id") == chunk_id:
                return m
        return None

    def count(self) -> int:
        """Return the number of vectors in the store."""
        return self._index.ntotal if self._index else 0

    def chunk_id_exists(self, chunk_id: str) -> bool:
        """Check if a chunk_id exists in the store.

        Args:
            chunk_id: The chunk identifier to check

        Returns:
            True if the chunk exists, False otherwise
        """
        return self.get_by_id(chunk_id) is not None

    def save(self) -> None:
        """Save the index and metadata to disk."""
        if self._index is None or self._index.ntotal == 0:
            return
        index_dir = os.path.dirname(self._index_path)
        if index_dir:
            os.makedirs(index_dir, exist_ok=True)
        if os.path.exists(self._index_path):
            shutil.copy2(self._index_path, self._bak_path)
        if os.path.exists(self._meta_path):
            shutil.copy2(self._meta_path, self._bak_meta)
        tmp_index = self._index_path + ".tmp"
        tmp_meta = self._meta_path + ".tmp"
        faiss.write_index(self._index, tmp_index)
        with open(tmp_meta, "w") as f:
            for m in self._metadata:
                f.write(json.dumps(m) + "\n")
        os.replace(tmp_index, self._index_path)
        os.replace(tmp_meta, self._meta_path)

    def load(self) -> None:
        """Load the index and metadata from disk."""
        if not os.path.exists(self._index_path):
            return
        try:
            self._index = faiss.read_index(self._index_path)
            self._dim = self._index.d
        except Exception:
            os.remove(self._index_path)
            if os.path.exists(self._meta_path):
                os.remove(self._meta_path)
            return
        self._metadata = []
        if os.path.exists(self._meta_path):
            with open(self._meta_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._metadata.append(json.loads(line))
