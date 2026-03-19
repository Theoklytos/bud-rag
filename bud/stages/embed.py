"""Embedding stage for Bud RAG Pipeline."""

import json
import os

from bud.lib.errors import EmbeddingError


MAX_EMBED_CHARS = 8000  # default; overridden per-model via model_registry


def embed_chunks(
    chunks: list[dict],
    embedding_client,
    store,
    queue_path: str,
    on_chunk=None,
    on_error=None,
    max_chars: int = MAX_EMBED_CHARS,
) -> int:
    """Embed chunks and add to vector store.

    Args:
        chunks: List of chunk dicts
        embedding_client: EmbeddingClient instance
        store: VectorStore instance
        queue_path: Path to embedding failure queue file
        on_chunk: Optional callback(done: int, total: int) called after each
            chunk attempt (success or failure) for live progress reporting.
        on_error: Optional callback(chunk: dict, error_msg: str) called for
            each chunk that fails to embed.  Useful for surfacing failures
            in the CLI without breaking the batch loop.
        max_chars: Maximum characters of chunk text sent to the embedding API.
            Derived from the model's context window via model_registry.

    Returns:
        Number of chunks that failed to embed
    """
    failures = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, 1):
        chunk_id = chunk["chunk_id"]
        if store.chunk_id_exists(chunk_id):
            if on_chunk:
                on_chunk(idx, total)
            continue
        try:
            # Truncate text to the model's effective context window
            text = chunk["text"][:max_chars]
            vector = embedding_client.embed(text)
            metadata = {k: v for k, v in chunk.items()}
            metadata["chunk_id"] = chunk_id

            # Ensure store has correct dimension before adding
            if embedding_client.dimension and store.dimension != embedding_client.dimension:
                # Recreate store with correct dimension
                if os.path.exists(store._index_path):
                    os.remove(store._index_path)
                if os.path.exists(store._meta_path):
                    os.remove(store._meta_path)
                store._dim = embedding_client.dimension
                store._create_index()

            store.add([vector], [metadata])
        except EmbeddingError as e:
            failures.append(chunk)
            if on_error:
                on_error(chunk, str(e))
        if on_chunk:
            on_chunk(idx, total)

    if failures:
        write_embed_queue(queue_path, failures, append=True)

    return len(failures)


def load_embed_queue(queue_path: str) -> list[dict]:
    """Load chunks from embedding queue.

    Args:
        queue_path: Path to queue file

    Returns:
        List of chunk dicts from queue
    """
    if not os.path.exists(queue_path):
        return []
    chunks = []
    with open(queue_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def write_embed_queue(queue_path: str, chunks: list[dict], append: bool = False) -> None:
    """Write chunks to embedding queue.

    Args:
        queue_path: Path to queue file
        chunks: List of chunk dicts
        append: If True, append to existing file
    """
    mode = "a" if append else "w"
    with open(queue_path, mode) as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def clear_embed_queue(queue_path: str) -> None:
    """Clear the embedding queue file.

    Args:
        queue_path: Path to queue file
    """
    if os.path.exists(queue_path):
        os.remove(queue_path)
