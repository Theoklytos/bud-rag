"""Tests for bud.stages.embed."""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from bud.lib.errors import EmbeddingError
from bud.stages.embed import (
    embed_chunks,
    load_embed_queue,
    write_embed_queue,
    clear_embed_queue,
)


def _make_chunk(chunk_id: str, text: str = "some text") -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "conversation_id": "conv-1",
        "source_file": "file.json",
    }


def _make_embedding_client(vector=None, dimension=4):
    client = MagicMock()
    client.embed.return_value = vector or [0.1, 0.2, 0.3, 0.4]
    client.dimension = dimension
    return client


def _make_store(existing_ids=None):
    store = MagicMock()
    existing = set(existing_ids or [])
    store.chunk_id_exists.side_effect = lambda cid: cid in existing
    store.dimension = 4
    store._index_path = "/fake/path.faiss"
    store._meta_path = "/fake/path_metadata.jsonl"
    return store


def test_embed_chunks_calls_embed_for_each_chunk(tmp_path):
    chunks = [_make_chunk("a"), _make_chunk("b")]
    client = _make_embedding_client()
    store = _make_store()
    queue = str(tmp_path / "queue.jsonl")

    embed_chunks(chunks, client, store, queue)

    assert client.embed.call_count == 2
    assert store.add.call_count == 2


def test_embed_chunks_skips_existing_chunk_ids(tmp_path):
    chunks = [_make_chunk("existing"), _make_chunk("new")]
    client = _make_embedding_client()
    store = _make_store(existing_ids=["existing"])
    queue = str(tmp_path / "queue.jsonl")

    embed_chunks(chunks, client, store, queue)

    assert client.embed.call_count == 1


def test_failed_chunks_written_to_queue(tmp_path):
    chunks = [_make_chunk("a")]
    client = _make_embedding_client()
    client.embed.side_effect = EmbeddingError("API failure")
    store = _make_store()
    queue = str(tmp_path / "queue.jsonl")

    failures = embed_chunks(chunks, client, store, queue)

    assert failures == 1
    queued = load_embed_queue(queue)
    assert len(queued) == 1
    assert queued[0]["chunk_id"] == "a"


def test_embed_chunks_returns_zero_on_success(tmp_path):
    chunks = [_make_chunk("a")]
    client = _make_embedding_client()
    store = _make_store()
    queue = str(tmp_path / "queue.jsonl")

    failures = embed_chunks(chunks, client, store, queue)
    assert failures == 0


def test_load_embed_queue_returns_empty_when_no_file(tmp_path):
    result = load_embed_queue(str(tmp_path / "nonexistent.jsonl"))
    assert result == []


def test_write_and_load_embed_queue(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    chunks = [_make_chunk("x"), _make_chunk("y")]
    write_embed_queue(path, chunks)
    loaded = load_embed_queue(path)
    assert len(loaded) == 2
    assert loaded[0]["chunk_id"] == "x"
    assert loaded[1]["chunk_id"] == "y"


def test_write_embed_queue_append(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_embed_queue(path, [_make_chunk("a")])
    write_embed_queue(path, [_make_chunk("b")], append=True)
    loaded = load_embed_queue(path)
    assert len(loaded) == 2


def test_clear_embed_queue_removes_file(tmp_path):
    path = str(tmp_path / "queue.jsonl")
    write_embed_queue(path, [_make_chunk("a")])
    clear_embed_queue(path)
    assert not (tmp_path / "queue.jsonl").exists()


def test_clear_embed_queue_noop_when_no_file(tmp_path):
    clear_embed_queue(str(tmp_path / "nofile.jsonl"))  # should not raise


# ---------------------------------------------------------------------------
# on_chunk callback tests
# ---------------------------------------------------------------------------


def test_on_chunk_called_for_each_chunk(tmp_path):
    client = _make_embedding_client()
    store = _make_store()
    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    calls = []
    embed_chunks(chunks, client, store, str(tmp_path / "q.jsonl"), on_chunk=lambda d, t: calls.append((d, t)))
    assert len(calls) == 3
    assert calls[0] == (1, 3)
    assert calls[2] == (3, 3)


def test_on_chunk_called_for_skipped_chunks(tmp_path):
    """on_chunk fires even when a chunk already exists in the store."""
    client = _make_embedding_client()
    store = _make_store(existing_ids=["dup"])
    chunks = [_make_chunk("dup")]
    calls = []
    embed_chunks(chunks, client, store, str(tmp_path / "q.jsonl"), on_chunk=lambda d, t: calls.append((d, t)))
    assert len(calls) == 1
    assert calls[0] == (1, 1)


def test_on_chunk_called_on_failure(tmp_path):
    """on_chunk fires even when embedding raises EmbeddingError."""
    client = _make_embedding_client()
    client.embed.side_effect = EmbeddingError("boom")
    store = _make_store()
    chunks = [_make_chunk("fail")]
    calls = []
    embed_chunks(chunks, client, store, str(tmp_path / "q.jsonl"), on_chunk=lambda d, t: calls.append((d, t)))
    assert len(calls) == 1


def test_on_chunk_none_does_not_raise(tmp_path):
    client = _make_embedding_client()
    store = _make_store()
    chunks = [_make_chunk("ok")]
    # no on_chunk — should complete without error
    result = embed_chunks(chunks, client, store, str(tmp_path / "q.jsonl"))
    assert result == 0
