"""Tests for bud.lib.store (VectorStore)."""

import pytest
from bud.lib.store import VectorStore


DIM = 4  # small dimension for fast tests


def _make_vector(val: float, dim: int = DIM) -> list[float]:
    return [val] * dim


def test_count_zero_on_empty_store(tmp_path):
    store = VectorStore(str(tmp_path / "idx"), dim=DIM)
    assert store.count() == 0


def test_add_increases_count(tmp_path):
    store = VectorStore(str(tmp_path / "idx"), dim=DIM)
    store.add([_make_vector(1.0)], [{"chunk_id": "a"}])
    assert store.count() == 1


def test_add_multiple_vectors(tmp_path):
    store = VectorStore(str(tmp_path / "idx"), dim=DIM)
    store.add(
        [_make_vector(0.1), _make_vector(0.9)],
        [{"chunk_id": "a"}, {"chunk_id": "b"}],
    )
    assert store.count() == 2


def test_search_returns_results(tmp_path):
    store = VectorStore(str(tmp_path / "idx"), dim=DIM)
    store.add([_make_vector(1.0)], [{"chunk_id": "a", "text": "hello"}])
    results = store.search(_make_vector(1.0), k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"
    assert "score" in results[0]


def test_chunk_id_exists(tmp_path):
    store = VectorStore(str(tmp_path / "idx"), dim=DIM)
    store.add([_make_vector(0.5)], [{"chunk_id": "abc123"}])
    assert store.chunk_id_exists("abc123") is True
    assert store.chunk_id_exists("nonexistent") is False


def test_get_by_id(tmp_path):
    store = VectorStore(str(tmp_path / "idx"), dim=DIM)
    store.add([_make_vector(0.5)], [{"chunk_id": "xyz", "extra": "data"}])
    result = store.get_by_id("xyz")
    assert result is not None
    assert result["extra"] == "data"


def test_get_by_id_returns_none_for_missing(tmp_path):
    store = VectorStore(str(tmp_path / "idx"), dim=DIM)
    assert store.get_by_id("missing") is None


def test_save_and_load_roundtrip(tmp_path):
    idx_path = str(tmp_path / "idx")
    store = VectorStore(idx_path, dim=DIM)
    store.add([_make_vector(0.8)], [{"chunk_id": "persisted"}])
    store.save()

    store2 = VectorStore(idx_path, dim=DIM)
    store2.load()
    assert store2.count() == 1
    assert store2.chunk_id_exists("persisted")


def test_save_noop_when_empty(tmp_path):
    store = VectorStore(str(tmp_path / "idx"), dim=DIM)
    store.save()  # should not raise


def test_dimension_inferred_from_first_vector(tmp_path):
    store = VectorStore(str(tmp_path / "idx"))  # default dim=768
    v = [0.1, 0.2, 0.3]  # dim=3
    store.add([v], [{"chunk_id": "x"}])
    assert store.dimension == 3
