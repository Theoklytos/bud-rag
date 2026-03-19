"""Tests for bud.lib.schema_manager."""

import pytest
from bud.lib.schema_manager import SchemaManager, DEFAULT_SCHEMA, REQUIRED_DIMENSIONS


def test_default_schema_has_five_dimensions():
    sm = SchemaManager.__new__(SchemaManager)
    schema = sm.get_default_schema()
    assert len(schema["dimensions"]) == 5
    for dim in REQUIRED_DIMENSIONS:
        assert dim in schema["dimensions"]


def test_load_returns_default_when_no_file(tmp_path):
    sm = SchemaManager(str(tmp_path / "schema.json"))
    schema = sm.load()
    assert schema["version"] == DEFAULT_SCHEMA["version"]
    assert "dimensions" in schema


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "schema.json")
    sm = SchemaManager(path)
    schema = sm.get_default_schema()
    schema["version"] = 42
    sm.save(schema)
    loaded = sm.load()
    assert loaded["version"] == 42


def test_validate_accepts_default_schema(tmp_path):
    sm = SchemaManager(str(tmp_path / "schema.json"))
    schema = sm.get_default_schema()
    assert sm.validate(schema) is True


def test_validate_rejects_missing_dimension(tmp_path):
    sm = SchemaManager(str(tmp_path / "schema.json"))
    schema = sm.get_default_schema()
    del schema["dimensions"]["geometry"]
    assert sm.validate(schema) is False


def test_validate_rejects_non_dict(tmp_path):
    sm = SchemaManager(str(tmp_path / "schema.json"))
    assert sm.validate("not a dict") is False


def test_propose_candidate_adds_to_candidates(tmp_path):
    path = str(tmp_path / "schema.json")
    sm = SchemaManager(path)
    sm.propose_candidate("terrain", "mythic", "example text here")
    schema = sm.load()
    assert "terrain.mythic" in schema["candidates"]
    assert schema["candidates"]["terrain.mythic"]["count"] == 1


def test_propose_candidate_increments_count(tmp_path):
    path = str(tmp_path / "schema.json")
    sm = SchemaManager(path)
    sm.propose_candidate("terrain", "mythic", "example 1")
    sm.propose_candidate("terrain", "mythic", "example 2")
    schema = sm.load()
    assert schema["candidates"]["terrain.mythic"]["count"] == 2


def test_apply_promotions_promotes_after_threshold(tmp_path):
    path = str(tmp_path / "schema.json")
    sm = SchemaManager(path)
    # Set count just at threshold
    schema = sm.get_default_schema()
    schema["candidates"]["terrain.novel_terrain"] = {
        "count": 3,
        "first_seen_batch": 0,
        "examples": ["example"],
    }
    sm.save(schema)
    config = {"pipeline": {"schema_evolution_confidence_threshold": 3}}
    promoted = sm.apply_promotions(config)
    assert "terrain.novel_terrain" in promoted
    updated = sm.load()
    assert "novel_terrain" in updated["dimensions"]["terrain"]


def test_apply_promotions_does_not_promote_below_threshold(tmp_path):
    path = str(tmp_path / "schema.json")
    sm = SchemaManager(path)
    schema = sm.get_default_schema()
    schema["candidates"]["terrain.rare_val"] = {
        "count": 2,
        "first_seen_batch": 0,
        "examples": [],
    }
    sm.save(schema)
    config = {"pipeline": {"schema_evolution_confidence_threshold": 5}}
    promoted = sm.apply_promotions(config)
    assert promoted == []
    # Candidate should still be there
    updated = sm.load()
    assert "terrain.rare_val" in updated["candidates"]
