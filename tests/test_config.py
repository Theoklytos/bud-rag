"""Tests for bud.config."""

from pathlib import Path
from unittest.mock import patch

import pytest

import bud.config as cfg


VALID_CONFIG = {
    "data_dir": "/tmp/data",
    "output_dir": "/tmp/output",
    "llm": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "llama3",
    },
    "embeddings": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text",
    },
}


def test_load_config_returns_defaults_when_missing(tmp_path, monkeypatch):
    nonexistent = tmp_path / "does_not_exist.yaml"
    monkeypatch.setattr(cfg, "CONFIG_FILE", nonexistent)
    result = cfg.load_config()
    assert result == cfg.DEFAULT_CONFIG.copy()


def test_load_config_reads_existing_file(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    import yaml
    config_file.write_text(yaml.dump(VALID_CONFIG))
    monkeypatch.setattr(cfg, "CONFIG_FILE", config_file)
    result = cfg.load_config()
    assert result["data_dir"] == "/tmp/data"


def test_load_config_returns_defaults_on_yaml_error(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(": invalid: yaml: {{")
    monkeypatch.setattr(cfg, "CONFIG_FILE", config_file)
    result = cfg.load_config()
    assert result == cfg.DEFAULT_CONFIG.copy()


def test_save_and_reload_config(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    monkeypatch.setattr(cfg, "CONFIG_FILE", config_file)
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    cfg.save_config(VALID_CONFIG)
    assert config_file.exists()
    result = cfg.load_config()
    assert result["data_dir"] == "/tmp/data"


def test_validate_config_accepts_valid():
    ok, errors = cfg.validate_config(VALID_CONFIG)
    assert ok is True
    assert errors == []


def test_validate_config_rejects_missing_data_dir():
    bad = {**VALID_CONFIG, "data_dir": None}
    ok, errors = cfg.validate_config(bad)
    assert ok is False
    assert any("data_dir" in e for e in errors)


def test_validate_config_rejects_missing_output_dir():
    bad = {**VALID_CONFIG, "output_dir": None}
    ok, errors = cfg.validate_config(bad)
    assert ok is False
    assert any("output_dir" in e for e in errors)


def test_validate_config_rejects_relative_data_dir():
    bad = {**VALID_CONFIG, "data_dir": "relative/path"}
    ok, errors = cfg.validate_config(bad)
    assert ok is False


def test_validate_config_rejects_invalid_llm_provider():
    bad_llm = {**VALID_CONFIG["llm"], "provider": "unknown"}
    bad = {**VALID_CONFIG, "llm": bad_llm}
    ok, errors = cfg.validate_config(bad)
    assert ok is False
    assert any("llm.provider" in e for e in errors)


def test_validate_config_rejects_invalid_llm_url():
    bad_llm = {**VALID_CONFIG["llm"], "base_url": "not-a-url"}
    bad = {**VALID_CONFIG, "llm": bad_llm}
    ok, errors = cfg.validate_config(bad)
    assert ok is False


def test_validate_config_rejects_invalid_embedding_provider():
    bad_emb = {**VALID_CONFIG["embeddings"], "provider": "anthropic"}
    bad = {**VALID_CONFIG, "embeddings": bad_emb}
    ok, errors = cfg.validate_config(bad)
    assert ok is False
