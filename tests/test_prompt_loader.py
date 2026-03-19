"""Tests for bud.lib.prompt_loader."""

import pytest
from bud.lib.prompt_loader import PromptLoader


def test_list_presets_returns_md_filenames(tmp_path):
    (tmp_path / "conversational.md").write_text("hello")
    (tmp_path / "factual.md").write_text("world")
    (tmp_path / "notme.txt").write_text("ignore")
    loader = PromptLoader(str(tmp_path))
    presets = loader.list_presets()
    assert sorted(presets) == ["conversational", "factual"]


def test_list_presets_returns_empty_for_missing_dir():
    loader = PromptLoader("/nonexistent/path/to/prompts")
    assert loader.list_presets() == []


def test_load_substitutes_variables(tmp_path):
    (tmp_path / "test.md").write_text("Hello {{name}}, you have {{count}} items.")
    loader = PromptLoader(str(tmp_path))
    result = loader.load("test", {"name": "Alice", "count": "5"})
    assert result == "Hello Alice, you have 5 items."


def test_load_returns_template_unchanged_when_no_vars(tmp_path):
    (tmp_path / "simple.md").write_text("No variables here.")
    loader = PromptLoader(str(tmp_path))
    result = loader.load("simple", {})
    assert result == "No variables here."


def test_load_raises_for_missing_preset(tmp_path):
    loader = PromptLoader(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent", {})
