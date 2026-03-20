"""Tests for Kaggle notebook source and kernel metadata."""

import json
import os

from bud.kaggle.notebook_source import render_notebook_source, NOTEBOOK_SOURCE
from bud.kaggle.kernel_meta import generate_kernel_metadata, write_kernel_package


def test_render_notebook_source():
    """Verify placeholders are substituted in the rendered source."""
    source = render_notebook_source(
        llm_model="gemma3:4b",
        embed_model="nomic-embed-text:v1.5",
        ngrok_static_domain="example.ngrok-free.dev",
        model_cache_slug="",
    )
    assert "gemma3:4b" in source
    assert "nomic-embed-text:v1.5" in source
    assert "example.ngrok-free.dev" in source
    assert "{llm_model}" not in source
    assert "{embed_model}" not in source
    assert "{ngrok_static_domain}" not in source


def test_render_contains_ollama_install():
    """Verify the notebook installs Ollama."""
    source = render_notebook_source(
        llm_model="m", embed_model="e", ngrok_static_domain="d",
    )
    assert "ollama" in source.lower()
    assert "ollama serve" in source or "ollama_proc" in source


def test_render_contains_ngrok_static_domain():
    """Verify static domain is used for ngrok tunnel."""
    source = render_notebook_source(
        llm_model="m", embed_model="e",
        ngrok_static_domain="my-domain.ngrok-free.dev",
    )
    assert "my-domain.ngrok-free.dev" in source


def test_render_contains_model_cache_logic():
    """Verify model cache restore/save logic is present."""
    source = render_notebook_source(
        llm_model="m", embed_model="e", ngrok_static_domain="d",
        model_cache_slug="user/bud-ollama-cache",
    )
    assert "user/bud-ollama-cache" in source
    assert "/kaggle/input/" in source
    assert "/kaggle/working/" in source


def test_render_uses_kaggle_secrets():
    """Verify ngrok token comes from Kaggle Secrets, not config."""
    source = render_notebook_source(
        llm_model="m", embed_model="e", ngrok_static_domain="d",
    )
    assert "UserSecretsClient" in source
    assert "NGROK_TOKEN" in source


def test_render_no_model_cache_when_empty():
    """Verify no dataset restore when model_cache_slug is empty."""
    source = render_notebook_source(
        llm_model="m", embed_model="e", ngrok_static_domain="d",
        model_cache_slug="",
    )
    assert "/kaggle/input/" not in source or "if MODEL_CACHE_SLUG" in source


# Keep existing kernel_meta tests below
def test_generate_kernel_metadata():
    """Verify kernel metadata structure."""
    meta = generate_kernel_metadata(
        username="testuser", kernel_slug="my-kernel",
    )
    assert meta["id"] == "testuser/my-kernel"
    assert meta["code_file"] == "script.py"
    assert meta["enable_gpu"] is True
    assert meta["enable_internet"] is True


def test_generate_kernel_metadata_defaults():
    """Verify default slug and title."""
    meta = generate_kernel_metadata(username="alice")
    assert meta["id"] == "alice/bud-embedding-server"


def test_write_kernel_package(tmp_path):
    """Verify files are written to the output directory."""
    source = "print('hello')"
    metadata = generate_kernel_metadata(username="bob", kernel_slug="test-k")
    write_kernel_package(str(tmp_path), source, metadata)

    meta_path = tmp_path / "kernel-metadata.json"
    script_path = tmp_path / "script.py"
    assert meta_path.exists()
    assert script_path.exists()

    with open(meta_path) as f:
        loaded_meta = json.load(f)
    assert loaded_meta["id"] == "bob/test-k"

    with open(script_path) as f:
        loaded_source = f.read()
    assert loaded_source == source
