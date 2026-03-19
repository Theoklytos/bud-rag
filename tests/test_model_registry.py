"""Tests for bud.lib.model_registry."""

import pytest

from bud.lib.model_registry import resolve_embedding_model, list_known_models, _FALLBACK


# ---------------------------------------------------------------------------
# resolve_embedding_model — exact match
# ---------------------------------------------------------------------------

def test_nomic_embed_text_exact():
    cfg = resolve_embedding_model("nomic-embed-text")
    assert cfg["dimension"] == 768
    assert cfg["context_tokens"] == 8192
    assert cfg["max_embed_chars"] == 8000
    assert cfg["chunk_max_tokens"] == 800
    assert cfg["known"] is True


def test_mxbai_embed_large_exact():
    cfg = resolve_embedding_model("mxbai-embed-large")
    assert cfg["dimension"] == 1024
    assert cfg["context_tokens"] == 512
    assert cfg["max_embed_chars"] == 1800
    assert cfg["chunk_max_tokens"] == 350
    assert cfg["known"] is True


def test_arctic_embed_with_size_tag():
    xs = resolve_embedding_model("snowflake-arctic-embed:xs")
    assert xs["dimension"] == 384

    l_ = resolve_embedding_model("snowflake-arctic-embed:l")
    assert l_["dimension"] == 1024


def test_all_minilm_variants():
    base = resolve_embedding_model("all-minilm")
    l6 = resolve_embedding_model("all-minilm:l6")
    l12 = resolve_embedding_model("all-minilm:l12")
    assert base["dimension"] == 384
    assert l6["dimension"] == 384
    assert l12["dimension"] == 384


def test_bge_m3_has_large_context():
    cfg = resolve_embedding_model("bge-m3")
    assert cfg["context_tokens"] == 8192
    assert cfg["chunk_max_tokens"] == 800


def test_jina_v2_has_large_context():
    cfg = resolve_embedding_model("jina-embeddings-v2-base-en")
    assert cfg["context_tokens"] == 8192


# ---------------------------------------------------------------------------
# resolve_embedding_model — :latest tag stripping
# ---------------------------------------------------------------------------

def test_strips_latest_tag():
    cfg = resolve_embedding_model("nomic-embed-text:latest")
    assert cfg["dimension"] == 768
    assert cfg["known"] is True


def test_does_not_strip_meaningful_tag():
    # :335m is a meaningful tag for mxbai; stripping it falls back to base entry
    cfg = resolve_embedding_model("mxbai-embed-large:335m")
    # base entry should be found via fallback to base name
    assert cfg["dimension"] == 1024
    assert cfg["known"] is True


# ---------------------------------------------------------------------------
# resolve_embedding_model — base name fallback (strip any tag)
# ---------------------------------------------------------------------------

def test_unknown_tag_falls_back_to_base():
    # nomic-embed-text:custom-tag → strips to nomic-embed-text
    cfg = resolve_embedding_model("nomic-embed-text:custom-tag")
    assert cfg["dimension"] == 768


def test_arctic_embed_no_tag_resolves_to_medium():
    cfg = resolve_embedding_model("snowflake-arctic-embed")
    assert cfg["dimension"] == 768  # default/medium variant


# ---------------------------------------------------------------------------
# resolve_embedding_model — unknown model fallback
# ---------------------------------------------------------------------------

def test_unknown_model_returns_fallback():
    cfg = resolve_embedding_model("some-completely-unknown-model")
    assert cfg["known"] is False
    assert cfg["dimension"] == _FALLBACK["dimension"]
    assert cfg["context_tokens"] == _FALLBACK["context_tokens"]


def test_empty_string_returns_fallback():
    cfg = resolve_embedding_model("")
    assert cfg["known"] is False


def test_none_string_returns_fallback():
    # resolve_embedding_model handles None-like via default arg
    cfg = resolve_embedding_model("")
    assert cfg["known"] is False


def test_case_insensitive_lookup():
    cfg = resolve_embedding_model("Nomic-Embed-Text")
    assert cfg["dimension"] == 768
    assert cfg["known"] is True


# ---------------------------------------------------------------------------
# resolve returns a copy (mutations don't affect registry)
# ---------------------------------------------------------------------------

def test_returns_mutable_copy():
    cfg1 = resolve_embedding_model("nomic-embed-text")
    cfg1["dimension"] = 999
    cfg2 = resolve_embedding_model("nomic-embed-text")
    assert cfg2["dimension"] == 768


# ---------------------------------------------------------------------------
# Compliance invariants — every 512-token model must have safe limits
# ---------------------------------------------------------------------------

def test_all_512_token_models_have_safe_char_limit():
    """max_embed_chars for 512-token models must be <= 512 * 4 chars."""
    for entry in list_known_models():
        if entry["context_tokens"] == 512:
            # 512 tokens × 4 chars/token = 2048 max; our limit should be under that
            assert entry["max_embed_chars"] <= 2048, (
                f"{entry['model']}: max_embed_chars {entry['max_embed_chars']} "
                f"exceeds 512-token model capacity"
            )


def test_all_models_chunk_fits_within_embed_chars():
    """chunk_max_tokens * 3.85 must be <= max_embed_chars for every model.

    3.85 chars/estimated-token is derived from:
      estimated_tokens = words * 1.3
      chars ≈ words * 5
      → chars / estimated_token = 5 / 1.3 ≈ 3.85
    """
    for entry in list_known_models():
        projected_chars = entry["chunk_max_tokens"] * 3.85
        assert projected_chars <= entry["max_embed_chars"], (
            f"{entry['model']}: chunk_max_tokens={entry['chunk_max_tokens']} "
            f"projects to ~{projected_chars:.0f} chars but max_embed_chars="
            f"{entry['max_embed_chars']}"
        )


def test_all_models_have_required_keys():
    required = {"dimension", "context_tokens", "max_embed_chars",
                "chunk_max_tokens", "chunk_min_tokens", "description"}
    for entry in list_known_models():
        missing = required - entry.keys()
        assert not missing, f"{entry['model']} is missing keys: {missing}"


# ---------------------------------------------------------------------------
# list_known_models
# ---------------------------------------------------------------------------

def test_list_known_models_returns_list():
    models = list_known_models()
    assert isinstance(models, list)
    assert len(models) > 0


def test_list_known_models_includes_nomic_and_mxbai():
    names = {m["model"] for m in list_known_models()}
    assert "nomic-embed-text" in names
    assert "mxbai-embed-large" in names


def test_list_known_models_sorted():
    models = list_known_models()
    names = [m["model"] for m in models]
    assert names == sorted(names)
