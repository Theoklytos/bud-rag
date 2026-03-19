"""Registry of known local embedding models and their configuration parameters.

When a user configures bud with a specific embedding model, this registry
provides the correct dimension, context window, max_embed_chars, and
chunk_max_tokens so that the chunker, embedder, and vector store stay
fully compliant with each other.

max_embed_chars derivation:
  Each model has a maximum token context.  At roughly 4 chars/token for
  English text, we set max_embed_chars = context_tokens * 4 * 0.9 to leave
  a safety margin for tokenizer differences and non-ASCII content.
  For large-context models (≥ 8192 tokens) we cap at 8000 chars so that
  API payloads stay manageable; the limit is well inside the model's window.

chunk_max_tokens derivation:
  The chunk.py heuristic estimates tokens as words × 1.3, and the average
  English word is ~5 chars, so estimated_tokens × 3.85 ≈ chars.
  We choose chunk_max_tokens so that chunk_max_tokens × 3.85 ≤ max_embed_chars.
  For 512-token models: 350 × 3.85 ≈ 1348 chars  (< 1800 ✓)
  For 8192-token models: 800 × 3.85 ≈ 3080 chars  (< 8000 ✓)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Conservative fallback used when no registry entry matches.
# Assumes a compact BERT-style model with 512-token context.
# ---------------------------------------------------------------------------
_FALLBACK: dict = {
    "dimension": 768,
    "context_tokens": 512,
    "max_embed_chars": 1800,
    "chunk_max_tokens": 350,
    "chunk_min_tokens": 10,
    "description": "Unknown model — using conservative 512-token defaults",
    "known": False,
}

# ---------------------------------------------------------------------------
# Registry
# Keys are full model names including optional :tag variants.
# Lookup strips :latest automatically but preserves meaningful tags like :335m.
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, dict] = {

    # ── Nomic Embed Text ────────────────────────────────────────────────────
    "nomic-embed-text": {
        "dimension": 768,
        "context_tokens": 8192,
        "max_embed_chars": 8000,
        "chunk_max_tokens": 800,
        "chunk_min_tokens": 10,
        "description": "Nomic Embed Text — 768 dims, 8192-token context",
        "known": True,
    },
    "nomic-embed-text-v1.5": {
        "dimension": 768,
        "context_tokens": 8192,
        "max_embed_chars": 8000,
        "chunk_max_tokens": 800,
        "chunk_min_tokens": 10,
        "description": "Nomic Embed Text v1.5 (matryoshka) — 768 dims, 8192-token context",
        "known": True,
    },

    # ── MixedBread AI ───────────────────────────────────────────────────────
    "mxbai-embed-large": {
        "dimension": 1024,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "MixedBread Embed Large — 1024 dims, 512-token context",
        "known": True,
    },

    # ── Snowflake Arctic Embed ──────────────────────────────────────────────
    # Default tag (no tag / :latest) resolves to the medium variant (768 dims).
    "snowflake-arctic-embed": {
        "dimension": 768,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Snowflake Arctic Embed (default/m) — 768 dims, 512-token context",
        "known": True,
    },
    "snowflake-arctic-embed:xs": {
        "dimension": 384,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Snowflake Arctic Embed XS (22M) — 384 dims, 512-token context",
        "known": True,
    },
    "snowflake-arctic-embed:22m": {
        "dimension": 384,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Snowflake Arctic Embed 22M — 384 dims, 512-token context",
        "known": True,
    },
    "snowflake-arctic-embed:s": {
        "dimension": 384,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Snowflake Arctic Embed S (33M) — 384 dims, 512-token context",
        "known": True,
    },
    "snowflake-arctic-embed:33m": {
        "dimension": 384,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Snowflake Arctic Embed 33M — 384 dims, 512-token context",
        "known": True,
    },
    "snowflake-arctic-embed:m": {
        "dimension": 768,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Snowflake Arctic Embed M (110M) — 768 dims, 512-token context",
        "known": True,
    },
    "snowflake-arctic-embed:110m": {
        "dimension": 768,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Snowflake Arctic Embed 110M — 768 dims, 512-token context",
        "known": True,
    },
    "snowflake-arctic-embed:l": {
        "dimension": 1024,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Snowflake Arctic Embed L (335M) — 1024 dims, 512-token context",
        "known": True,
    },
    "snowflake-arctic-embed:335m": {
        "dimension": 1024,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Snowflake Arctic Embed 335M — 1024 dims, 512-token context",
        "known": True,
    },

    # ── all-MiniLM ──────────────────────────────────────────────────────────
    "all-minilm": {
        "dimension": 384,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "all-MiniLM — 384 dims, 512-token context",
        "known": True,
    },
    "all-minilm:l6": {
        "dimension": 384,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "all-MiniLM L6 — 384 dims, 512-token context",
        "known": True,
    },
    "all-minilm:l12": {
        "dimension": 384,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "all-MiniLM L12 — 384 dims, 512-token context",
        "known": True,
    },

    # ── BGE ─────────────────────────────────────────────────────────────────
    "bge-small-en": {
        "dimension": 384,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "BGE Small EN — 384 dims, 512-token context",
        "known": True,
    },
    "bge-base-en": {
        "dimension": 768,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "BGE Base EN — 768 dims, 512-token context",
        "known": True,
    },
    "bge-large-en": {
        "dimension": 1024,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "BGE Large EN — 1024 dims, 512-token context",
        "known": True,
    },
    "bge-m3": {
        "dimension": 1024,
        "context_tokens": 8192,
        "max_embed_chars": 8000,
        "chunk_max_tokens": 800,
        "chunk_min_tokens": 10,
        "description": "BGE M3 — 1024 dims, 8192-token context",
        "known": True,
    },

    # ── Jina ────────────────────────────────────────────────────────────────
    "jina-embeddings-v2-base-en": {
        "dimension": 768,
        "context_tokens": 8192,
        "max_embed_chars": 8000,
        "chunk_max_tokens": 800,
        "chunk_min_tokens": 10,
        "description": "Jina Embeddings v2 Base EN — 768 dims, 8192-token context",
        "known": True,
    },

    # ── Multilingual ─────────────────────────────────────────────────────────
    "paraphrase-multilingual": {
        "dimension": 768,
        "context_tokens": 512,
        "max_embed_chars": 1800,
        "chunk_max_tokens": 350,
        "chunk_min_tokens": 10,
        "description": "Paraphrase Multilingual — 768 dims, 512-token context",
        "known": True,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_embedding_model(model_name: str) -> dict:
    """Return configuration parameters for the given embedding model name.

    Matching strategy (first match wins):

    1. Exact match including tag  (e.g. ``snowflake-arctic-embed:xs``)
    2. Strip ``:latest`` only, then exact match  (e.g. ``nomic-embed-text:latest``
       → ``nomic-embed-text``)
    3. Base name without any tag  (e.g. ``mxbai-embed-large:335m``
       → ``mxbai-embed-large``)
    4. Conservative fallback defaults (512-token context, 768 dims)

    Returns a **copy** of the matched template so callers can mutate it safely.
    """
    name = (model_name or "").strip().lower()

    # 1. Exact match
    if name in _REGISTRY:
        return dict(_REGISTRY[name])

    # 2. Strip :latest only (not meaningful tags like :335m)
    if name.endswith(":latest"):
        base_no_latest = name[: -len(":latest")]
        if base_no_latest in _REGISTRY:
            return dict(_REGISTRY[base_no_latest])

    # 3. Base name (strip any tag)
    base = name.split(":")[0]
    if base in _REGISTRY:
        return dict(_REGISTRY[base])

    # 4. Fallback
    return dict(_FALLBACK)


def list_known_models() -> list[dict]:
    """Return all registered model entries sorted by name.

    Each entry is a dict with keys: model, dimension, context_tokens,
    max_embed_chars, chunk_max_tokens, description.
    """
    return [
        {"model": key, **{k: v for k, v in cfg.items() if k != "known"}}
        for key, cfg in sorted(_REGISTRY.items())
    ]
