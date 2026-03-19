# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bud is a personal conversation RAG pipeline that ingests exported AI conversation data (Claude format `conversations_*.json`), processes it through an LLM-driven chunking pipeline, and builds a FAISS vector index for semantic search. It uses Ollama (default), OpenAI, or Anthropic as LLM/embedding providers.

## Commands

```bash
make dev          # Install with dev dependencies into .venv
make test         # Run full test suite (pytest -v)
make test-fast    # Run tests without verbose output

# Run a single test
.venv/bin/pytest tests/test_chunk.py -v
.venv/bin/pytest tests/test_chunk.py::test_function_name -v

# CLI (after make dev)
bud configure     # Interactive config setup (~/.config/bud/config.yaml)
bud process       # Run full pipeline: parse → chunk → embed → index
bud discover      # Iterative pattern discovery phase
bud query "text"  # Search the vector index
bud status        # Show pipeline status
bud models        # List supported embedding models
```

## Architecture

### Pipeline Stages (`bud/stages/`)

Data flows through stages in this order:

1. **parse** — Reads `conversations_*.json` files, extracts turns (text + thinking blocks), outputs JSONL to `parsed/`
2. **discover** (optional) — Samples parsed conversations, asks LLM to identify structural/geometric patterns, builds a concept map (`discovery_map.json`) that can inform chunking
3. **chunk** — Sends conversations to LLM with a system prompt + schema, gets back structured JSON chunks with dimension tags (geometry, coherence, texture, terrain, motifs) and schema evolution proposals
4. **embed** — Sends chunk text to embedding API, adds vectors to FAISS store. Failed embeddings queue to `embed_queue.jsonl` for retry via `--resume`
5. **blend** — Cross-boundary sampling mode for discovery: slices across conversation files to find patterns invisible to per-conversation sampling

### Key Libraries (`bud/lib/`)

- **model_registry** — Maps embedding model names to their dimension, context window, max_embed_chars, and chunk_max_tokens. These limits propagate to the chunker and embedder automatically. Unknown models get conservative 512-token fallback defaults.
- **store** — FAISS `IndexFlatIP` (cosine similarity via L2-normalized vectors). Metadata stored as sidecar JSONL. Atomic saves with `.tmp` + `os.replace`.
- **embeddings** — Supports Ollama (current `/api/embed` + legacy `/api/embeddings` fallback) and OpenAI-compatible endpoints.
- **llm** — Supports Ollama (`/api/chat`), OpenAI-compatible (Grok), and Anthropic Claude providers.
- **schema_manager** — Manages the tagging schema with dimension values and schema evolution (LLM-proposed new dimension values get promoted after reaching a confidence threshold).
- **prompt_loader** — Loads Markdown prompt templates from `bud/prompts/` with variable substitution.

### Key Design Patterns

- **Model registry drives limits**: When a user configures an embedding model, `resolve_embedding_model()` returns chunk_max_tokens and max_embed_chars that the chunker and embedder both respect. This keeps the pipeline compliant end-to-end.
- **Schema evolution**: The LLM can propose new dimension values during chunking (`schema_proposals`). SchemaManager accumulates candidates and promotes them when they exceed a confidence threshold.
- **Resume/checkpoint**: `ProgressTracker` tracks completed batches. `embed_queue.jsonl` persists failed embeddings. Both support `--resume` for incremental runs.
- **IndexManager** (`bud/stages/index.py`): Central path manager for all output artifacts (index dir, schema, progress, embed queue, discovery map, blend cursor).

### Configuration

Config lives at `~/.config/bud/config.yaml` (YAML). Key sections: `data_dir`, `output_dir`, `llm` (provider/base_url/model), `embeddings` (provider/base_url/model). Pipeline limits are derived at runtime from model_registry, not stored in config.

## Testing

Tests use pytest with pytest-mock. All tests are in `tests/` and mock external services (LLM, embedding API, FAISS). No integration tests require running services.
