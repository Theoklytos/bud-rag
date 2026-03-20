# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bud is a personal conversation RAG pipeline that ingests exported AI conversation data (Claude format `conversations_*.json`), processes it through an LLM-driven chunking pipeline, and builds a FAISS vector index for semantic search. It uses Ollama (default), OpenAI, Anthropic, or Kaggle GPU as LLM/embedding providers. An MCP server exposes the index as structural memory tools.

## Prerequisites

- Python 3.10+
- Ollama running locally (default `http://localhost:11434`) for LLM and embedding

## Commands

```bash
# Setup
make install      # Bootstrap .venv with runtime deps only
make dev          # Install with dev dependencies (pytest, pytest-mock)
make clean        # Remove __pycache__, *.pyc, build artifacts
make distclean    # clean + remove .venv entirely

# Testing
make test         # Run full test suite (pytest -v)
make test-fast    # Run tests without verbose output
.venv/bin/pytest tests/test_chunk.py -v                    # Single file
.venv/bin/pytest tests/test_chunk.py::test_function_name -v # Single test

# CLI (after make dev)
bud configure     # Interactive config setup (~/.config/bud/config.yaml)
bud parse         # Parse raw conversations → parsed/*.jsonl (standalone)
bud process       # Run full pipeline: parse → chunk → embed → index
bud discover      # Iterative pattern discovery phase
bud query "text"  # Search the vector index
bud status        # Show pipeline status
bud models        # List supported embedding models
bud gpu           # Manage Kaggle GPU kernel (--start, --stop, --status)
bud serve         # Start MCP server (stdio transport)
```

Optional dependency groups: `pip install -e ".[dev]"`, `.[kaggle]`, `.[mcp]`

## Architecture

### Pipeline Stages (`bud/stages/`)

Data flows through stages in this order:

1. **parse** — Reads `conversations_*.json` files, extracts turns (text + thinking blocks), outputs JSONL to `parsed/`. Can be run standalone via `bud parse` or as part of `bud process`.
2. **discover** (optional) — Samples parsed conversations, asks LLM to identify structural/geometric patterns, builds a concept map (`discovery_map.json`) that can inform chunking. Three sampling modes: whole-conversation, random blend, progressive cursor-based blend.
3. **chunk** — Sends conversations to LLM with a system prompt + schema, gets back structured JSON chunks with dimension tags (geometry, coherence, texture, terrain, motifs) and schema evolution proposals.
4. **embed** — Sends chunk text to embedding API, adds vectors to FAISS store. Failed embeddings queue to `embed_queue.jsonl` for retry via `--resume`.
5. **blend** — Cross-boundary sampling mode for discovery: slices across conversation files to find patterns invisible to per-conversation sampling.

### Key Libraries (`bud/lib/`)

- **model_registry** — Maps embedding model names to their dimension, context window, max_embed_chars, and chunk_max_tokens. These limits propagate to the chunker and embedder automatically. Unknown models get conservative 512-token fallback defaults.
- **store** — FAISS `IndexFlatIP` (cosine similarity via L2-normalized vectors). Metadata stored as sidecar JSONL. Atomic saves with `.tmp` + `os.replace`.
- **embeddings** — Supports Ollama (current `/api/embed` + legacy `/api/embeddings` fallback), OpenAI-compatible, and Kaggle GPU endpoints. Provider selection via `config["embeddings"]["provider"]`.
- **llm** — Supports Ollama (`/api/chat`), OpenAI-compatible (Grok), and Anthropic Claude providers.
- **kaggle_gpu** — Kaggle GPU lifecycle manager. `KaggleGPUManager` handles kernel start/stop (push notebook via `kaggle kernels push`, poll static ngrok domain for health, cancel via `kaggle kernels cancel`). `kaggle_gpu_session(config)` context manager wraps AI-using CLI commands for automatic start/stop. No-op when no `kaggle` config section is present.
- **schema_manager** — Manages the tagging schema with dimension values and schema evolution (LLM-proposed new dimension values get promoted after reaching a confidence threshold).
- **prompt_loader** — Loads Markdown prompt templates from `bud/prompts/` with `{{variable}}` substitution.

### Kaggle GPU Module (`bud/kaggle/`)

Packages an Ollama server as a Kaggle kernel exposed via a static ngrok domain:
- **notebook_source** — Template rendered with `{llm_model}`, `{embed_model}`, `{ngrok_static_domain}`, `{model_cache_slug}`. Installs Ollama, pulls models, sets up ngrok tunnel. Supports model persistence via Kaggle dataset cache (tar/restore `~/.ollama/`).
- **kernel_meta** — Generates `kernel-metadata.json` + `script.py` for the `kaggle kernels push` CLI. Supports optional `dataset_sources` for model cache datasets.

Both LLM and embedding calls route through the remote Ollama via ngrok. The `kaggle_gpu_session` context manager in `bud/lib/kaggle_gpu.py` wraps `process`, `discover`, and `query` commands for automatic kernel lifecycle management.

### MCP Server (`bud/mcp/`)

FastMCP server (stdio transport) exposing three tools:
- **bud_query_structural_memory** — Embeds a query, retrieves top-k chunks, optionally generates an interpretive LLM briefing using the discovery map, logs the interaction.
- **bud_get_active_patterns** — Returns the discovery map, optionally filtered by relevance via LLM.
- **bud_log_outcome** — Records delayed outcome feedback linked to a previous interaction.

Supporting modules:
- **briefing** — Composes structural briefings via `PromptLoader` templates in `bud/mcp/prompts/`.
- **logger** — Append-only JSONL interaction log with outcome linking.

Server lifecycle loads shared resources (VectorStore, DiscoveryMap, LLMClient, EmbeddingClient) once via FastMCP `lifespan` context manager.

### Key Design Patterns

- **Model registry drives limits**: When a user configures an embedding model, `resolve_embedding_model()` returns chunk_max_tokens and max_embed_chars that the chunker and embedder both respect.
- **Schema evolution**: The LLM can propose new dimension values during chunking (`schema_proposals`). SchemaManager accumulates candidates and promotes them when they exceed a confidence threshold.
- **Resume/checkpoint**: `ProgressTracker` tracks completed batches. `embed_queue.jsonl` persists failed embeddings. Both support `--resume` for incremental runs.
- **IndexManager** (`bud/stages/index.py`): Central path manager for all output artifacts (index dir, schema, progress, embed queue, discovery map, blend cursor, interaction log, briefing cache).
- **Embedding client interface**: All embedding providers implement `embed(text: str) -> list[float]` and a `dimension` property.
- **kaggle_gpu_session wraps AI commands**: When `config["kaggle"]["ngrok_static_domain"]` is set, the `kaggle_gpu_session` context manager auto-starts the Kaggle kernel before AI work and cancels it on exit. Both `llm.base_url` and `embeddings.base_url` point to the static ngrok domain, routing all AI calls through the tunnel transparently.
- **Config env expansion**: String values containing `${VAR_NAME}` in `config.yaml` are expanded from `os.environ` at load time. Useful for `ngrok_token: ${NGROK_TOKEN}`.

### Configuration

Config lives at `~/.config/bud/config.yaml` (YAML). Key sections: `data_dir`, `output_dir`, `llm` (provider/base_url/model), `embeddings` (provider/base_url/model). Valid embedding providers: `ollama`, `openai`. Pipeline limits are derived at runtime from model_registry, not stored in config.

Optional `kaggle` section for remote GPU: `ngrok_static_domain` (required), `username`, `kernel_slug`, `poll_timeout_seconds`, `model_cache_dataset`. When present, `llm.base_url` and `embeddings.base_url` should point to `https://<ngrok_static_domain>`.

## Implementation Specifications

Two detailed architectural specs exist for the newer subsystems:

- **BUD_KAGGLE_GPU_SPEC.md** — Full spec for the Kaggle GPU embedding manager (notebook template, lifecycle, CLI integration, testing). Read before modifying `bud/kaggle/` or `bud/lib/kaggle_gpu.py`.
- **BUD_MCP_SPEC.md** — Full spec for the MCP server (tool definitions, briefing composer, interaction logging, testing). Read before modifying `bud/mcp/`.

## Testing

Tests use pytest with pytest-mock. All tests are in `tests/` and mock external services (LLM, embedding API, FAISS, Kaggle CLI, subprocess). No integration tests require running services.

## Project Status

See **PROJECT_STATE.md** for current test pass rates, known issues, uncommitted changes, and component interaction map.
