# Project State

Last updated: 2026-03-20

## Verified Working

| Module | Test File | Key Files | Status |
|---|---|---|---|
| blend | test_blend.py | bud/stages/blend.py | [x] |
| chunk | test_chunk.py | bud/stages/chunk.py | [x] |
| config | test_config.py | bud/config.py | [x] |
| discover | test_discover.py | bud/stages/discover.py | [x] |
| embed | test_embed.py | bud/stages/embed.py, bud/lib/embeddings.py | [x] |
| errors | test_errors.py | bud/lib/errors.py | [x] |
| kaggle_gpu | test_kaggle_gpu.py | bud/lib/kaggle_gpu.py | [x] |
| kaggle_notebook | test_kaggle_notebook.py | bud/kaggle/notebook_source.py, bud/kaggle/kernel_meta.py | [x] |
| model_registry | test_model_registry.py | bud/lib/model_registry.py | [x] |
| parse | test_parse.py | bud/stages/parse.py | [x] |
| progress | test_progress.py | bud/lib/progress.py | [x] |
| prompt_loader | test_prompt_loader.py | bud/lib/prompt_loader.py, bud/prompts/ | [x] |
| schema_manager | test_schema_manager.py | bud/lib/schema_manager.py | [x] |
| store | test_store.py | bud/lib/store.py | [x] |
| mcp_briefing | test_mcp_briefing.py | bud/mcp/briefing.py | [x] |
| mcp_logger | test_mcp_logger.py | bud/mcp/logger.py | [x] |

All 229 tests pass (pytest 2026-03-20).

## Recent Sessions

### 2026-03-20-1 — Kaggle GPU rewrite

Rewrote the Kaggle GPU module to match the actual production architecture:

**Before:** FastAPI + sentence-transformers server with random ngrok URL polling.
Only embedding was offloaded to Kaggle; LLM always ran locally.
`kaggle_gpu` was a separate embedding provider.

**After:** Full Ollama server exposed via static ngrok domain. Both LLM and
embedding route through remote Ollama. `kaggle_gpu_session` context manager
wraps all AI-using CLI commands (process, discover, query) for automatic
kernel start/stop. Shutdown via `kaggle kernels cancel`. Model persistence
via Kaggle datasets (tar `~/.ollama/` to `/kaggle/working/`).

Key changes:
- `bud/kaggle/notebook_source.py` — Ollama + ngrok + model cache template
- `bud/lib/kaggle_gpu.py` — Lifecycle-only manager + context manager (removed
  embed/shutdown methods, removed KaggleEmbeddingClient, removed FallbackEmbeddingClient)
- `bud/lib/embeddings.py` — Removed `kaggle_gpu` provider (uses standard `ollama`)
- `bud/config.py` — New `kaggle` config section, removed `kaggle_gpu` provider
- `bud/cli.py` — All AI commands wrapped with `kaggle_gpu_session`

### 2026-03-19-1 — Initial seed

Project has a complete, working RAG pipeline for ingesting AI conversation exports.
The architecture comprises four CLI stages: `parse`, `discover`, `chunk`, `embed`/`store`,
all orchestrated via `bud process` in `bud/cli.py`.

Key recent work (from git log):
- Added embedding model registry (`model_registry.py`) that drives chunk_max_tokens and
  max_embed_chars for both chunker and embedder automatically at configure/process time.
- Fixed Ollama API compatibility and surfaced embedding failures with a retry queue
  (`embed_queue.jsonl`).
- Added `blend` stage: cross-boundary and progressive cursor-based archive sampling for
  the `bud discover` command.
- Added granular live Rich progress output to `process` and `discover` commands.
- Structured output paths centralized in `IndexManager` (bud/stages/index.py).

## Known Issues

- None currently known.

## Component Interactions

**Pipeline flow:**
```
parse → chunk (LLM + schema_manager) → embed (embeddings + model_registry) → store (FAISS)
```

**Kaggle GPU integration:**
```
bud process/discover/query
  → kaggle_gpu_session(config)
    → KaggleGPUManager.start() [push kernel, poll health at static ngrok domain]
    → pipeline work [all Ollama calls route through ngrok to Kaggle]
    → KaggleGPUManager.stop() [kaggle kernels cancel]
```

**Non-obvious couplings:**

- **model_registry → chunker + embedder**: `resolve_embedding_model()` in
  `bud/lib/model_registry.py` is called at startup in `bud process` and `bud discover`.
  It sets `chunk_max_tokens`, `chunk_min_tokens`, and `max_embed_chars` in the pipeline
  config dict, so both the chunker and embedder respect the same model-specific limits
  without explicit user input.

- **schema_manager accumulates during chunking**: Each chunk returned by `chunk_conversation`
  may carry `schema_proposals`. The CLI feeds these into `schema_mgr.propose_candidate()`.
  After all batches, `apply_promotions()` promotes candidates that exceed the confidence
  threshold, evolving the schema in place.

- **IndexManager centralizes all output paths**: `bud/stages/index.py` is the single
  source of truth for: FAISS index dir, `schema.json`, `progress.json`,
  `embed_queue.jsonl`, `discovery_map.json`, and `blend_cursor.json`.

- **discover → chunk via `--with-discovery`**: Running `bud discover` writes a
  `discovery_map.json`. When `bud process --with-discovery` is used, the CLI loads
  `DiscoveryMap.to_summary()` and passes it as `concept_map_summary` into
  `chunk_conversation`, injecting it into the chunking system prompt.

- **kaggle_gpu_session wraps AI commands**: When `config["kaggle"]` is present with
  `ngrok_static_domain`, the context manager auto-starts/stops the Kaggle kernel.
  Both `llm.base_url` and `embeddings.base_url` point to the same static ngrok
  domain, so all AI calls route through the tunnel transparently.

- **blend cursor persistence**: The `BlendCursor` (bud/stages/blend.py) saves per-file
  turn offsets to `blend_cursor.json` via `IndexManager`. Progressive mode in `bud discover`
  advances this cursor each run so the full archive is covered exhaustively across sessions.
