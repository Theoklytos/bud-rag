# Project State

Last updated: 2026-03-19

## Verified Working

| Module | Test File | Key Files | Status |
|---|---|---|---|
| blend | test_blend.py | bud/stages/blend.py | [x] |
| chunk | test_chunk.py | bud/stages/chunk.py | [x] |
| config | test_config.py | bud/config.py | [x] |
| discover | test_discover.py | bud/stages/discover.py | [x] |
| embed | test_embed.py | bud/stages/embed.py, bud/lib/embeddings.py | [x] |
| errors | test_errors.py | bud/lib/errors.py | [x] |
| model_registry | test_model_registry.py | bud/lib/model_registry.py | [x] |
| parse | test_parse.py | bud/stages/parse.py | [x] |
| progress | test_progress.py | bud/lib/progress.py | [x] |
| prompt_loader | test_prompt_loader.py | bud/lib/prompt_loader.py, bud/prompts/ | [x] |
| schema_manager | test_schema_manager.py | bud/lib/schema_manager.py | [x] |
| store | test_store.py | bud/lib/store.py | [x] |

All 188 tests pass (pytest 2026-03-19).

## Recent Sessions

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

- **Uncommitted changes**: `bud/lib/embeddings.py` and `bud/stages/embed.py` have local
  modifications not yet committed to main (visible via `git diff`). These changes may
  include in-progress work on embedding behavior.
- No test failures detected at seed time (188/188 pass), but the uncommitted embed changes
  have not been reviewed — they could affect embed-stage behavior in ways not covered by
  current tests.

## Component Interactions

**Pipeline flow:**
```
parse → chunk (LLM + schema_manager) → embed (embeddings + model_registry) → store (FAISS)
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

- **blend cursor persistence**: The `BlendCursor` (bud/stages/blend.py) saves per-file
  turn offsets to `blend_cursor.json` via `IndexManager`. Progressive mode in `bud discover`
  advances this cursor each run so the full archive is covered exhaustively across sessions.
