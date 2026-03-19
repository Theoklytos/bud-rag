# Project State Tracker — Design Spec

## Problem

When an AI coding assistant opens a repository, it has no quick way to know what's currently working, what changed recently, or where bugs likely live. It must read scattered files and git history to reconstruct context. This wastes tokens and time, and the model often misses non-obvious component interactions.

## Solution

A living `PROJECT_STATE.md` file in the repo root that serves as a memory-efficient changelog with graceful degradation. Recent work is detailed; older work compresses over time. A checklist of verified-working features tells the model what's solid ground.

Two hooks keep it current automatically:
- **Stop hook** (end of session): appends a detailed session entry
- **SessionStart hook** (start of next session): compacts older entries, updates the verified-working checklist

## Deliverables

1. **`PROJECT_STATE.md`** — seeded with bud-rag's current state
2. **`templates/PROJECT_STATE_TEMPLATE.md`** — generic template for any repo
3. **Stop hook** — agent hook in `.claude/settings.local.json`
4. **SessionStart hook** — agent hook in `.claude/settings.local.json`

## Document Structure

```markdown
# Project State

Last updated: YYYY-MM-DD

## Verified Working
- [x] Feature name — files: `path/a.py`, `path/b.py` — verified by: test suite / manual
- [ ] Incomplete or unverified feature

## Recent Sessions
### YYYY-MM-DD-N — Session title
Detailed: what changed, what was tested, what was verified, what broke.
(N is a sequence number: 1, 2, 3... for multiple sessions on the same date)

### YYYY-MM-DD-N — Older session title
Compressed to 1-2 sentences.

## Known Issues
Active bugs, failing tests, incomplete work.

## Component Interactions
Non-obvious coupling between modules. Updated only when architecture changes.
Example granularity: "embed stage depends on lib/embeddings + lib/model_registry for dimension/limit compliance" — not individual function calls.
```

## Memory Degradation Rules

Applied by the SessionStart compaction hook:

Counting is by position in the file (newest entry = position 1):

| Position | Treatment |
|----------|-----------|
| 1-2 (two newest entries) | Full detail |
| 3-5 | Compressed to 1-2 sentences each |
| 6+ | Dropped entirely |

Target size: the entire file should stay under 200 lines / 5KB. If it exceeds this after compaction, compress more aggressively.

The "Verified Working" checklist persists across compactions — it accumulates rather than degrades. Items are only removed when a feature breaks or is removed.

"Known Issues" entries persist until resolved. "Component Interactions" updates only when architecture changes.

## Hook Design

### Stop Hook (end of session)

- **Event:** `Stop`
- **Type:** `agent`
- **Prompt:** Instructs the agent to:
  1. Read `PROJECT_STATE.md` (if it doesn't exist, create it from the template structure with empty sections)
  2. Run `git diff` to see what changed this session
  3. Run the test suite (`.venv/bin/pytest tests/ -q --tb=no`, capped at 30s timeout) to check what passes. If tests can't run (syntax errors, missing deps), note this instead of blocking.
  4. Append a new dated entry to "Recent Sessions" with: what changed, what was tested, results. Use heading format `### YYYY-MM-DD-N` where N is a sequence number (check existing entries for the same date).
  5. Update "Verified Working" checklist: check items that tests confirm, uncheck items that broke. Map tests to features by file naming convention (`test_<module>.py` covers `<module>`).
  6. Update "Known Issues" if new issues surfaced or old ones were resolved
  7. Write the updated file

### SessionStart Hook (start of next session)

- **Event:** `SessionStart`
- **Type:** `agent`
- **Prompt:** Instructs the agent to:
  1. Read `PROJECT_STATE.md` (if it doesn't exist, create it from the template structure with empty sections)
  2. Apply memory degradation rules to "Recent Sessions" (compress older, drop oldest) using positional counting
  3. Ensure "Verified Working", "Known Issues", and "Component Interactions" are consistent with the session log
  4. Rewrite the file cleanly

### Hook Placement

Both hooks go in `.claude/settings.local.json` (per-user, should be gitignored). Ensure `.claude/settings.local.json` is in `.gitignore`. The document itself (`PROJECT_STATE.md`) is committed to the repo so all collaborators (human or AI) benefit.

## Template

`templates/PROJECT_STATE_TEMPLATE.md` (in the repo root, not inside `.claude/`) contains the same structure with HTML comments as placeholder instructions. No project-specific content. Can be copied to any repo root and renamed to `PROJECT_STATE.md`.

## Seeding the bud-rag Instance

The initial `PROJECT_STATE.md` for bud-rag will be seeded by:
1. Running the test suite to determine what passes
2. Reading git log to summarize recent work
3. Populating "Verified Working" from passing tests and their module coverage
4. Populating "Component Interactions" from the architecture (pipeline stages, lib modules, CLI)
5. Populating "Known Issues" from any current failures or dirty git state

## Scope Boundaries

- This is a passive context document, not a database or structured data store
- No tooling beyond the two hooks — no scripts, no CLI commands
- The hooks use agent type so Claude does the reasoning; no brittle text parsing
- The document is Markdown, human-readable, and human-editable
