# Project State Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a living `PROJECT_STATE.md` that gives AI assistants instant context about what works, what changed, and where bugs likely live — kept current by two Claude Code hooks.

**Architecture:** A single Markdown file in the repo root, updated by a Stop hook (appends session entry) and a SessionStart hook (compacts older entries). A generic template is provided separately for use in other repos.

**Tech Stack:** Markdown, Claude Code hooks (agent type), `.claude/settings.local.json`

**Spec:** `docs/superpowers/specs/2026-03-19-project-state-tracker-design.md`

---

## File Structure

| Action | Path | Purpose |
|--------|------|---------|
| Create | `templates/PROJECT_STATE_TEMPLATE.md` | Generic template for any repo |
| Create | `PROJECT_STATE.md` | Seeded bud-rag instance |
| Modify | `.gitignore` | Add `.claude/settings.local.json` |
| Modify | `.claude/settings.local.json` | Add Stop + SessionStart hooks |

---

### Task 1: Create the generic template

**Files:**
- Create: `templates/PROJECT_STATE_TEMPLATE.md`

- [ ] **Step 1: Create the templates directory and template file**

Run: `mkdir -p templates`

Then create the file:

```markdown
# Project State

Last updated: YYYY-MM-DD

## Verified Working
<!-- Checklist of features confirmed working. Format:
- [x] Feature name — files: `path/to/file.py`, `other.py` — verified by: tests / manual
- [ ] Feature not yet verified
Items persist across session compactions. Only remove when a feature is deleted or permanently broken. -->

## Recent Sessions
<!-- Session entries are added by the Stop hook and compacted by the SessionStart hook.
Heading format: ### YYYY-MM-DD-N — Title (N is sequence number for same-date sessions)
- Position 1-2 (newest): full detail
- Position 3-5: compressed to 1-2 sentences
- Position 6+: dropped entirely
Target: entire file under 200 lines / 5KB -->

## Known Issues
<!-- Active bugs, failing tests, incomplete work. Remove when resolved. -->

## Component Interactions
<!-- Non-obvious coupling between modules. Update only when architecture changes.
Granularity: "module A depends on module B for X" — not individual function calls. -->
```

- [ ] **Step 2: Verify the file exists**

Run: `cat templates/PROJECT_STATE_TEMPLATE.md | head -5`
Expected: Shows `# Project State` header

- [ ] **Step 3: Commit**

```bash
git add templates/PROJECT_STATE_TEMPLATE.md
git commit -m "feat: add generic PROJECT_STATE template for any repo"
```

---

### Task 2: Seed PROJECT_STATE.md for bud-rag

**Files:**
- Create: `PROJECT_STATE.md`

This task requires reading the codebase and git history to populate the initial state. The content below is based on the current project state (test suite status, git log, architecture).

- [ ] **Step 1: Read current project state**

Run these to gather context:
```bash
git log --oneline -10
git diff --name-only
.venv/bin/pytest tests/ -q --tb=no 2>&1 || echo "Tests could not run — note failures in Known Issues"
```

Read key files: `bud/cli.py`, `bud/stages/*.py`, `bud/lib/*.py`, `tests/test_*.py`

- [ ] **Step 2: Create the seeded PROJECT_STATE.md**

Write `PROJECT_STATE.md` with all four sections populated:

- **Verified Working**: One entry per module that has a corresponding test file (`test_<module>.py`). Mark `[x]` for modules with tests, `[ ]` for any without. List the key files involved in each feature.
- **Recent Sessions**: One entry summarizing the current state (this is the seed, not a real session — label it as "Initial seed").
- **Known Issues**: Note any currently modified/dirty files (`bud/lib/embeddings.py`, `bud/stages/embed.py` have uncommitted changes), any test failures.
- **Component Interactions**: Document the non-obvious couplings:
  - Pipeline flow: parse → chunk (needs LLM + schema) → embed (needs embeddings + model_registry) → store (FAISS)
  - model_registry drives chunk_max_tokens and max_embed_chars for both chunker and embedder
  - schema_manager accumulates LLM proposals from chunking and promotes them at threshold
  - IndexManager centralizes all output paths (index, schema, progress, embed queue, discovery map, blend cursor)
  - discover stage feeds concept_map_summary into chunk stage's system prompt via `--with-discovery`

- [ ] **Step 3: Verify the file**

Run: `wc -l PROJECT_STATE.md`
Expected: Under 200 lines

- [ ] **Step 4: Commit**

```bash
git add PROJECT_STATE.md
git commit -m "feat: seed PROJECT_STATE.md with current bud-rag state"
```

---

### Task 3: Update .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add settings.local.json to .gitignore**

Append to `.gitignore`:
```
.claude/settings.local.json
```

If the file is already tracked by git, also run: `git rm --cached .claude/settings.local.json`

- [ ] **Step 2: Verify**

Run: `grep 'settings.local' .gitignore`
Expected: `.claude/settings.local.json`

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore .claude/settings.local.json"
```

---

### Task 4: Add Stop hook to settings.local.json

**Files:**
- Modify: `.claude/settings.local.json`

- [ ] **Step 1: Read current settings**

Read `.claude/settings.local.json` to see existing content (currently just `{"outputStyle": "default"}`).

- [ ] **Step 2: Add the Stop hook**

Merge a `hooks.Stop` entry into the existing settings. The agent prompt should instruct the agent to:

1. Read `PROJECT_STATE.md` (if missing, create from template structure with empty sections)
2. Run `git diff` to see what changed
3. Run `.venv/bin/pytest tests/ -q --tb=no` (30s timeout). If tests can't run, note it and continue.
4. Append a new `### YYYY-MM-DD-N — Title` entry to Recent Sessions (check existing entries for same-date sequence number)
5. Update Verified Working checklist (check/uncheck based on test results, map by `test_<module>.py` convention)
6. Update Known Issues if new issues surfaced or old ones resolved
7. Write the updated file

**Important:** Preserve the existing `outputStyle` key when merging. The final file after this task should look like:

```json
{
  "outputStyle": "default",
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "agent",
            "prompt": "<the full prompt text>",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 3: Validate JSON syntax**

Run: `jq '.' .claude/settings.local.json`
Expected: Valid JSON output with both `outputStyle` and `hooks` keys

- [ ] **Step 4: Verify hook structure**

Run: `jq -e '.hooks.Stop[0].hooks[0].type' .claude/settings.local.json`
Expected: `"agent"`

---

### Task 5: Add SessionStart hook to settings.local.json

**Files:**
- Modify: `.claude/settings.local.json`

- [ ] **Step 1: Read current settings**

Read `.claude/settings.local.json` (now has Stop hook from Task 4).

- [ ] **Step 2: Add the SessionStart hook**

Merge a `hooks.SessionStart` entry. The agent prompt should instruct the agent to:

1. Read `PROJECT_STATE.md` (if missing, create from template structure with empty sections)
2. Apply memory degradation: position 1-2 keep full detail, position 3-5 compress to 1-2 sentences, position 6+ drop
3. Ensure Verified Working, Known Issues, and Component Interactions are consistent
4. If file exceeds 200 lines / 5KB, compress more aggressively
5. Rewrite the file cleanly

**Important:** Preserve all existing keys (`outputStyle`, `hooks.Stop`). Add `SessionStart` alongside `Stop` inside the existing `hooks` object. The final file should look like:

```json
{
  "outputStyle": "default",
  "hooks": {
    "Stop": [ ... ],
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "agent",
            "prompt": "<the full prompt text>",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 3: Validate JSON syntax**

Run: `jq '.' .claude/settings.local.json`
Expected: Valid JSON output with `outputStyle`, `hooks.Stop`, and `hooks.SessionStart`

- [ ] **Step 4: Verify both hooks exist**

Run: `jq -e '.hooks | keys' .claude/settings.local.json`
Expected: `["SessionStart", "Stop"]`

- [ ] **Step 5: Commit the spec and plan docs**

```bash
git add docs/superpowers/
git commit -m "docs: add project state tracker spec and implementation plan"
```

Note: `.claude/settings.local.json` is gitignored and should NOT be committed.
