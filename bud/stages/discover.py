"""Iterative pattern discovery for optimal chunking schema.

Two-phase pipeline:
  Phase 1 (discover): Sample conversations randomly, ask the LLM to notice
    structural/geometric/topological patterns, accumulate into a concept map.
  Phase 2 (process --with-discovery): Inject the concept map into the chunking
    system prompt so the LLM chunks with informed, archive-aware boundaries.
"""

import json
import os
import random
from pathlib import Path
from typing import Callable, Optional

from bud.lib.errors import LLMError

DISCOVERY_SYSTEM_PROMPT = """You are a structural analyst studying an AI conversation archive.

Your task is to notice the fundamental patterns in the data — not the topics or content,
but the geometry, topology, and structure of how conversations unfold.

Think about:
- GEOMETRIC patterns: How conversations grow, contract, spiral, or branch
- STRUCTURAL patterns: The scaffolding and load-bearing elements of exchanges
- TOPOLOGICAL patterns: What connects to what, what remains invariant, what transforms
- RHYTHMIC patterns: Repetitions, cycles, call-and-response structures
- BOUNDARY patterns: Where natural breaks occur and why
- COHERENCE patterns: What holds a chunk together vs. what pushes it apart
- TENSION patterns: Where a conversation resists splitting vs. where it invites it

Return ONLY valid JSON with this exact structure:
{
  "observations": [
    {
      "pattern_type": "geometric|structural|topological|rhythmic|boundary|coherence|tension",
      "name": "short descriptive name",
      "description": "what this pattern looks like in the data",
      "chunking_implication": "how this should inform chunking decisions",
      "confidence": 0.85
    }
  ],
  "concept_map_updates": {
    "boundary_signals": ["signals that indicate a good chunk boundary"],
    "coherence_anchors": ["things that hold a chunk together as a unit"],
    "chunk_archetypes": ["names for recurring chunk types you identified"],
    "anti_patterns": ["chunking strategies to avoid for this archive"]
  },
  "stability_score": 0.6
}

stability_score: 0.0 = still discovering new patterns, 1.0 = map is stable."""

DISCOVERY_USER_TEMPLATE = """Current concept map (accumulated observations so far):
{concept_map}

Conversation samples to analyze:
{samples}

Analyze these samples and update your understanding. Focus on patterns not yet in the map,
or patterns that confirm and refine existing ones. Return only the JSON object."""


class DiscoveryMap:
    """Manages the accumulated concept map from iterative discovery runs."""

    def __init__(self, path: str):
        self._path = path
        self._data: dict = self._default()

    def _default(self) -> dict:
        return {
            "version": 1,
            "iterations_completed": 0,
            "stability_score": 0.0,
            "boundary_signals": [],
            "coherence_anchors": [],
            "chunk_archetypes": [],
            "anti_patterns": [],
            "observations": [],
        }

    def load(self) -> "DiscoveryMap":
        """Load concept map from disk, or start fresh if missing/corrupt."""
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = self._default()
        return self

    def save(self) -> None:
        """Atomically save concept map to disk."""
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self._path)

    def apply_update(self, llm_response: dict) -> None:
        """Merge one LLM iteration response into the accumulated concept map."""
        updates = llm_response.get("concept_map_updates", {})

        # Merge list fields, deduplicating
        for key in ("boundary_signals", "coherence_anchors", "chunk_archetypes", "anti_patterns"):
            existing = set(self._data.get(key, []))
            new_items = updates.get(key, [])
            self._data[key] = list(existing | set(new_items))

        # Append new observations
        self._data["observations"].extend(llm_response.get("observations", []))

        # Exponential moving average for stability score
        alpha = 0.3
        new_stability = float(llm_response.get("stability_score", 0.0))
        current = float(self._data.get("stability_score", 0.0))
        self._data["stability_score"] = round(alpha * new_stability + (1 - alpha) * current, 4)

        self._data["iterations_completed"] = self._data.get("iterations_completed", 0) + 1
        self._data["version"] = self._data.get("version", 1) + 1

    def to_summary(self) -> str:
        """Compact JSON summary for injection into chunking system prompts."""
        return json.dumps(
            {
                "boundary_signals": self._data.get("boundary_signals", []),
                "coherence_anchors": self._data.get("coherence_anchors", []),
                "chunk_archetypes": self._data.get("chunk_archetypes", []),
                "anti_patterns": self._data.get("anti_patterns", []),
            },
            indent=2,
        )

    @property
    def stability_score(self) -> float:
        return float(self._data.get("stability_score", 0.0))

    @property
    def iterations_completed(self) -> int:
        return int(self._data.get("iterations_completed", 0))

    @property
    def data(self) -> dict:
        return self._data

    def is_empty(self) -> bool:
        return self._data.get("iterations_completed", 0) == 0


def _build_turn_pool(parsed_dir: str) -> list[dict]:
    """Flatten all turns from all conversations into a single ordered pool.

    Each entry carries its parent conversation ID so cross-boundary transitions
    can be detected and annotated in blended samples.
    """
    pool = []
    for jsonl_file in sorted(Path(parsed_dir).glob("*.jsonl")):
        try:
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        conv = json.loads(line)
                        conv_id = conv.get("id", "?")
                        conv_name = conv.get("conversation_name", "")
                        for turn in conv.get("turns", []):
                            pool.append({
                                "conv_id": conv_id,
                                "conv_name": conv_name,
                                "sender": turn.get("sender", "?"),
                                "text": turn.get("text", ""),
                            })
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
    return pool


def blend_archive(
    parsed_dir: str,
    n_slices: int = 6,
    slice_width: int = 8,
    max_chars_per_turn: int = 300,
    rng: "random.Random | None" = None,
) -> str:
    """Build a blended sample by slicing across conversation boundaries.

    Unlike ``_sample_conversations`` (which picks whole conversations),
    this treats every turn from every conversation as a flat pool and cuts
    at arbitrary positions.  Cross-conversation slices expose structural
    patterns — turn syntax, sender alternation, topic-transition markers —
    that are invisible when sampling complete conversational units.

    The resulting text looks like a surrealist transcript: topics jump
    mid-thought, speakers change context, conversations end abruptly and
    begin elsewhere.  This forces the LLM to reason about geometry and
    structure rather than semantic content.

    Args:
        parsed_dir: Directory containing parsed JSONL conversation files.
        n_slices: Number of independent random slices to include.
        slice_width: Number of consecutive turns per slice.
        max_chars_per_turn: Truncation limit for each turn's text.
        rng: Optional seeded ``random.Random`` instance (for reproducibility).

    Returns:
        Formatted multi-slice string ready to send to the LLM, or ``""``
        if the parsed directory is empty.
    """
    if rng is None:
        rng = random.Random()

    pool = _build_turn_pool(parsed_dir)
    if not pool:
        return ""

    parts = []
    for slice_num in range(n_slices):
        max_start = max(0, len(pool) - slice_width)
        start = rng.randint(0, max_start)
        turns = pool[start : start + slice_width]

        lines = [f"--- blend slice {slice_num + 1} (pool offset {start}) ---"]
        prev_conv_id = None
        for turn in turns:
            if prev_conv_id is not None and turn["conv_id"] != prev_conv_id:
                lines.append("[[ CONVERSATION BOUNDARY ]]")
            sender = turn["sender"]
            text = turn["text"][:max_chars_per_turn]
            lines.append(f"[{sender}]: {text}")
            prev_conv_id = turn["conv_id"]

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _sample_conversations(parsed_dir: str, n: int) -> list[dict]:
    """Randomly sample up to n conversations from parsed JSONL files."""
    all_conversations = []
    for jsonl_file in Path(parsed_dir).glob("*.jsonl"):
        try:
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_conversations.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

    if not all_conversations:
        return []

    return random.sample(all_conversations, min(n, len(all_conversations)))


def _format_samples(
    conversations: list[dict],
    max_turns: int = 15,
    max_chars_per_turn: int = 400,
) -> str:
    """Format sampled conversations into a readable block for the LLM."""
    parts = []
    for i, conv in enumerate(conversations):
        turns = conv.get("turns", [])[:max_turns]
        if not turns:
            continue
        name = conv.get("conversation_name", f"Conversation {i + 1}")
        parts.append(f"=== {name} ===")
        for turn in turns:
            sender = turn.get("sender", "?")
            text = turn.get("text", "")[:max_chars_per_turn]
            parts.append(f"[{sender}]: {text}")
        parts.append("")
    return "\n".join(parts)


def run_discovery(
    parsed_dir: str,
    concept_map: DiscoveryMap,
    llm,
    n_samples: int = 5,
    stability_threshold: float = 0.75,
    max_iterations: int = 10,
    on_iteration: Optional[Callable] = None,
    on_sampling: Optional[Callable] = None,
    use_blend: bool = False,
    blend_slices: int = 6,
    blend_width: int = 8,
) -> DiscoveryMap:
    """Run the iterative pattern discovery loop.

    Each iteration either samples whole conversations (default) or builds a
    cross-boundary blended sample (``use_blend=True``), asks the LLM to notice
    structural patterns, and accumulates a concept map until stability is reached
    or ``max_iterations`` is hit.

    Args:
        parsed_dir: Directory containing parsed JSONL conversation files.
        concept_map: DiscoveryMap to accumulate into (pre-loaded to resume).
        llm: LLMClient instance from bud.lib.llm.
        n_samples: Conversations to sample per iteration (ignored when blending).
        stability_threshold: Stop early when stability_score >= this value.
        max_iterations: Hard cap on iterations.
        on_iteration: Optional callback(iteration_num, stability_score, concept_map)
            called after the LLM responds and the map is updated.
        on_sampling: Optional callback(iteration_num, max_iterations) called
            immediately before the LLM call so UIs can show a "waiting" state.
        use_blend: When True, use ``blend_archive`` instead of
            ``_sample_conversations``.  Blending crosses conversation boundaries,
            exposing structural patterns invisible to whole-conversation sampling.
        blend_slices: Number of cross-boundary slices per blended sample.
        blend_width: Turns per slice when blending.

    Returns:
        The updated DiscoveryMap (also saved to disk after each iteration).
    """
    for i in range(max_iterations):
        if use_blend:
            sample_text = blend_archive(parsed_dir, n_slices=blend_slices, slice_width=blend_width)
            if not sample_text:
                break
        else:
            samples = _sample_conversations(parsed_dir, n_samples)
            if not samples:
                break
            sample_text = _format_samples(samples)

        if on_sampling:
            on_sampling(i + 1, max_iterations)

        current_map_json = json.dumps(concept_map.data, indent=2)

        user_prompt = DISCOVERY_USER_TEMPLATE.format(
            concept_map=current_map_json,
            samples=sample_text,
        )

        try:
            response_text = llm.complete(
                system=DISCOVERY_SYSTEM_PROMPT,
                user=user_prompt,
            )
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            response = json.loads(text)
            concept_map.apply_update(response)
            concept_map.save()
        except (LLMError, json.JSONDecodeError, OSError):
            # Skip failed iterations — the loop continues
            pass

        if on_iteration:
            on_iteration(i + 1, concept_map.stability_score, concept_map)

        if concept_map.stability_score >= stability_threshold:
            break

    return concept_map
