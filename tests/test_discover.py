"""Tests for bud/stages/discover.py"""

import json
import os

import pytest

from bud.stages.discover import (
    DiscoveryMap,
    _format_samples,
    _sample_conversations,
    run_discovery,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESPONSE = {
    "observations": [
        {
            "pattern_type": "boundary",
            "name": "topic-pivot",
            "description": "Conversation pivots to a new topic",
            "chunking_implication": "Split at pivot point",
            "confidence": 0.9,
        }
    ],
    "concept_map_updates": {
        "boundary_signals": ["topic change", "new question asked"],
        "coherence_anchors": ["shared problem context"],
        "chunk_archetypes": ["exchange", "deep-dive"],
        "anti_patterns": ["splitting mid-reasoning"],
    },
    "stability_score": 0.5,
}

SAMPLE_CONVERSATION = {
    "id": "conv-1",
    "conversation_name": "Test Chat",
    "source_file": "conversations_1.json",
    "turns": [
        {"sender": "human", "text": "Hello, how are you?"},
        {"sender": "assistant", "text": "I'm doing well, thanks!"},
        {"sender": "human", "text": "Can you explain recursion?"},
        {"sender": "assistant", "text": "Sure! Recursion is when a function calls itself."},
    ],
}


def _write_jsonl(path, conversations):
    with open(path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


# ---------------------------------------------------------------------------
# DiscoveryMap tests
# ---------------------------------------------------------------------------


class TestDiscoveryMap:
    def test_default_state(self, tmp_path):
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        assert dm.stability_score == 0.0
        assert dm.iterations_completed == 0
        assert dm.is_empty()

    def test_load_missing_file_returns_default(self, tmp_path):
        dm = DiscoveryMap(str(tmp_path / "nonexistent.json")).load()
        assert dm.is_empty()

    def test_load_corrupt_file_returns_default(self, tmp_path):
        p = tmp_path / "dm.json"
        p.write_text("not json {{{")
        dm = DiscoveryMap(str(p)).load()
        assert dm.is_empty()

    def test_save_and_load_round_trip(self, tmp_path):
        p = str(tmp_path / "dm.json")
        dm = DiscoveryMap(p)
        dm.apply_update(SAMPLE_RESPONSE)
        dm.save()

        dm2 = DiscoveryMap(p).load()
        assert dm2.iterations_completed == 1
        assert "topic change" in dm2.data["boundary_signals"]
        assert not dm2.is_empty()

    def test_save_uses_atomic_rename(self, tmp_path):
        p = str(tmp_path / "dm.json")
        dm = DiscoveryMap(p)
        dm.apply_update(SAMPLE_RESPONSE)
        dm.save()
        assert not os.path.exists(p + ".tmp")
        assert os.path.exists(p)

    def test_apply_update_merges_lists(self, tmp_path):
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        dm.apply_update(SAMPLE_RESPONSE)
        # Apply again with overlapping data + a new item
        second = {
            **SAMPLE_RESPONSE,
            "concept_map_updates": {
                "boundary_signals": ["topic change", "long silence"],
                "coherence_anchors": [],
                "chunk_archetypes": [],
                "anti_patterns": [],
            },
        }
        dm.apply_update(second)
        signals = dm.data["boundary_signals"]
        # Deduplication: "topic change" appears once
        assert signals.count("topic change") == 1
        assert "long silence" in signals

    def test_apply_update_increments_iterations(self, tmp_path):
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        dm.apply_update(SAMPLE_RESPONSE)
        dm.apply_update(SAMPLE_RESPONSE)
        assert dm.iterations_completed == 2

    def test_stability_score_ema(self, tmp_path):
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        # First update with score 1.0 → alpha*1.0 + (1-alpha)*0.0 = 0.3
        dm.apply_update({**SAMPLE_RESPONSE, "stability_score": 1.0})
        assert abs(dm.stability_score - 0.3) < 0.01

    def test_to_summary_returns_valid_json(self, tmp_path):
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        dm.apply_update(SAMPLE_RESPONSE)
        summary = dm.to_summary()
        parsed = json.loads(summary)
        assert "boundary_signals" in parsed
        assert "coherence_anchors" in parsed
        assert "chunk_archetypes" in parsed
        assert "anti_patterns" in parsed

    def test_to_summary_excludes_observations(self, tmp_path):
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        dm.apply_update(SAMPLE_RESPONSE)
        summary_dict = json.loads(dm.to_summary())
        assert "observations" not in summary_dict
        assert "stability_score" not in summary_dict


# ---------------------------------------------------------------------------
# _sample_conversations tests
# ---------------------------------------------------------------------------


class TestSampleConversations:
    def test_empty_dir_returns_empty(self, tmp_path):
        result = _sample_conversations(str(tmp_path), 5)
        assert result == []

    def test_samples_up_to_n(self, tmp_path):
        _write_jsonl(
            tmp_path / "conversations_1.jsonl",
            [SAMPLE_CONVERSATION, {**SAMPLE_CONVERSATION, "id": "conv-2"}],
        )
        result = _sample_conversations(str(tmp_path), 1)
        assert len(result) == 1

    def test_samples_all_when_n_exceeds_available(self, tmp_path):
        _write_jsonl(tmp_path / "conversations_1.jsonl", [SAMPLE_CONVERSATION])
        result = _sample_conversations(str(tmp_path), 100)
        assert len(result) == 1

    def test_skips_corrupt_jsonl_lines(self, tmp_path):
        p = tmp_path / "conversations_1.jsonl"
        p.write_text('{"id": "ok"}\nnot json\n{"id": "ok2"}\n')
        result = _sample_conversations(str(tmp_path), 10)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _format_samples tests
# ---------------------------------------------------------------------------


class TestFormatSamples:
    def test_basic_formatting(self):
        output = _format_samples([SAMPLE_CONVERSATION])
        assert "Test Chat" in output
        assert "[human]:" in output
        assert "[assistant]:" in output

    def test_empty_conversations_skipped(self):
        conv_no_turns = {**SAMPLE_CONVERSATION, "turns": []}
        output = _format_samples([conv_no_turns])
        assert output.strip() == ""

    def test_truncates_long_turns(self):
        long_conv = {
            **SAMPLE_CONVERSATION,
            "turns": [{"sender": "human", "text": "x" * 1000}],
        }
        output = _format_samples([long_conv], max_chars_per_turn=50)
        assert "x" * 50 in output
        assert "x" * 51 not in output

    def test_respects_max_turns(self):
        many_turns_conv = {
            **SAMPLE_CONVERSATION,
            "turns": [{"sender": "human", "text": f"turn {i}"} for i in range(30)],
        }
        output = _format_samples([many_turns_conv], max_turns=5)
        assert "turn 4" in output
        assert "turn 5" not in output

    def test_fallback_name_when_missing(self):
        unnamed = {**SAMPLE_CONVERSATION, "conversation_name": ""}
        unnamed.pop("conversation_name", None)
        output = _format_samples([unnamed])
        assert "Conversation 1" in output


# ---------------------------------------------------------------------------
# run_discovery tests
# ---------------------------------------------------------------------------


class TestRunDiscovery:
    def _make_llm(self, responses=None, raises=None):
        """Create a minimal mock LLM."""
        from unittest.mock import MagicMock

        llm = MagicMock()
        if raises:
            llm.complete.side_effect = raises
        elif responses:
            llm.complete.side_effect = [json.dumps(r) for r in responses]
        else:
            llm.complete.return_value = json.dumps(SAMPLE_RESPONSE)
        return llm

    def _make_parsed_dir(self, tmp_path, conversations=None):
        parsed = tmp_path / "parsed"
        parsed.mkdir()
        convs = conversations or [SAMPLE_CONVERSATION]
        _write_jsonl(parsed / "conversations_1.jsonl", convs)
        return str(parsed)

    def test_runs_and_saves_concept_map(self, tmp_path):
        llm = self._make_llm()
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        result = run_discovery(
            parsed_dir=self._make_parsed_dir(tmp_path),
            concept_map=dm,
            llm=llm,
            n_samples=1,
            max_iterations=2,
        )
        assert result.iterations_completed == 2
        assert os.path.exists(str(tmp_path / "dm.json"))

    def test_stops_at_stability_threshold(self, tmp_path):
        # First response has high stability so it stops after 1 iteration
        high_stable = {**SAMPLE_RESPONSE, "stability_score": 1.0}
        llm = self._make_llm(responses=[high_stable] * 10)
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        run_discovery(
            parsed_dir=self._make_parsed_dir(tmp_path),
            concept_map=dm,
            llm=llm,
            n_samples=1,
            stability_threshold=0.25,
            max_iterations=10,
        )
        # With alpha=0.3 and score=1.0: after 1 iter stability=0.3 >= 0.25 → stops
        assert dm.iterations_completed == 1

    def test_skips_failed_llm_iterations(self, tmp_path):
        from bud.lib.errors import LLMError

        llm = self._make_llm(raises=LLMError("boom"))
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        run_discovery(
            parsed_dir=self._make_parsed_dir(tmp_path),
            concept_map=dm,
            llm=llm,
            n_samples=1,
            max_iterations=3,
        )
        # All iterations failed → map stays empty but no exception raised
        assert dm.iterations_completed == 0

    def test_skips_invalid_json_response(self, tmp_path):
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.complete.return_value = "not valid json at all"
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        run_discovery(
            parsed_dir=self._make_parsed_dir(tmp_path),
            concept_map=dm,
            llm=llm,
            n_samples=1,
            max_iterations=2,
        )
        assert dm.iterations_completed == 0

    def test_on_iteration_callback_called(self, tmp_path):
        llm = self._make_llm()
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        calls = []
        run_discovery(
            parsed_dir=self._make_parsed_dir(tmp_path),
            concept_map=dm,
            llm=llm,
            n_samples=1,
            max_iterations=3,
            on_iteration=lambda num, score, cmap: calls.append((num, score)),
        )
        assert len(calls) == 3
        assert calls[0][0] == 1
        assert calls[2][0] == 3

    def test_empty_parsed_dir_returns_immediately(self, tmp_path):
        empty_parsed = tmp_path / "parsed"
        empty_parsed.mkdir()
        llm = self._make_llm()
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        run_discovery(
            parsed_dir=str(empty_parsed),
            concept_map=dm,
            llm=llm,
            max_iterations=5,
        )
        assert dm.iterations_completed == 0
        llm.complete.assert_not_called()

    def test_strips_markdown_fences_from_response(self, tmp_path):
        from unittest.mock import MagicMock

        fenced = "```json\n" + json.dumps(SAMPLE_RESPONSE) + "\n```"
        llm = MagicMock()
        llm.complete.return_value = fenced
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        run_discovery(
            parsed_dir=self._make_parsed_dir(tmp_path),
            concept_map=dm,
            llm=llm,
            n_samples=1,
            max_iterations=1,
        )
        assert dm.iterations_completed == 1
