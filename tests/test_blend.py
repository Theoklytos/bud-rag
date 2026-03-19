"""Tests for bud/stages/blend.py — BlendCursor and blend_progressive."""

import json
import os

import pytest

from bud.stages.blend import BlendCursor, blend_progressive, _load_turns


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

CONV_A = {
    "id": "conv-a",
    "conversation_name": "Alpha",
    "turns": [
        {"sender": "human",     "text": f"human turn {i}"}
        if i % 2 == 0
        else {"sender": "assistant", "text": f"assistant turn {i}"}
        for i in range(6)
    ],
}

CONV_B = {
    "id": "conv-b",
    "conversation_name": "Beta",
    "turns": [
        {"sender": "human",     "text": f"beta human {i}"}
        if i % 2 == 0
        else {"sender": "assistant", "text": f"beta assistant {i}"}
        for i in range(4)
    ],
}


def _write_jsonl(path, conversations):
    with open(path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


# ---------------------------------------------------------------------------
# BlendCursor tests
# ---------------------------------------------------------------------------


class TestBlendCursor:
    def test_default_is_empty(self, tmp_path):
        c = BlendCursor(str(tmp_path / "cursor.json"))
        assert c.is_empty()
        assert c.data == {}

    def test_get_offset_unknown_file_returns_zero(self, tmp_path):
        c = BlendCursor(str(tmp_path / "cursor.json"))
        assert c.get_offset("nope.jsonl") == 0

    def test_set_and_get_offset(self, tmp_path):
        c = BlendCursor(str(tmp_path / "cursor.json"))
        c.set_offset("file.jsonl", 12)
        assert c.get_offset("file.jsonl") == 12

    def test_set_offset_negative_clamped_to_zero(self, tmp_path):
        c = BlendCursor(str(tmp_path / "cursor.json"))
        c.set_offset("file.jsonl", -5)
        assert c.get_offset("file.jsonl") == 0

    def test_save_and_load_round_trip(self, tmp_path):
        p = str(tmp_path / "cursor.json")
        c = BlendCursor(p)
        c.set_offset("a.jsonl", 10)
        c.set_offset("b.jsonl", 20)
        c.save()

        c2 = BlendCursor(p).load()
        assert c2.get_offset("a.jsonl") == 10
        assert c2.get_offset("b.jsonl") == 20
        assert not c2.is_empty()

    def test_save_atomic_no_tmp_left(self, tmp_path):
        p = str(tmp_path / "cursor.json")
        c = BlendCursor(p)
        c.set_offset("x.jsonl", 5)
        c.save()
        assert not os.path.exists(p + ".tmp")
        assert os.path.exists(p)

    def test_load_missing_file_returns_empty(self, tmp_path):
        c = BlendCursor(str(tmp_path / "nope.json")).load()
        assert c.is_empty()

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        p = tmp_path / "cursor.json"
        p.write_text("not json {{{")
        c = BlendCursor(str(p)).load()
        assert c.is_empty()

    def test_reset_clears_state(self, tmp_path):
        c = BlendCursor(str(tmp_path / "cursor.json"))
        c.set_offset("a.jsonl", 5)
        c.reset()
        assert c.is_empty()

    def test_coverage_fraction(self, tmp_path):
        c = BlendCursor(str(tmp_path / "cursor.json"))
        c.set_offset("a.jsonl", 5)
        cov = c.coverage({"a.jsonl": 10})
        assert abs(cov["a.jsonl"] - 0.5) < 0.01

    def test_coverage_capped_at_one(self, tmp_path):
        c = BlendCursor(str(tmp_path / "cursor.json"))
        c.set_offset("a.jsonl", 100)
        cov = c.coverage({"a.jsonl": 10})
        assert cov["a.jsonl"] == 1.0

    def test_coverage_zero_total_excluded(self, tmp_path):
        c = BlendCursor(str(tmp_path / "cursor.json"))
        cov = c.coverage({"empty.jsonl": 0})
        assert "empty.jsonl" not in cov


# ---------------------------------------------------------------------------
# _load_turns tests
# ---------------------------------------------------------------------------


class TestLoadTurns:
    def test_loads_all_turns_from_file(self, tmp_path):
        p = tmp_path / "c.jsonl"
        _write_jsonl(p, [CONV_A])
        turns = _load_turns(p)
        assert len(turns) == 6

    def test_each_turn_has_required_keys(self, tmp_path):
        p = tmp_path / "c.jsonl"
        _write_jsonl(p, [CONV_A])
        for t in _load_turns(p):
            assert "conv_id" in t
            assert "sender" in t
            assert "text" in t

    def test_conv_id_set_correctly(self, tmp_path):
        p = tmp_path / "c.jsonl"
        _write_jsonl(p, [CONV_A])
        assert all(t["conv_id"] == "conv-a" for t in _load_turns(p))

    def test_multiple_conversations_flattened(self, tmp_path):
        p = tmp_path / "c.jsonl"
        _write_jsonl(p, [CONV_A, CONV_B])
        turns = _load_turns(p)
        assert len(turns) == 10  # 6 + 4

    def test_skips_corrupt_lines(self, tmp_path):
        p = tmp_path / "c.jsonl"
        p.write_text(json.dumps(CONV_A) + "\nnot json\n")
        turns = _load_turns(p)
        assert len(turns) == 6

    def test_missing_file_returns_empty(self, tmp_path):
        turns = _load_turns(tmp_path / "nope.jsonl")
        assert turns == []


# ---------------------------------------------------------------------------
# blend_progressive tests
# ---------------------------------------------------------------------------


class TestBlendProgressive:
    def _setup(self, tmp_path):
        """Two files: file1 has CONV_A (6 turns), file2 has CONV_B (4 turns)."""
        _write_jsonl(tmp_path / "file1.jsonl", [CONV_A])
        _write_jsonl(tmp_path / "file2.jsonl", [CONV_B])
        return str(tmp_path)

    def test_empty_dir_returns_empty(self, tmp_path):
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        text, totals = blend_progressive(str(tmp_path), cursor)
        assert text == ""
        assert totals == {}

    def test_returns_text_for_each_file(self, tmp_path):
        parsed = self._setup(tmp_path)
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        text, totals = blend_progressive(parsed, cursor, slice_width=2)
        assert "file1.jsonl" in text
        assert "file2.jsonl" in text

    def test_file_totals_correct(self, tmp_path):
        parsed = self._setup(tmp_path)
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        _, totals = blend_progressive(parsed, cursor)
        assert totals["file1.jsonl"] == 6
        assert totals["file2.jsonl"] == 4

    def test_cursor_advances_after_call(self, tmp_path):
        parsed = self._setup(tmp_path)
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        blend_progressive(parsed, cursor, slice_width=3)
        assert cursor.get_offset("file1.jsonl") == 3
        assert cursor.get_offset("file2.jsonl") == 3

    def test_second_call_continues_from_cursor(self, tmp_path):
        parsed = self._setup(tmp_path)
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        blend_progressive(parsed, cursor, slice_width=2)  # offsets: file1→2, file2→2
        text2, _ = blend_progressive(parsed, cursor, slice_width=2)
        # file1 should now show "turns 2–3", file2 "turns 2–3"
        assert "turns 2" in text2

    def test_wrap_around_resets_exhausted_file(self, tmp_path):
        """A file with 4 turns and slice_width=4 should wrap on the next call."""
        parsed = self._setup(tmp_path)
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        # Exhaust file2 (4 turns)
        blend_progressive(parsed, cursor, slice_width=4, wrap_around=True)
        assert cursor.get_offset("file2.jsonl") == 0  # wrapped

    def test_no_wrap_skips_exhausted_file(self, tmp_path):
        parsed = self._setup(tmp_path)
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        # Exhaust file2
        blend_progressive(parsed, cursor, slice_width=4, wrap_around=False)
        # file2 offset is 4 (exhausted); next call should skip it
        text2, _ = blend_progressive(parsed, cursor, slice_width=2, wrap_around=False)
        assert "file2.jsonl" not in text2

    def test_conversation_boundary_annotated(self, tmp_path):
        """Slice crossing a conv boundary within a file gets annotated."""
        # file_mixed has two conversations; a slice across both shows boundary
        conv1 = {**CONV_A, "id": "x1"}
        conv2 = {**CONV_B, "id": "x2"}
        _write_jsonl(tmp_path / "mixed.jsonl", [conv1, conv2])
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        # Start near the end of conv1 so the slice crosses into conv2
        cursor.set_offset("mixed.jsonl", 4)  # conv1 has 6 turns; offset 4 → last 2 + first 2 of conv2
        text, _ = blend_progressive(str(tmp_path), cursor, slice_width=4)
        assert "CONVERSATION BOUNDARY" in text

    def test_respects_max_chars_per_turn(self, tmp_path):
        long_conv = {
            **CONV_A,
            "turns": [{"sender": "human", "text": "x" * 500}],
        }
        _write_jsonl(tmp_path / "long.jsonl", [long_conv])
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        text, _ = blend_progressive(str(tmp_path), cursor, slice_width=1, max_chars_per_turn=50)
        assert "x" * 50 in text
        assert "x" * 51 not in text

    def test_slice_header_shows_offset_range(self, tmp_path):
        parsed = self._setup(tmp_path)
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        text, _ = blend_progressive(parsed, cursor, slice_width=2)
        assert "turns 0–1" in text

    def test_slice_width_larger_than_file_does_not_crash(self, tmp_path):
        parsed = self._setup(tmp_path)
        cursor = BlendCursor(str(tmp_path / "cur.json"))
        text, _ = blend_progressive(parsed, cursor, slice_width=1000)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# run_discovery progressive mode integration tests
# ---------------------------------------------------------------------------


class TestRunDiscoveryProgressiveMode:
    def _make_llm(self):
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.complete.return_value = json.dumps({
            "observations": [],
            "concept_map_updates": {
                "boundary_signals": ["topic shift"],
                "coherence_anchors": [],
                "chunk_archetypes": [],
                "anti_patterns": [],
            },
            "stability_score": 0.4,
        })
        return llm

    def _make_parsed_dir(self, tmp_path):
        parsed = tmp_path / "parsed"
        parsed.mkdir()
        _write_jsonl(parsed / "conversations_1.jsonl", [CONV_A])
        _write_jsonl(parsed / "conversations_2.jsonl", [CONV_B])
        return str(parsed)

    def test_progressive_mode_runs(self, tmp_path):
        from bud.stages.discover import DiscoveryMap, run_discovery

        llm = self._make_llm()
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        cursor = BlendCursor(str(tmp_path / "cursor.json"))
        run_discovery(
            parsed_dir=self._make_parsed_dir(tmp_path),
            concept_map=dm,
            llm=llm,
            max_iterations=2,
            use_progressive=True,
            cursor=cursor,
            blend_width=3,
        )
        assert dm.iterations_completed == 2

    def test_progressive_advances_cursor(self, tmp_path):
        from bud.stages.discover import DiscoveryMap, run_discovery

        llm = self._make_llm()
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        cursor = BlendCursor(str(tmp_path / "cursor.json"))
        run_discovery(
            parsed_dir=self._make_parsed_dir(tmp_path),
            concept_map=dm,
            llm=llm,
            max_iterations=1,
            use_progressive=True,
            cursor=cursor,
            blend_width=3,
        )
        # Cursor should have advanced for both files
        assert cursor.get_offset("conversations_1.jsonl") > 0
        assert cursor.get_offset("conversations_2.jsonl") > 0

    def test_progressive_saves_cursor_to_disk(self, tmp_path):
        from bud.stages.discover import DiscoveryMap, run_discovery

        llm = self._make_llm()
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        cursor_path = str(tmp_path / "cursor.json")
        cursor = BlendCursor(cursor_path)
        run_discovery(
            parsed_dir=self._make_parsed_dir(tmp_path),
            concept_map=dm,
            llm=llm,
            max_iterations=1,
            use_progressive=True,
            cursor=cursor,
            blend_width=2,
        )
        assert os.path.exists(cursor_path)

    def test_progressive_raises_without_cursor(self, tmp_path):
        from bud.stages.discover import DiscoveryMap, run_discovery

        llm = self._make_llm()
        dm = DiscoveryMap(str(tmp_path / "dm.json"))
        with pytest.raises(ValueError, match="cursor"):
            run_discovery(
                parsed_dir=self._make_parsed_dir(tmp_path),
                concept_map=dm,
                llm=llm,
                max_iterations=1,
                use_progressive=True,
                cursor=None,
            )
