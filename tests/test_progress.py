"""Tests for bud.lib.progress."""

import pytest
from bud.lib.progress import ProgressTracker


def test_load_returns_empty_when_no_file(tmp_path):
    pt = ProgressTracker(str(tmp_path / "sub" / "progress.json"))
    assert pt.load() == {}


def test_mark_complete_persists(tmp_path):
    path = str(tmp_path / "progress" / "state.json")
    pt = ProgressTracker(path)
    pt.mark_complete("file.json", 0)
    assert pt.is_complete("file.json", 0)


def test_is_complete_returns_false_before_mark(tmp_path):
    pt = ProgressTracker(str(tmp_path / "progress.json"))
    assert pt.is_complete("file.json", 0) is False


def test_mark_failed_records_error(tmp_path):
    pt = ProgressTracker(str(tmp_path / "progress.json"))
    pt.mark_failed("file.json", 1, "some error")
    failed = pt.get_failed()
    assert "file.json" in failed
    assert 1 in failed["file.json"]


def test_mark_complete_clears_failure(tmp_path):
    pt = ProgressTracker(str(tmp_path / "progress.json"))
    pt.mark_failed("file.json", 0, "error")
    pt.mark_complete("file.json", 0)
    failed = pt.get_failed()
    assert "file.json" not in failed


def test_multiple_batches_tracked_independently(tmp_path):
    pt = ProgressTracker(str(tmp_path / "progress.json"))
    pt.mark_complete("file.json", 0)
    pt.mark_complete("file.json", 1)
    assert pt.is_complete("file.json", 0)
    assert pt.is_complete("file.json", 1)
    assert not pt.is_complete("file.json", 2)


def test_get_failed_returns_empty_when_no_failures(tmp_path):
    pt = ProgressTracker(str(tmp_path / "progress.json"))
    pt.mark_complete("file.json", 0)
    assert pt.get_failed() == {}
