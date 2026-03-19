"""Progressive cross-file blending for archive-aware pattern discovery.

BlendCursor tracks a per-file turn-level cursor so that successive calls to
blend_progressive consume the archive exhaustively rather than sampling at
random.  Each invocation takes one contiguous slice from every parsed JSONL
file, advances each file's cursor by slice_width turns, and wraps back to the
beginning when a file is exhausted.  Slices from all files are interleaved
into one blended text block with [[ CONVERSATION BOUNDARY ]] annotations at
every cross-conversation seam.

Typical lifecycle
-----------------
cursor = BlendCursor(cursor_path).load()
text, totals = blend_progressive(parsed_dir, cursor, slice_width=8)
cursor.save()
# ... send text to LLM ...
# next iteration automatically picks up from saved offsets
"""

import json
import os
from pathlib import Path


class BlendCursor:
    """Per-file turn-level cursor for progressive archive blending.

    State is a plain ``{filename: offset}`` dict where *offset* is the number
    of turns already consumed from that file.  Saves atomically via a
    ``.tmp`` rename so a crash mid-write never corrupts the cursor file.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._data: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> "BlendCursor":
        """Load cursor state from disk.  Returns self for chaining."""
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    raw = json.load(f)
                # Guard against corrupt / wrong-type values
                self._data = {k: int(v) for k, v in raw.items() if isinstance(v, (int, float))}
            except (json.JSONDecodeError, OSError, ValueError):
                self._data = {}
        return self

    def save(self) -> None:
        """Atomically save cursor state to disk."""
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self._path)

    def reset(self) -> None:
        """Clear all cursor offsets (start over from the beginning)."""
        self._data = {}

    # ------------------------------------------------------------------
    # Offset management
    # ------------------------------------------------------------------

    def get_offset(self, filename: str) -> int:
        """Return current turn offset for *filename* (0 if unseen)."""
        return self._data.get(filename, 0)

    def set_offset(self, filename: str, offset: int) -> None:
        """Set the turn offset for *filename*."""
        self._data[filename] = max(0, offset)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def data(self) -> dict[str, int]:
        """Read-only view of the raw ``{filename: offset}`` state."""
        return dict(self._data)

    def coverage(self, file_totals: dict[str, int]) -> dict[str, float]:
        """Return ``{filename: fraction_covered}`` for each file.

        Args:
            file_totals: Mapping of ``{filename: total_turns}`` as returned
                by :func:`blend_progressive`.

        Returns:
            Dict where values are in ``[0.0, 1.0]``.
        """
        result = {}
        for fname, total in file_totals.items():
            if total > 0:
                result[fname] = min(1.0, self._data.get(fname, 0) / total)
        return result

    def is_empty(self) -> bool:
        """True when no cursor state has been recorded yet."""
        return len(self._data) == 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_turns(path: Path) -> list[dict]:
    """Flatten all turns from a parsed JSONL file into an ordered list."""
    turns: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    conv = json.loads(line)
                    conv_id = conv.get("id", "?")
                    for turn in conv.get("turns", []):
                        turns.append({
                            "conv_id": conv_id,
                            "sender": turn.get("sender", "?"),
                            "text": turn.get("text", ""),
                        })
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return turns


def _format_slice(
    fname: str,
    turns: list[dict],
    offset: int,
    total: int,
    max_chars: int,
) -> str:
    """Format a single file's slice into annotated text."""
    end = offset + len(turns) - 1
    lines = [f"--- {fname}  turns {offset}–{end} / {total - 1} ---"]
    prev_conv_id = None
    for turn in turns:
        if prev_conv_id is not None and turn["conv_id"] != prev_conv_id:
            lines.append("[[ CONVERSATION BOUNDARY ]]")
        text = turn["text"][:max_chars]
        lines.append(f"[{turn['sender']}]: {text}")
        prev_conv_id = turn["conv_id"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def blend_progressive(
    parsed_dir: str,
    cursor: BlendCursor,
    slice_width: int = 8,
    max_chars_per_turn: int = 300,
    wrap_around: bool = True,
) -> tuple[str, dict[str, int]]:
    """Take one contiguous slice per file and advance each file's cursor.

    On each call the function:

    1. Reads every ``*.jsonl`` file in *parsed_dir* (sorted for
       determinism).
    2. For each file, extracts *slice_width* turns starting at that
       file's current cursor offset.
    3. Annotates ``[[ CONVERSATION BOUNDARY ]]`` wherever the slice crosses
       a conversation border within the file.
    4. Advances the cursor by the number of turns consumed.  If the file is
       exhausted, wraps back to 0 (when *wrap_around* is True) or skips the
       file for this pass.
    5. Returns the combined text plus a ``{filename: total_turns}`` dict
       that callers can use for coverage reporting.

    The cursor **is not saved to disk here** — call ``cursor.save()`` after
    each iteration so a crash does not lose progress.

    Args:
        parsed_dir: Directory containing parsed JSONL conversation files.
        cursor: :class:`BlendCursor` instance (pre-loaded to resume).
        slice_width: Turns to extract per file per call.
        max_chars_per_turn: Character truncation limit per turn.
        wrap_around: Wrap exhausted files back to offset 0 so the archive
            cycles indefinitely.  Set to ``False`` to skip exhausted files.

    Returns:
        ``(blended_text, file_totals)`` where *blended_text* is ready to
        send to the LLM and *file_totals* maps filename → total turn count.
        Returns ``("", {})`` when *parsed_dir* is empty or all files are
        exhausted with *wrap_around=False*.
    """
    files = sorted(Path(parsed_dir).glob("*.jsonl"))
    if not files:
        return "", {}

    parts: list[str] = []
    file_totals: dict[str, int] = {}

    for jsonl_file in files:
        fname = jsonl_file.name
        all_turns = _load_turns(jsonl_file)
        total = len(all_turns)
        file_totals[fname] = total

        if total == 0:
            continue

        offset = cursor.get_offset(fname)

        if offset >= total:
            if wrap_around:
                offset = 0
            else:
                continue  # exhausted — skip this file this pass

        slice_turns = all_turns[offset : offset + slice_width]
        new_offset = offset + len(slice_turns)
        if new_offset >= total and wrap_around:
            new_offset = 0  # ready to cycle on the next call

        cursor.set_offset(fname, new_offset)
        parts.append(_format_slice(fname, slice_turns, offset, total, max_chars_per_turn))

    return "\n\n".join(parts), file_totals
