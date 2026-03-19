"""Progress tracking for Bud RAG Pipeline."""

import json
import os


class ProgressTracker:
    """Tracks pipeline progress for resume capability."""

    def __init__(self, path: str):
        self._path = path

    def load(self) -> dict:
        """Load progress state from file."""
        if not os.path.exists(self._path):
            return {}
        with open(self._path) as f:
            return json.load(f)

    def _save(self, state: dict) -> None:
        """Save progress state to file."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(state, f, indent=2)

    def _ensure_file(self, state: dict, filename: str) -> None:
        """Initialize file entry in state if needed."""
        if filename not in state:
            state[filename] = {"completed": [], "failed": {}, "in_progress": None}

    def mark_complete(self, filename: str, batch: int) -> None:
        """Mark a batch as complete.

        Args:
            filename: The source file name
            batch: Batch number
        """
        state = self.load()
        self._ensure_file(state, filename)
        if batch not in state[filename]["completed"]:
            state[filename]["completed"].append(batch)
        state[filename]["failed"].pop(str(batch), None)
        self._save(state)

    def mark_failed(self, filename: str, batch: int, error: str) -> None:
        """Mark a batch as failed.

        Args:
            filename: The source file name
            batch: Batch number
            error: Error message
        """
        state = self.load()
        self._ensure_file(state, filename)
        state[filename]["failed"][str(batch)] = error
        self._save(state)

    def is_complete(self, filename: str, batch: int) -> bool:
        """Check if a batch is complete.

        Args:
            filename: The source file name
            batch: Batch number

        Returns:
            True if batch completed successfully
        """
        state = self.load()
        return batch in state.get(filename, {}).get("completed", [])

    def get_failed(self) -> dict:
        """Get all failed batches.

        Returns:
            Dict of {filename: [batch_numbers]}
        """
        state = self.load()
        result = {}
        for fname, data in state.items():
            if data.get("failed"):
                result[fname] = [int(k) for k in data["failed"].keys()]
        return result
