"""Index management for Bud RAG Pipeline."""

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class IndexManager:
    """Manages the FAISS vector index for the RAG pipeline."""

    def __init__(self, output_dir: Path, config: dict):
        self._output_dir = output_dir
        self._config = config
        self._index_dir = output_dir / "index"
        self._schema_path = str(output_dir / "schema.json")
        self._progress_path = str(output_dir / "progress.json")
        self._embed_queue_path = str(output_dir / "embed_queue.jsonl")
        self._discovery_map_path = str(output_dir / "discovery_map.json")
        self._blend_cursor_path = str(output_dir / "blend_cursor.json")

    @property
    def index_dir(self) -> Path:
        """Return the index directory path."""
        return self._index_dir

    @property
    def schema_path(self) -> str:
        """Return the schema file path."""
        return self._schema_path

    @property
    def progress_path(self) -> str:
        """Return the progress tracking file path."""
        return self._progress_path

    @property
    def embed_queue_path(self) -> str:
        """Return the embed queue file path."""
        return self._embed_queue_path

    @property
    def discovery_map_path(self) -> str:
        """Return the discovery map file path."""
        return self._discovery_map_path

    @property
    def blend_cursor_path(self) -> str:
        """Return the blend cursor file path."""
        return self._blend_cursor_path

    def ensure_directories(self) -> None:
        """Create required output directories."""
        self._index_dir.mkdir(parents=True, exist_ok=True)

    def show_summary(self, total_chunks: int) -> None:
        """Display index summary.

        Args:
            total_chunks: Total number of chunks in index
        """
        table = Table(title="Vector Index Summary", box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Index directory", str(self._index_dir))
        table.add_row("Schema file", self._schema_path)
        table.add_row("Chunks indexed", str(total_chunks))

        console.print(table)

    def get_existing_index_count(self) -> int:
        """Get count of existing chunks in index.

        Returns:
            Number of chunks, or 0 if no index exists
        """
        from bud.lib.store import VectorStore

        index_path = str(self._index_dir / "chunks")
        store = VectorStore(index_path)
        if os.path.exists(f"{index_path}.faiss"):
            store.load()
            return store.count()
        return 0

    def check_completion(self, filename: str, batch: int) -> bool:
        """Check if a batch is already processed.

        Args:
            filename: Source file name
            batch: Batch number

        Returns:
            True if completed, False otherwise
        """
        from bud.lib.progress import ProgressTracker

        tracker = ProgressTracker(self._progress_path)
        return tracker.is_complete(filename, batch)

    def mark_complete(self, filename: str, batch: int) -> None:
        """Mark a batch as complete.

        Args:
            filename: Source file name
            batch: Batch number
        """
        from bud.lib.progress import ProgressTracker

        tracker = ProgressTracker(self._progress_path)
        tracker.mark_complete(filename, batch)

    def save_index(self, store) -> None:
        """Save the vector index.

        Args:
            store: VectorStore instance to save
        """
        store.save()

    def get_failed_batches(self) -> dict:
        """Get failed batches from progress tracking.

        Returns:
            Dict of {filename: [batch_numbers]}
        """
        from bud.lib.progress import ProgressTracker

        tracker = ProgressTracker(self._progress_path)
        return tracker.get_failed()
