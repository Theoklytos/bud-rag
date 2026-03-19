"""Schema manager for Bud RAG Pipeline."""

import copy
import json
import os
from datetime import datetime, timezone

REQUIRED_DIMENSIONS = ["geometry", "coherence", "texture", "terrain", "motifs"]
REQUIRED_FIELDS = ["version", "last_updated", "dimensions", "chunk_types",
                   "multi_value_dimensions", "candidates", "evolution_log"]

DEFAULT_SCHEMA = {
    "version": 1,
    "last_updated": "2026-03-18T00:00:00Z",
    "dimensions": {
        "geometry": ["linear", "recursive", "spiral", "bifurcating", "convergent"],
        "coherence": ["tight", "loose", "fragmented", "emergent", "contradictory"],
        "texture": ["dense", "sparse", "lyrical", "technical", "mythic", "raw"],
        "terrain": ["conceptual", "emotional", "procedural", "speculative", "relational"],
        "motifs": ["identity", "threshold", "resonance", "system-design", "becoming"],
    },
    "chunk_types": ["exchange", "monologue", "breakthrough", "definition", "artifact"],
    "multi_value_dimensions": ["motifs"],
    "candidates": {},
    "evolution_log": [],
}


class SchemaManager:
    """Manages schema evolution for conversation chunks."""

    def __init__(self, schema_path: str):
        self._path = schema_path
        self._tmp_path = schema_path + ".tmp"

    def get_default_schema(self) -> dict:
        """Return a copy of the default schema."""
        return copy.deepcopy(DEFAULT_SCHEMA)

    def load(self) -> dict:
        """Load schema from file, or return default if not exists."""
        if not os.path.exists(self._path):
            return self.get_default_schema()
        with open(self._path) as f:
            return json.load(f)

    def validate(self, schema: dict) -> bool:
        """Validate that schema has all required fields."""
        if not isinstance(schema, dict):
            return False
        for field in REQUIRED_FIELDS:
            if field not in schema:
                return False
        for dim in REQUIRED_DIMENSIONS:
            if dim not in schema.get("dimensions", {}):
                return False
        return True

    def save(self, schema: dict) -> None:
        """Save schema to file atomically."""
        schema["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self._tmp_path, "w") as f:
            json.dump(schema, f, indent=2)
        os.replace(self._tmp_path, self._path)

    def propose_candidate(self, dimension: str, value: str, example: str) -> None:
        """Register a candidate dimension value.

        Args:
            dimension: The dimension name
            value: The proposed value
            example: Example text showing this value
        """
        schema = self.load()
        key = f"{dimension}.{value}"
        if key not in schema["candidates"]:
            schema["candidates"][key] = {
                "count": 0,
                "first_seen_batch": 0,
                "examples": [],
            }
        schema["candidates"][key]["count"] += 1
        if len(schema["candidates"][key]["examples"]) < 3:
            schema["candidates"][key]["examples"].append(example[:120])
        self.save(schema)

    def apply_promotions(self, config: dict) -> list[str]:
        """Promote candidate values to dimensions if threshold is met.

        Args:
            config: Pipeline configuration

        Returns:
            List of promoted dimension values
        """
        threshold = config["pipeline"]["schema_evolution_confidence_threshold"]
        schema = self.load()
        promoted = []
        remaining_candidates = {}
        for key, data in schema["candidates"].items():
            if data["count"] >= threshold:
                dimension, value = key.split(".", 1)
                if dimension in schema["dimensions"] and value not in schema["dimensions"][dimension]:
                    schema["dimensions"][dimension].append(value)
                    schema["version"] += 1
                    schema["evolution_log"].append({
                        "version": schema["version"],
                        "value": key,
                        "count": data["count"],
                        "added_at": datetime.now(timezone.utc).isoformat(),
                    })
                    promoted.append(key)
            else:
                remaining_candidates[key] = data
        schema["candidates"] = remaining_candidates
        self.save(schema)
        return promoted
