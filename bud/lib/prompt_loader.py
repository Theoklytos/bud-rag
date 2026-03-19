"""Prompt loader for Bud RAG Pipeline."""

import os


class PromptLoader:
    """Loads and processes prompt templates."""

    def __init__(self, prompts_dir: str):
        self._dir = prompts_dir

    def list_presets(self) -> list[str]:
        """List available prompt presets.

        Returns:
            List of preset names (filenames without .md extension)
        """
        if not os.path.exists(self._dir):
            return []
        return [
            f[:-3] for f in os.listdir(self._dir)
            if f.endswith(".md")
        ]

    def load(self, preset_name: str, variables: dict) -> str:
        """Load a prompt preset and substitute variables.

        Args:
            preset_name: Name of the prompt preset
            variables: Dict of {variable_name: value} for substitution

        Returns:
            Processed prompt text

        Raises:
            FileNotFoundError: If preset not found
        """
        path = os.path.join(self._dir, f"{preset_name}.md")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt preset not found: {path}")
        with open(path) as f:
            template = f.read()
        for key, value in variables.items():
            template = template.replace(f"{{{{{key}}}}}", str(value))
        return template
