"""Configuration management for Bud RAG Pipeline."""

import os
import shutil
import tempfile
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "data_dir": None,
    "output_dir": None,
    "llm": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "gemini-3-flash-preview:latest",
    },
    "embeddings": {
        "provider": "ollama",
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text",
    },
}

CONFIG_DIR = Path.home() / ".config" / "bud"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    return CONFIG_DIR


def get_config_file() -> Path:
    """Get the configuration file path."""
    return CONFIG_FILE


def ensure_config_dir() -> Path:
    """Ensure the configuration directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config() -> dict:
    """Load configuration from config file."""
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE, "r") as f:
            config = yaml.safe_load(f) or {}
        return config
    except yaml.YAMLError:
        return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> Path:
    """Save configuration to config file."""
    ensure_config_dir()

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=CONFIG_DIR, suffix=".yaml"
    ) as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        temp_path = f.name

    shutil.move(temp_path, CONFIG_FILE)
    return CONFIG_FILE


def _is_valid_url(url: str) -> bool:
    """Check if a string is a valid HTTP/HTTPS URL."""
    if not url or not isinstance(url, str):
        return False
    return url.startswith("http://") or url.startswith("https://")


def validate_config(config: dict) -> tuple[bool, list[str]]:
    """Validate configuration.

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Required string fields
    if not config.get("data_dir"):
        errors.append("data_dir is required")
    elif not Path(config["data_dir"]).is_absolute():
        errors.append("data_dir must be an absolute path")

    if not config.get("output_dir"):
        errors.append("output_dir is required")
    elif not Path(config["output_dir"]).is_absolute():
        errors.append("output_dir must be an absolute path")

    # LLM configuration
    llm = config.get("llm", {})
    if not llm.get("provider"):
        errors.append("llm.provider is required (choose from: ollama, openai, anthropic)")
    elif llm.get("provider") not in ["ollama", "openai", "anthropic"]:
        errors.append(f"llm.provider must be one of: ollama, openai, anthropic (got: {llm.get('provider')})")
    if not llm.get("base_url"):
        errors.append("llm.base_url is required")
    elif not _is_valid_url(llm.get("base_url")):
        errors.append("llm.base_url must be a valid HTTP/HTTPS URL")
    if not llm.get("model"):
        errors.append("llm.model is required")

    # Embeddings configuration
    embeddings = config.get("embeddings", {})
    if not embeddings.get("provider"):
        errors.append("embeddings.provider is required (choose from: ollama, openai)")
    elif embeddings.get("provider") not in ["ollama", "openai"]:
        errors.append(f"embeddings.provider must be one of: ollama, openai (got: {embeddings.get('provider')})")
    if not embeddings.get("base_url"):
        errors.append("embeddings.base_url is required")
    elif not _is_valid_url(embeddings.get("base_url")):
        errors.append("embeddings.base_url must be a valid HTTP/HTTPS URL")
    if not embeddings.get("model"):
        errors.append("embeddings.model is required")

    return len(errors) == 0, errors


def get_data_dir() -> Path:
    """Get the data directory from config."""
    config = load_config()
    data_dir_str = config.get("data_dir")
    if not data_dir_str:
        raise ValueError("data_dir is not configured. Run 'bud configure' first.")
    data_dir = Path(data_dir_str)
    if not data_dir.is_absolute():
        data_dir = Path.home() / data_dir
    return data_dir


def get_output_dir() -> Path:
    """Get the output directory from config."""
    config = load_config()
    output_dir_str = config.get("output_dir")
    if not output_dir_str:
        raise ValueError("output_dir is not configured. Run 'bud configure' first.")
    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = Path.home() / output_dir
    return output_dir
