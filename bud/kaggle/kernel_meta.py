"""Kaggle kernel metadata generation and packaging."""

import json
import os
from pathlib import Path


def generate_kernel_metadata(
    username: str,
    kernel_slug: str = "bud-embedding-server",
    title: str = "Bud Embedding Server",
    language: str = "python",
    kernel_type: str = "script",
    enable_gpu: bool = True,
    enable_internet: bool = True,
    dataset_sources: list[str] | None = None,
) -> dict:
    """Generate Kaggle kernel-metadata.json content.

    Args:
        username: Kaggle username.
        kernel_slug: Kernel slug identifier.
        title: Human-readable kernel title.
        language: Programming language.
        kernel_type: Type of kernel (script or notebook).
        enable_gpu: Whether to request GPU accelerator.
        enable_internet: Whether to enable internet access.
        dataset_sources: Optional list of Kaggle dataset slugs to attach.

    Returns:
        Dictionary suitable for writing as kernel-metadata.json.
    """
    meta = {
        "id": f"{username}/{kernel_slug}",
        "title": title,
        "code_file": "script.py",
        "language": language,
        "kernel_type": kernel_type,
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
    }
    if dataset_sources:
        meta["dataset_sources"] = dataset_sources
    return meta


def write_kernel_package(
    output_dir: str | Path,
    source_code: str,
    metadata: dict,
) -> None:
    """Write a Kaggle kernel package (metadata + script) to a directory.

    Args:
        output_dir: Directory to write files into.
        source_code: Python source code for script.py.
        metadata: Kernel metadata dictionary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "kernel-metadata.json"
    script_path = output_dir / "script.py"

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(script_path, "w") as f:
        f.write(source_code)
