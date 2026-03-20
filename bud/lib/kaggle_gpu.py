"""Kaggle GPU lifecycle manager.

Manages a Kaggle kernel running an Ollama server exposed via a static ngrok
domain.  Provides a context manager (``kaggle_gpu_session``) that auto-starts
the kernel before AI work and cancels it afterwards.
"""

import logging
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


class KaggleGPUError(Exception):
    """Error from Kaggle GPU operations."""


@dataclass
class KaggleGPUConfig:
    """Configuration for Kaggle GPU kernel lifecycle."""

    kaggle_username: str = ""
    kernel_slug: str = "bud-gpu-server"
    ngrok_static_domain: str = ""
    llm_model: str = ""
    embed_model: str = ""
    poll_timeout_seconds: int = 600
    poll_interval_seconds: int = 15
    health_check_retries: int = 10
    model_cache_dataset: str = ""

    @property
    def kernel_ref(self) -> str:
        return f"{self.kaggle_username}/{self.kernel_slug}"

    @property
    def health_url(self) -> str:
        return f"https://{self.ngrok_static_domain}"


class KaggleGPUManager:
    """Manages a Kaggle GPU kernel running Ollama via ngrok."""

    def __init__(self, config: KaggleGPUConfig):
        self.config = config

    @property
    def is_running(self) -> bool:
        """Check if the remote Ollama is reachable at the static domain."""
        if not self.config.ngrok_static_domain:
            return False
        try:
            resp = requests.get(self.config.health_url, timeout=10)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start(self) -> str:
        """Start the Kaggle kernel if not already running.

        Returns the public URL of the Ollama server.
        """
        if self.is_running:
            logger.info("Kaggle kernel already running")
            return self.config.health_url

        self._push_kernel()
        self._wait_for_health()
        return self.config.health_url

    def stop(self) -> None:
        """Cancel the Kaggle kernel (best-effort)."""
        try:
            subprocess.run(
                ["kaggle", "kernels", "cancel", self.config.kernel_ref],
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            pass  # Best effort

    def _push_kernel(self) -> None:
        """Push the notebook to Kaggle."""
        from bud.kaggle.notebook_source import render_notebook_source
        from bud.kaggle.kernel_meta import (
            generate_kernel_metadata,
            write_kernel_package,
        )

        source = render_notebook_source(
            llm_model=self.config.llm_model,
            embed_model=self.config.embed_model,
            ngrok_static_domain=self.config.ngrok_static_domain,
            model_cache_slug=self.config.model_cache_dataset,
        )

        dataset_sources = None
        if self.config.model_cache_dataset:
            dataset_sources = [self.config.model_cache_dataset]

        metadata = generate_kernel_metadata(
            username=self.config.kaggle_username,
            kernel_slug=self.config.kernel_slug,
            dataset_sources=dataset_sources,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            write_kernel_package(tmpdir, source, metadata)
            try:
                subprocess.run(
                    ["kaggle", "kernels", "push", "-p", tmpdir],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise KaggleGPUError(
                    f"Failed to push kernel: {e.stderr}"
                ) from e
            except FileNotFoundError:
                raise KaggleGPUError(
                    "kaggle CLI not found. Install with: pip install kaggle"
                )

    def _wait_for_health(self) -> None:
        """Poll the static domain until Ollama responds or timeout."""
        deadline = time.time() + self.config.poll_timeout_seconds

        while time.time() < deadline:
            try:
                resp = requests.get(self.config.health_url, timeout=10)
                if resp.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(self.config.poll_interval_seconds)

        raise KaggleGPUError(
            f"Timed out waiting for Ollama at {self.config.health_url} "
            f"after {self.config.poll_timeout_seconds}s"
        )


def _build_config(config: dict) -> KaggleGPUConfig:
    """Build KaggleGPUConfig from the kaggle section of bud config."""
    kaggle_cfg = config.get("kaggle", {})
    return KaggleGPUConfig(
        kaggle_username=kaggle_cfg.get("username", ""),
        kernel_slug=kaggle_cfg.get("kernel_slug", "bud-gpu-server"),
        ngrok_static_domain=kaggle_cfg.get("ngrok_static_domain", ""),
        llm_model=config.get("llm", {}).get("model", ""),
        embed_model=config.get("embeddings", {}).get("model", ""),
        poll_timeout_seconds=kaggle_cfg.get("poll_timeout_seconds", 600),
        poll_interval_seconds=kaggle_cfg.get("poll_interval_seconds", 15),
        health_check_retries=kaggle_cfg.get("health_check_retries", 10),
        model_cache_dataset=kaggle_cfg.get("model_cache_dataset", ""),
    )


@contextmanager
def kaggle_gpu_session(config: dict):
    """Context manager that starts a Kaggle GPU kernel and stops it on exit.

    If no ``kaggle`` section is present in config, this is a no-op.
    """
    kaggle_cfg = config.get("kaggle")
    if not kaggle_cfg or not kaggle_cfg.get("ngrok_static_domain"):
        yield
        return

    manager = KaggleGPUManager(_build_config(config))
    manager.start()
    try:
        yield
    finally:
        manager.stop()
