> **Note (2026-03-20):** This spec has been superseded by the Ollama-based
> architecture. The notebook now runs Ollama directly (not FastAPI +
> sentence-transformers). See CLAUDE.md for current architecture.

# Bud Kaggle GPU Manager — Architectural Specification

**Purpose**: This document is an implementation blueprint for adding managed Kaggle GPU embedding to the `bud` pipeline. The system starts a Kaggle notebook with a T4 GPU, exposes an embedding endpoint via ngrok, uses it for batch embedding, and shuts it down when finished. Includes a dead-man's switch for idle timeout protection.

**Companion to**: `BUD_MCP_SPEC.md` — these are independent modules that share the same codebase.

**Target runtime**: Bud CLI on Termux (Android). The Kaggle notebook runs remotely.

---

## 1. Problem Statement

Embedding on a phone via local Ollama is brutally slow and resource-constrained. Kaggle offers 30 hours/week of free T4 GPU time. But Kaggle notebooks keep running until manually stopped — wasting GPU quota on idle time. The solution is a managed lifecycle: bud starts the notebook, uses it, and kills it automatically.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Termux (Android)                                       │
│                                                         │
│  bud process                                            │
│    │                                                    │
│    ├─ parse → chunk (local Ollama LLM, unchanged)       │
│    │                                                    │
│    └─ embed ──► KaggleGPUManager                        │
│                  │  1. Push & start notebook via API     │
│                  │  2. Poll for ngrok URL                │
│                  │  3. Return embedding client           │
│                  │  4. After batch: POST /shutdown       │
│                  │  5. Fallback to local Ollama on fail  │
└──────────────────┼──────────────────────────────────────┘
                   │
                   ▼ (internet)
┌─────────────────────────────────────────────────────────┐
│  Kaggle Notebook (2x T4 GPU)                            │
│                                                         │
│  FastAPI server                                         │
│    POST /embed    — batch embed texts, return vectors   │
│    POST /shutdown — graceful self-termination            │
│    GET  /health   — liveness check                      │
│                                                         │
│  Dead-man's switch                                      │
│    └─ If no request in N minutes → self-terminate       │
│                                                         │
│  ngrok tunnel                                           │
│    └─ Exposes FastAPI on public URL                     │
│    └─ Writes URL to notebook output for polling         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. File Structure

```
bud/
├── lib/
│   ├── kaggle_gpu.py        # KaggleGPUManager — lifecycle orchestration
│   └── embeddings.py        # (existing) — add kaggle_gpu provider
├── kaggle/
│   ├── __init__.py
│   ├── notebook_source.py   # Python source code for the Kaggle notebook
│   └── kernel_meta.py       # Generates kernel-metadata.json for Kaggle API
```

---

## 4. Kaggle Notebook Code

This is the Python code that runs ON Kaggle. It will be stored as a string template in `bud/kaggle/notebook_source.py` so bud can push it programmatically via the Kaggle API.

### 4.1 Full Notebook Source (`bud/kaggle/notebook_source.py`)

```python
"""Source code template for the Kaggle embedding server notebook.

This module contains the Python source that bud pushes to Kaggle via
the Kaggle API. The notebook runs on a T4 GPU, loads an embedding
model, exposes a FastAPI server via ngrok, and self-terminates on
idle timeout or explicit shutdown.
"""

# The NGROK_TOKEN and MODEL_NAME placeholders are substituted at push time.
NOTEBOOK_SOURCE = '''
#!/usr/bin/env python3
"""Bud Embedding Server — runs on Kaggle with T4 GPU.

Exposes a FastAPI embedding endpoint via ngrok tunnel.
Self-terminates after idle timeout or explicit /shutdown call.
"""

import os
import sys
import time
import signal
import threading
from datetime import datetime, timezone

# ── Install dependencies ─────────────────────────────────────────
os.system("pip install -q fastapi uvicorn pyngrok sentence-transformers")

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from pyngrok import ngrok
import uvicorn


# ── Configuration ────────────────────────────────────────────────

MODEL_NAME = "{model_name}"
IDLE_TIMEOUT_SECONDS = {idle_timeout_seconds}
PORT = 8000
NGROK_TOKEN = "{ngrok_token}"

# ── State ────────────────────────────────────────────────────────

last_request_time = time.time()
request_lock = threading.Lock()
shutdown_flag = threading.Event()


def update_last_request():
    global last_request_time
    with request_lock:
        last_request_time = time.time()


# ── Load model ───────────────────────────────────────────────────

print(f"Loading embedding model: {{MODEL_NAME}}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {{device}}")

model = SentenceTransformer(MODEL_NAME, device=device)

dim = model.get_sentence_embedding_dimension()
print(f"Model loaded. Dimension: {{dim}}")


# ── FastAPI app ──────────────────────────────────────────────────

app = FastAPI(title="Bud Embedding Server")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to embed", min_length=1)
    normalize: bool = Field(default=True, description="L2-normalize vectors")


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    dimension: int
    count: int
    model: str
    device: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    dimension: int
    uptime_seconds: float
    idle_seconds: float
    idle_timeout_seconds: int


SERVER_START_TIME = time.time()


@app.get("/health")
async def health() -> HealthResponse:
    update_last_request()
    with request_lock:
        idle = time.time() - last_request_time
    return HealthResponse(
        status="ok",
        model=MODEL_NAME,
        device=device,
        dimension=dim,
        uptime_seconds=round(time.time() - SERVER_START_TIME, 1),
        idle_seconds=round(idle, 1),
        idle_timeout_seconds=IDLE_TIMEOUT_SECONDS,
    )


@app.post("/embed")
async def embed(req: EmbedRequest) -> EmbedResponse:
    update_last_request()
    try:
        vectors = model.encode(
            req.texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=req.normalize,
            convert_to_numpy=True,
        )
        return EmbedResponse(
            vectors=vectors.tolist(),
            dimension=dim,
            count=len(req.texts),
            model=MODEL_NAME,
            device=device,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shutdown")
async def shutdown():
    """Graceful shutdown. Returns confirmation, then exits."""
    update_last_request()
    shutdown_flag.set()
    return {{"status": "shutting_down", "message": "Server will terminate in 2 seconds."}}


# ── Dead-man's switch ────────────────────────────────────────────

def watchdog():
    """Background thread: terminates server after idle timeout."""
    while not shutdown_flag.is_set():
        time.sleep(30)  # Check every 30 seconds
        with request_lock:
            idle = time.time() - last_request_time
        if idle > IDLE_TIMEOUT_SECONDS:
            print(f"Idle timeout ({{IDLE_TIMEOUT_SECONDS}}s). Shutting down.")
            shutdown_flag.set()
            break

    # Give in-flight requests 2 seconds to complete
    time.sleep(2)
    print("Bud embedding server terminated.")
    os._exit(0)


watchdog_thread = threading.Thread(target=watchdog, daemon=True)
watchdog_thread.start()


# ── ngrok tunnel ─────────────────────────────────────────────────

ngrok.set_auth_token(NGROK_TOKEN)
tunnel = ngrok.connect(PORT)
public_url = tunnel.public_url

# This print is critical — bud polls notebook output for this marker
print(f"BUD_NGROK_URL={{public_url}}")
sys.stdout.flush()


# ── Run server ───────────────────────────────────────────────────

print(f"Starting embedding server on port {{PORT}}")
uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
'''


def render_notebook_source(
    model_name: str = "all-MiniLM-L6-v2",
    ngrok_token: str = "",
    idle_timeout_seconds: int = 600,
) -> str:
    """Render the notebook source with configuration substituted.

    Args:
        model_name: HuggingFace sentence-transformers model name.
        ngrok_token: ngrok authentication token.
        idle_timeout_seconds: Seconds of idle before self-termination.

    Returns:
        Complete Python source string ready to push to Kaggle.
    """
    return NOTEBOOK_SOURCE.format(
        model_name=model_name,
        ngrok_token=ngrok_token,
        idle_timeout_seconds=idle_timeout_seconds,
    )
```

### 4.2 Kernel Metadata Generator (`bud/kaggle/kernel_meta.py`)

```python
"""Generate Kaggle kernel-metadata.json for pushing notebooks via API."""

import json


def generate_kernel_metadata(
    username: str,
    kernel_slug: str = "bud-embedding-server",
    title: str = "Bud Embedding Server",
    enable_gpu: bool = True,
    enable_internet: bool = True,
) -> dict:
    """Generate the metadata dict Kaggle API requires for kernel push.

    Args:
        username: Kaggle username.
        kernel_slug: URL-safe kernel identifier.
        title: Human-readable kernel title.
        enable_gpu: Whether to request GPU acceleration.
        enable_internet: Whether to enable internet (required for ngrok).

    Returns:
        Dict matching Kaggle's kernel-metadata.json schema.
    """
    return {
        "id": f"{username}/{kernel_slug}",
        "title": title,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "competition_sources": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def write_kernel_package(
    output_dir: str,
    source_code: str,
    metadata: dict,
) -> None:
    """Write the kernel package files that `kaggle kernels push` expects.

    Creates:
        output_dir/kernel-metadata.json
        output_dir/script.py

    Args:
        output_dir: Directory to write files into.
        source_code: Python source code for the notebook.
        metadata: Kernel metadata dict from generate_kernel_metadata().
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    meta_path = os.path.join(output_dir, "kernel-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    script_path = os.path.join(output_dir, "script.py")
    with open(script_path, "w") as f:
        f.write(source_code)
```

---

## 5. Kaggle GPU Manager (`bud/lib/kaggle_gpu.py`)

This module lives on the Termux side and orchestrates the full lifecycle.

```python
"""Managed Kaggle GPU lifecycle for remote embedding.

Handles: push notebook → poll for ngrok URL → provide embedding
client → shutdown when done → fallback to local on failure.
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from bud.kaggle.notebook_source import render_notebook_source
from bud.kaggle.kernel_meta import generate_kernel_metadata, write_kernel_package


@dataclass
class KaggleGPUConfig:
    """Configuration for the Kaggle GPU manager."""

    kaggle_username: str
    kernel_slug: str = "bud-embedding-server"
    ngrok_token: str = ""
    model_name: str = "all-MiniLM-L6-v2"
    idle_timeout_minutes: int = 10
    poll_timeout_seconds: int = 180
    poll_interval_seconds: int = 15
    health_check_retries: int = 3

    @property
    def kernel_ref(self) -> str:
        return f"{self.kaggle_username}/{self.kernel_slug}"


class KaggleGPUError(Exception):
    """Raised when Kaggle GPU operations fail."""
    pass


class KaggleGPUManager:
    """Manages the lifecycle of a Kaggle embedding server notebook.

    Usage:
        manager = KaggleGPUManager(config)
        try:
            url = manager.start()          # Push, start, poll for URL
            vectors = manager.embed(texts)  # Batch embed via remote GPU
        finally:
            manager.stop()                 # Explicit shutdown
    """

    def __init__(self, config: KaggleGPUConfig, on_status=None):
        """Initialize the manager.

        Args:
            config: KaggleGPUConfig with credentials and settings.
            on_status: Optional callback(message: str) for progress updates.
        """
        self._config = config
        self._ngrok_url: Optional[str] = None
        self._on_status = on_status or (lambda msg: None)
        self._tmp_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "bud", "kaggle_kernel"
        )

    @property
    def is_running(self) -> bool:
        """Check if the remote server is reachable."""
        if not self._ngrok_url:
            return False
        try:
            r = requests.get(
                f"{self._ngrok_url}/health",
                timeout=10,
                headers={"ngrok-skip-browser-warning": "true"},
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    def start(self) -> str:
        """Push the notebook to Kaggle, start it, and poll for ngrok URL.

        Returns:
            The public ngrok URL for the embedding server.

        Raises:
            KaggleGPUError: If notebook fails to start or URL not found.
        """
        self._on_status("Preparing Kaggle embedding kernel...")

        # 1. Render notebook source with config
        source = render_notebook_source(
            model_name=self._config.model_name,
            ngrok_token=self._config.ngrok_token,
            idle_timeout_seconds=self._config.idle_timeout_minutes * 60,
        )

        # 2. Generate metadata
        metadata = generate_kernel_metadata(
            username=self._config.kaggle_username,
            kernel_slug=self._config.kernel_slug,
        )

        # 3. Write kernel package to temp dir
        write_kernel_package(self._tmp_dir, source, metadata)
        self._on_status("Pushing kernel to Kaggle...")

        # 4. Push via Kaggle CLI
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", self._tmp_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise KaggleGPUError(
                f"Kaggle push failed: {result.stderr.strip()}"
            )
        self._on_status("Kernel pushed. Waiting for startup...")

        # 5. Poll for notebook output containing the ngrok URL
        self._ngrok_url = self._poll_for_url()
        self._on_status(f"Embedding server live at {self._ngrok_url}")

        # 6. Verify health
        self._wait_for_health()

        return self._ngrok_url

    def stop(self) -> None:
        """Send shutdown command to the remote server.

        Also attempts to stop the Kaggle kernel via API as a fallback.
        Silently handles errors — stop() should never raise.
        """
        # Try graceful shutdown via API
        if self._ngrok_url:
            try:
                requests.post(
                    f"{self._ngrok_url}/shutdown",
                    timeout=10,
                    headers={"ngrok-skip-browser-warning": "true"},
                )
                self._on_status("Shutdown signal sent to embedding server.")
            except requests.RequestException:
                self._on_status("Could not reach server for graceful shutdown.")

        # Also cancel via Kaggle API as belt-and-suspenders
        try:
            subprocess.run(
                [
                    "kaggle", "kernels", "status",
                    self._config.kernel_ref,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Note: Kaggle API doesn't have a direct "stop" command,
            # but the notebook's self-termination handles this.
            # The dead-man's switch is the true safety net.
        except (subprocess.TimeoutExpired, OSError):
            pass

        self._ngrok_url = None

    def embed(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        """Embed a batch of texts via the remote Kaggle GPU server.

        Args:
            texts: List of text strings to embed.
            normalize: Whether to L2-normalize the vectors.

        Returns:
            List of embedding vectors (list of floats).

        Raises:
            KaggleGPUError: If the server is unreachable or returns an error.
        """
        if not self._ngrok_url:
            raise KaggleGPUError("Embedding server not started. Call start() first.")

        try:
            response = requests.post(
                f"{self._ngrok_url}/embed",
                json={"texts": texts, "normalize": normalize},
                timeout=120,  # Large batches may take time
                headers={"ngrok-skip-browser-warning": "true"},
            )
            response.raise_for_status()
            data = response.json()
            return data["vectors"]
        except requests.Timeout:
            raise KaggleGPUError("Embedding request timed out.")
        except requests.RequestException as e:
            raise KaggleGPUError(f"Embedding request failed: {e}")

    def embed_single(self, text: str, normalize: bool = True) -> list[float]:
        """Embed a single text. Convenience wrapper around embed()."""
        vectors = self.embed([text], normalize=normalize)
        return vectors[0]

    # ── Private methods ──────────────────────────────────────────

    def _poll_for_url(self) -> str:
        """Poll Kaggle kernel output for the BUD_NGROK_URL marker.

        Returns:
            The ngrok public URL.

        Raises:
            KaggleGPUError: If URL not found within timeout.
        """
        deadline = time.time() + self._config.poll_timeout_seconds
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            self._on_status(
                f"Polling for ngrok URL (attempt {attempt})..."
            )
            time.sleep(self._config.poll_interval_seconds)

            try:
                result = subprocess.run(
                    [
                        "kaggle", "kernels", "output",
                        self._config.kernel_ref,
                        "-p", self._tmp_dir,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Check both stdout and any output files
                output = result.stdout + result.stderr

                # Also check log file if it was downloaded
                log_path = os.path.join(self._tmp_dir, "output.log")
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        output += f.read()

                # Search for our marker
                match = re.search(r"BUD_NGROK_URL=(https://[^\s]+)", output)
                if match:
                    return match.group(1)

            except (subprocess.TimeoutExpired, OSError):
                continue

        raise KaggleGPUError(
            f"Timed out waiting for ngrok URL after "
            f"{self._config.poll_timeout_seconds}s. "
            f"Check Kaggle notebook logs for errors."
        )

    def _wait_for_health(self) -> None:
        """Wait for the /health endpoint to respond.

        Raises:
            KaggleGPUError: If health check fails after retries.
        """
        for i in range(self._config.health_check_retries):
            try:
                r = requests.get(
                    f"{self._ngrok_url}/health",
                    timeout=15,
                    headers={"ngrok-skip-browser-warning": "true"},
                )
                if r.status_code == 200:
                    data = r.json()
                    self._on_status(
                        f"Health OK — model: {data.get('model')}, "
                        f"device: {data.get('device')}, "
                        f"dim: {data.get('dimension')}"
                    )
                    return
            except requests.RequestException:
                pass
            time.sleep(5)

        raise KaggleGPUError(
            f"Health check failed after {self._config.health_check_retries} "
            f"attempts at {self._ngrok_url}/health"
        )


class KaggleEmbeddingClient:
    """Drop-in embedding client compatible with bud's embedding interface.

    Wraps KaggleGPUManager to match the same interface as the Ollama
    and OpenAI embedding clients in bud.lib.embeddings. This allows
    the pipeline to swap providers without any changes to the
    embed stage.

    Usage in the pipeline:
        client = KaggleEmbeddingClient(kaggle_config)
        client.start()  # Boots the Kaggle notebook
        vectors = client.embed(text)  # Single text
        vectors = client.embed_batch(texts)  # Batch
        client.stop()  # Shuts down notebook
    """

    def __init__(self, config: KaggleGPUConfig, on_status=None):
        self._manager = KaggleGPUManager(config, on_status=on_status)
        self._started = False

    def start(self) -> None:
        """Start the remote embedding server."""
        self._manager.start()
        self._started = True

    def stop(self) -> None:
        """Stop the remote embedding server."""
        self._manager.stop()
        self._started = False

    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Matches existing bud interface."""
        if not self._started:
            self.start()
        return self._manager.embed_single(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. More efficient than single calls."""
        if not self._started:
            self.start()
        return self._manager.embed(texts)

    @property
    def dimension(self) -> Optional[int]:
        """Get embedding dimension from remote server."""
        if not self._started or not self._manager._ngrok_url:
            return None
        try:
            r = requests.get(
                f"{self._manager._ngrok_url}/health",
                timeout=10,
                headers={"ngrok-skip-browser-warning": "true"},
            )
            return r.json().get("dimension")
        except requests.RequestException:
            return None
```

---

## 6. Configuration

### 6.1 Config Schema Addition

Add to `~/.config/bud/config.yaml`:

```yaml
embeddings:
  # Provider: "ollama", "openai", or "kaggle_gpu"
  provider: kaggle_gpu

  # Kaggle GPU settings (only used when provider is kaggle_gpu)
  kaggle_username: your-kaggle-username
  ngrok_token: your-ngrok-authtoken
  model_name: all-MiniLM-L6-v2    # any sentence-transformers model
  idle_timeout_minutes: 10
  poll_timeout_seconds: 180

  # Fallback to local provider if remote fails
  fallback_provider: ollama
  fallback_base_url: http://localhost:11434
  fallback_model: nomic-embed-text
```

### 6.2 Secrets Management

The ngrok token and Kaggle API key are sensitive. They should be loaded from environment variables or a secrets file, not hardcoded in config. The recommended approach:

```yaml
# In config.yaml, reference env vars:
embeddings:
  ngrok_token: ${NGROK_TOKEN}
  kaggle_username: ${KAGGLE_USERNAME}
```

The config loader should expand `${VAR}` references from the environment. If this pattern doesn't exist in `bud/config.py` yet, add it.

---

## 7. Integration with Existing Pipeline

### 7.1 Changes to `bud/lib/embeddings.py`

Add a new provider branch. The key design goal: the embed stage doesn't know or care which provider is active. It calls `client.embed()` and gets vectors.

```python
# In the function that creates embedding clients based on config:

def get_embedding_client(config: dict):
    """Factory for embedding clients based on provider config."""
    provider = config.get("provider", "ollama")

    if provider == "kaggle_gpu":
        from bud.lib.kaggle_gpu import KaggleEmbeddingClient, KaggleGPUConfig

        gpu_config = KaggleGPUConfig(
            kaggle_username=config["kaggle_username"],
            ngrok_token=config["ngrok_token"],
            model_name=config.get("model_name", "all-MiniLM-L6-v2"),
            idle_timeout_minutes=config.get("idle_timeout_minutes", 10),
            poll_timeout_seconds=config.get("poll_timeout_seconds", 180),
        )
        return KaggleEmbeddingClient(gpu_config)

    elif provider == "ollama":
        # ... existing Ollama client creation ...
        pass

    elif provider == "openai":
        # ... existing OpenAI client creation ...
        pass
```

### 7.2 Changes to `bud/stages/embed.py`

The embed stage needs to handle the lifecycle — start before embedding, stop after. Add hooks at the batch level:

```python
# At the start of the embed run:
if hasattr(embed_client, "start"):
    embed_client.start()

# ... existing batch embedding loop ...

# At the end of the embed run (in a finally block):
if hasattr(embed_client, "stop"):
    embed_client.stop()
```

This is duck-typed so Ollama and OpenAI clients (which don't have start/stop) are unaffected.

### 7.3 Fallback Logic

When `fallback_provider` is set, wrap the Kaggle client in a fallback wrapper:

```python
class FallbackEmbeddingClient:
    """Tries primary client, falls back to secondary on failure."""

    def __init__(self, primary, secondary, on_fallback=None):
        self._primary = primary
        self._secondary = secondary
        self._on_fallback = on_fallback or (lambda: None)
        self._using_fallback = False

    def embed(self, text: str) -> list[float]:
        if not self._using_fallback:
            try:
                return self._primary.embed(text)
            except Exception:
                self._using_fallback = True
                self._on_fallback()
        return self._secondary.embed(text)

    def start(self):
        try:
            if hasattr(self._primary, "start"):
                self._primary.start()
        except Exception:
            self._using_fallback = True
            self._on_fallback()

    def stop(self):
        if hasattr(self._primary, "stop"):
            self._primary.stop()
```

---

## 8. CLI Integration

### 8.1 New Command: `bud gpu`

Add a diagnostic command for managing the Kaggle GPU manually:

```python
@cli.command()
@click.option("--start", is_flag=True, help="Start the embedding server")
@click.option("--stop", is_flag=True, help="Stop the embedding server")
@click.option("--status", is_flag=True, help="Check server status")
def gpu(start, stop, status):
    """Manage the Kaggle GPU embedding server."""
    config = load_config()
    gpu_config = KaggleGPUConfig(
        kaggle_username=config["embeddings"]["kaggle_username"],
        ngrok_token=config["embeddings"]["ngrok_token"],
        model_name=config["embeddings"].get("model_name", "all-MiniLM-L6-v2"),
    )
    manager = KaggleGPUManager(gpu_config, on_status=click.echo)

    if start:
        url = manager.start()
        click.echo(f"Server running at: {url}")
    elif stop:
        manager.stop()
        click.echo("Shutdown signal sent.")
    elif status:
        if manager.is_running:
            click.echo("Embedding server is running.")
        else:
            click.echo("Embedding server is not reachable.")
```

### 8.2 Integration with `bud process`

When provider is `kaggle_gpu`, the process command should show GPU lifecycle in its Rich progress output:

```
→ Starting Kaggle GPU embedding server...
  Pushing kernel to Kaggle...
  Polling for ngrok URL (attempt 1)...
  Polling for ngrok URL (attempt 2)...
  Embedding server live at https://xxxx.ngrok-free.app
  Health OK — model: all-MiniLM-L6-v2, device: cuda, dim: 384

→ Chunking and embedding 43 conversations
  embed    batch 1/1    166/223 chunks    ████████████    0:02:15

→ Shutting down Kaggle GPU server...
  Shutdown signal sent to embedding server.
```

---

## 9. Dependencies to Add

On Termux (bud side):
```
requests          # Likely already present
kaggle            # Kaggle CLI + API (pip install kaggle)
```

On Kaggle (notebook side — installed at runtime):
```
fastapi
uvicorn
pyngrok
sentence-transformers
```

These are installed inside the notebook via `pip install -q` at runtime. They don't need to be in bud's dependencies.

---

## 10. Testing Strategy

### Unit tests (`tests/test_kaggle_gpu.py`)

- `test_render_notebook_source` — verify template substitution works, ngrok token and model name appear in rendered source
- `test_generate_kernel_metadata` — verify metadata structure matches Kaggle API requirements
- `test_write_kernel_package` — verify files are written to correct paths
- `test_kaggle_gpu_config` — verify kernel_ref property, default values
- `test_embed_batch_request` — mock requests.post, verify correct payload sent to /embed
- `test_embed_single_wraps_batch` — verify single calls batch with size 1
- `test_fallback_client` — mock primary failure, verify secondary is used
- `test_stop_sends_shutdown` — mock requests.post, verify /shutdown is called
- `test_poll_finds_url` — mock subprocess output containing BUD_NGROK_URL marker

### Integration test (manual)

1. Run `bud gpu --start` and verify URL is printed
2. Run `bud gpu --status` and verify "running"
3. Run a small embedding batch manually
4. Run `bud gpu --stop` and verify shutdown
5. Wait for idle timeout and verify auto-shutdown
6. Run `bud process` with `kaggle_gpu` provider end-to-end

---

## 11. Embedding Model Compatibility

The notebook uses `sentence-transformers`, which supports hundreds of models. Some recommended options for bud:

| Model | Dimensions | Speed | Quality | Notes |
|-------|-----------|-------|---------|-------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Best for constrained environments |
| `all-mpnet-base-v2` | 768 | Medium | Better | Good general-purpose |
| `nomic-embed-text-v1.5` | 768 | Medium | Better | What you may be using with Ollama |
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Good | Strong for retrieval |

**Important**: The model used for embedding must match between index build time and query time. If you embed with `all-MiniLM-L6-v2` on Kaggle, you must query with the same model (or same dimensionality + compatible vectors). The `model_registry` in bud should track which model was used to build the index.

---

## 12. Security Notes

- The ngrok token is sensitive — never commit it to git or print it in logs
- The Kaggle API key (`~/.kaggle/kaggle.json`) is also sensitive
- The ngrok free tier creates public URLs — the embedding endpoint is technically accessible to anyone who guesses the URL, though it's ephemeral and short-lived
- For additional security, the notebook could require a shared secret token in request headers, validated server-side. This is optional but recommended if paranoia warrants it.

---

## 13. Known Limitations and Edge Cases

1. **Kaggle session limits**: Notebooks can run for a maximum of ~12 hours before Kaggle kills them regardless. The dead-man's switch should fire long before this.

2. **ngrok free tier limits**: ngrok free tier allows one tunnel at a time. If you have another ngrok tunnel running, the notebook tunnel may fail. Kill other tunnels first.

3. **Model download time**: The first time a notebook runs with a new model, sentence-transformers will download it. This can take 1-3 minutes depending on model size. Subsequent runs use Kaggle's cache (if the kernel hasn't been evicted).

4. **Polling reliability**: `kaggle kernels output` may not return output immediately after the notebook starts. The polling loop handles this, but if Kaggle is slow, increase `poll_timeout_seconds`.

5. **Concurrent runs**: Don't run multiple bud processes that all try to start the same Kaggle kernel. The manager should be a singleton or use a lockfile.

---

## 14. Implementation Order

For the implementing agent (Claude Code Opus), proceed in this order:

1. **Create `bud/kaggle/` package** with `__init__.py`
2. **Implement `bud/kaggle/notebook_source.py`** — the template and render function
3. **Implement `bud/kaggle/kernel_meta.py`** — metadata generation
4. **Write tests** for template rendering and metadata — `tests/test_kaggle_notebook.py`
5. **Implement `bud/lib/kaggle_gpu.py`** — KaggleGPUManager + KaggleEmbeddingClient + FallbackEmbeddingClient
6. **Write tests** for the manager — `tests/test_kaggle_gpu.py`
7. **Integrate into `bud/lib/embeddings.py`** — add kaggle_gpu provider branch
8. **Integrate into `bud/stages/embed.py`** — add start/stop lifecycle hooks
9. **Add `bud gpu` CLI command** to `bud/cli.py`
10. **Update `bud/config.py`** — handle new config fields and env var expansion
11. **Add `kaggle` to dependencies**
12. **Manual end-to-end test** on Termux + Kaggle

Each step should pass tests before proceeding to the next.

---

## 15. Adaptation Notes

Same caveat as the MCP spec: the code here is structurally correct but must be adapted to match exact function signatures in the existing codebase. Specifically:

- `get_embedding_client()` — may not exist as a factory function; check how `bud/stages/embed.py` currently creates its client
- The embed interface (`embed(text) -> list[float]`) — verify this matches what the embed stage calls
- Config loading — check how `bud/config.py` parses YAML and whether it already supports env var expansion
- Rich progress integration — check how `bud process` currently displays progress and match the pattern for GPU lifecycle messages
- `model_registry` — the Kaggle model dimensions need to be registered so the chunker knows the limits

The implementing agent should read `bud/lib/embeddings.py`, `bud/stages/embed.py`, and `bud/config.py` before writing code.
