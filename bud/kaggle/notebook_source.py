"""Kaggle notebook source template for GPU Ollama server."""

NOTEBOOK_SOURCE = r'''
import subprocess
import time
import os
import sys
import shutil

# ── 1. Install zstd (required by Ollama installer) ──────────────────────────
print("Installing zstd...")
subprocess.run(["apt-get", "update", "-qq"], check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], check=True)
print("zstd installed.")

# ── 2. Install Ollama ────────────────────────────────────────────────────────
print("Installing Ollama...")
result = subprocess.run(
    "curl -fsSL https://ollama.com/install.sh | sh",
    shell=True, capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print(f"Ollama install stderr: {{result.stderr}}", file=sys.stderr)
    sys.exit(1)

# ── 3. Install pyngrok ──────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyngrok"], check=True)

# ── 4. Start ollama serve ───────────────────────────────────────────────────
print("Starting ollama serve...")
ollama_proc = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for Ollama readiness
for i in range(30):
    try:
        import urllib.request
        req = urllib.request.urlopen("http://localhost:11434", timeout=2)
        if req.status == 200:
            print("Ollama is ready.")
            break
    except Exception:
        pass
    time.sleep(2)
else:
    print("ERROR: Ollama failed to start within 60 seconds", file=sys.stderr)
    sys.exit(1)

# ── 5. Restore cached models if dataset is mounted ─────────────────────────
MODEL_CACHE_SLUG = "{model_cache_slug}"
if MODEL_CACHE_SLUG:
    cache_input = f"/kaggle/input/{{MODEL_CACHE_SLUG.split('/')[-1]}}"
    cache_tar = os.path.join(cache_input, "ollama_models.tar.zst")
    if os.path.exists(cache_tar):
        print(f"Restoring cached models from {{cache_tar}}...")
        subprocess.run(
            ["tar", "--zstd", "-xf", cache_tar, "-C", "/"],
            check=True
        )
        print("Cached models restored.")

# ── 6. Pull models ─────────────────────────────────────────────────────────
LLM_MODEL = "{llm_model}"
EMBED_MODEL = "{embed_model}"

print(f"Pulling model: {{LLM_MODEL}} ...")
subprocess.run(["ollama", "pull", LLM_MODEL], capture_output=True, text=True, timeout=600)

print(f"Pulling embedding model: {{EMBED_MODEL}} ...")
subprocess.run(["ollama", "pull", EMBED_MODEL], capture_output=True, text=True, timeout=600)

# ── 7. Save models to working dir for future caching ───────────────────────
CACHE_WORKING = "/kaggle/working/ollama_cache"
os.makedirs(CACHE_WORKING, exist_ok=True)
cache_tar_out = os.path.join(CACHE_WORKING, "ollama_models.tar.zst")
print(f"Saving models to {{cache_tar_out}} for future caching...")
home = os.path.expanduser("~")
ollama_dir = os.path.join(home, ".ollama")
if os.path.isdir(ollama_dir):
    subprocess.run(
        ["tar", "--zstd", "-cf", cache_tar_out, "-C", "/", ollama_dir.lstrip("/")],
        check=True
    )
    print("Models saved.")

# ── 8. Set up ngrok tunnel ──────────────────────────────────────────────────
print("Setting up ngrok tunnel...")

from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
ngrok_token = secrets.get_secret("NGROK_TOKEN")

from pyngrok import ngrok

ngrok.set_auth_token(ngrok_token)

STATIC_DOMAIN = "{ngrok_static_domain}"

tunnel = ngrok.connect(
    addr="11434",
    proto="http",
    hostname=STATIC_DOMAIN,
    host_header="localhost:11434",
)
print(f"ngrok tunnel active: {{tunnel.public_url}}")

# ── 9. Keep alive ──────────────────────────────────────────────────────────
print("Server is running. Tunnel active. Waiting...")
try:
    while True:
        if ollama_proc.poll() is not None:
            print("Ollama process died. Exiting.")
            break
        time.sleep(30)
except KeyboardInterrupt:
    pass
finally:
    print("Shutting down...")
    ngrok.disconnect(tunnel.public_url)
    ollama_proc.terminate()
    print("Done.")
'''


def render_notebook_source(
    llm_model: str,
    embed_model: str,
    ngrok_static_domain: str,
    model_cache_slug: str = "",
) -> str:
    """Render the notebook source template with the given parameters.

    Args:
        llm_model: Ollama model name for LLM.
        embed_model: Ollama model name for embeddings.
        ngrok_static_domain: Static ngrok domain for the tunnel.
        model_cache_slug: Optional Kaggle dataset slug for cached models.

    Returns:
        Rendered Python source code string.
    """
    return NOTEBOOK_SOURCE.format(
        llm_model=llm_model,
        embed_model=embed_model,
        ngrok_static_domain=ngrok_static_domain,
        model_cache_slug=model_cache_slug,
    )
