#!/usr/bin/env bash
# install.sh — bootstrap a local bud-rag development environment
set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}▶ $*${NC}"; }
success() { echo -e "${GREEN}✓ $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠  $*${NC}"; }
error()   { echo -e "${RED}✗ $*${NC}" >&2; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}$*${NC}\n"; }

# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------
header "bud-rag installer"

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" &>/dev/null || error "Python not found. Install Python 3.10+ and try again."

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ required, found $PY_VERSION. Set PYTHON=/path/to/python3.10+ and retry."
fi
success "Python $PY_VERSION"

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-.venv}"

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR (skipping creation)"
else
    info "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
    success "Virtual environment created"
fi

# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
info "Upgrading pip ..."
pip install --upgrade pip --quiet

DEV_FLAG=""
if [[ "${1:-}" == "--dev" ]]; then
    DEV_FLAG="[dev]"
    info "Installing bud-rag with dev dependencies ..."
else
    info "Installing bud-rag (add --dev for test/lint tools) ..."
fi

pip install -e ".${DEV_FLAG}" --quiet
success "bud-rag installed  →  $(bud --version 2>&1 || echo '(version unavailable)')"

# ---------------------------------------------------------------------------
# Ollama reachability check (informational only)
# ---------------------------------------------------------------------------
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
if curl -s --connect-timeout 2 "${OLLAMA_URL}/api/tags" &>/dev/null; then
    success "Ollama is reachable at $OLLAMA_URL"
else
    warn "Ollama not detected at $OLLAMA_URL"
    echo "    Make sure Ollama is running before using bud."
    echo "    Install: https://ollama.com"
fi

# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "  1. Activate your environment:"
echo "       source ${VENV_DIR}/bin/activate"
echo ""
echo "  2. Configure bud (sets data/output dirs, LLM model, etc.):"
echo "       bud configure"
echo ""
echo "  3. Run the pipeline:"
echo "       bud process"
echo ""
echo "  Run 'bud --help' to see all commands."
echo ""
