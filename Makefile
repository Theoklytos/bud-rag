# Makefile — common development tasks for bud-rag
.PHONY: install dev test clean distclean check fmt help

VENV     := .venv
PYTHON   := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

install: ## Bootstrap venv and install bud-rag (runtime deps only)
	bash install.sh

dev: $(VENV) ## Install bud-rag with dev dependencies (pytest, etc.)
	$(PIP) install -e ".[dev]" --quiet
	@echo "✓ Dev dependencies installed"

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test: $(VENV) ## Run the full test suite
	$(PYTEST) tests/ -v

test-fast: $(VENV) ## Run tests without verbose output
	$(PYTEST) tests/ -q

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean: ## Remove Python bytecode and build artefacts
	find . -type d -name __pycache__ ! -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" ! -path './.venv/*' -delete
	rm -rf *.egg-info dist build .pytest_cache

distclean: clean ## clean + remove the virtual environment
	rm -rf $(VENV)

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""

.DEFAULT_GOAL := help
