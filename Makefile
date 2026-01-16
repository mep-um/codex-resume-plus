.DEFAULT_GOAL := help

# Prefer the repo-local Poetry (created by our bootstrap step) when present so
# contributors don't need a global Poetry install. Override with:
#   make POETRY=poetry check
POETRY ?= $(shell if [ -x .bootstrap/bin/poetry ]; then echo .bootstrap/bin/poetry; else echo poetry; fi)
RUN := $(POETRY) run

.PHONY: help check lint format-check format type test

help:
	@echo "Targets:"
	@echo "  make check         Run formatting check, lint, typecheck, and tests"
	@echo "  make lint          Run ruff"
	@echo "  make format-check  Verify formatting via black --check"
	@echo "  make format        Auto-format via black"
	@echo "  make type          Run pyright"
	@echo "  make test          Run pytest"

check: format-check lint type test

lint:
	@$(RUN) ruff check .

format-check:
	@$(RUN) black --check scripts tests

format:
	@$(RUN) black scripts tests

type:
	@$(RUN) pyright

test:
	@$(RUN) pytest -q
