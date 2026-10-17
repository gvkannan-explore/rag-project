#!/usr/bin/env bash
# set -euo pipefail

# Run from the project root: bash runs/perform_ci.sh
export UV_PROJECT_ENVIRONMENT=.data

printf '\nRunning Ruff\n'
uv run ruff check

printf '\nRunning Pyright\n'
uv run pyright

printf '\nRunning pytest\n'
uv run pytest
