#!/usr/bin/env bash
set -uo pipefail

export UV_PROJECT_ENVIRONMENT=.data
export DATA_DIR="data/"
export CONFIG_DIR="configs/default_query.yaml"

printf "Running Data ingestion from NCBI\n"
time uv run python -m pubmed_rag.ingest \
    --save_dir "$DATA_DIR" \
    --query_config "$CONFIG_DIR" \
    --timeout 60
