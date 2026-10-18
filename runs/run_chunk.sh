#!/usr/bin/env bash
set -uo pipefail

export UV_PROJECT_ENVIRONMENT=.data
export DATA_DIR="data/"

printf "Running Embedding Generation\n"
time uv run python -m pubmed_rag.chunk \
    --corpus-path "$DATA_DIR/corpus_Oct2026.jsonl" \
    --embedding-path "$DATA_DIR/vectors/vector_embeddings.npz" \
    --batch-size 128 \
    --limit-abstracts -1