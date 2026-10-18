#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../scripts/runfile-common.sh" || exit 1
use_environment || exit 1
run_step 'Document embedding generation' "$env_dir/bin/python" -m pubmed_rag.chunk \
    --corpus-path data/corpus_Oct2026.jsonl \
    --embedding-path data/vectors/vector_embeddings.npz \
    --batch-size 128 \
    --limit-abstracts -1
exit "$?"
