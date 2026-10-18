#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../scripts/runfile-common.sh" || exit 1
use_environment || exit 1
run_step 'PubMed ingestion' "$env_dir/bin/python" -m pubmed_rag.ingest \
    --save_dir data/ \
    --query_config configs/default_query.yaml \
    --timeout 60
exit "$?"
