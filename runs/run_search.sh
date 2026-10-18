#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../scripts/runfile-common.sh" || exit 1
use_environment || exit 1
rag="$env_dir/bin/rag"
if [[ ! -x "$rag" ]]; then
    echo "rag command not found in the chosen environment: $env_dir" >&2
    exit 1
fi

mode="${1:-bm25}"
if [[ $# -gt 0 ]]; then
    shift
fi
query="${*:-HER2 breast cancer treatment}"

case "$mode" in
    bm25)
        run_step 'BM25 retrieval' "$rag" search "$query" --method bm25 --limit 5
        exit "$?"
        ;;
    index)
        run_step 'Cached-vector indexing' "$rag" index
        exit "$?"
        ;;
    dense)
        if [[ "${ALLOW_PAID_QUERY:-}" != "1" ]]; then
            echo "Dense mode makes one paid question-embedding request. Set ALLOW_PAID_QUERY=1 to continue." >&2
            exit 2
        fi
        if [[ -z "${OPENAI_API_KEY:-}" ]]; then
            echo "Set OPENAI_API_KEY before dense search." >&2
            exit 2
        fi
        if [[ -z "${QUERY_BUDGET_USD:-}" ]]; then
            echo 'Set QUERY_BUDGET_USD to the agreed spending ceiling for this query.' >&2
            exit 2
        fi
        if ! run_step 'Cached-vector indexing' "$rag" index; then
            echo "Indexing failed. Check that Postgres is running and try again." >&2
            exit 1
        fi
        run_step 'Dense retrieval' "$rag" search "$query" --method dense --limit 10 \
            --allow-paid-query --query-budget-usd "$QUERY_BUDGET_USD"
        exit "$?"
        ;;
    *)
        echo "Usage: $0 [bm25|index|dense] [search query...]" >&2
        echo "Start Postgres first for index or dense mode: docker compose up -d db" >&2
        exit 2
        ;;
esac
