#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! cd "$repo_root"; then
    echo "Could not enter project directory: $repo_root" >&2
    exit 1
fi

rag="$repo_root/.data/bin/rag"
if [[ ! -x "$rag" ]]; then
    echo "Existing .data environment not found. Use the project's configured Python environment first." >&2
    exit 1
fi

mode="${1:-bm25}"
if [[ $# -gt 0 ]]; then
    shift
fi
query="${*:-HER2 breast cancer treatment}"

case "$mode" in
    bm25)
        "$rag" search "$query" --method bm25 --limit 5
        exit "$?"
        ;;
    index)
        "$rag" index
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
        if ! "$rag" index; then
            echo "Indexing failed. Check that Postgres is running and try again." >&2
            exit 1
        fi
        "$rag" search "$query" --method dense --limit 10 --allow-paid-query
        exit "$?"
        ;;
    *)
        echo "Usage: $0 [bm25|index|dense] [search query...]" >&2
        echo "Start Postgres first for index or dense mode: docker compose up -d db" >&2
        exit 2
        ;;
esac
