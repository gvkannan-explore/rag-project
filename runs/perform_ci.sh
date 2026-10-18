#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
env_dir="${1:-$repo_root/.data}"
if [[ ! -d "$env_dir" ]]; then
    echo "Existing Python environment not found: $env_dir" >&2
    echo 'Usage: bash runs/perform_ci.sh [existing-environment-path]' >&2
    exit 1
fi
env_dir=$(cd "$env_dir" && pwd)
for tool in python ruff pyright pytest; do
    if [[ ! -x "$env_dir/bin/$tool" ]]; then
        echo "Required tool not found: $env_dir/bin/$tool" >&2
        exit 1
    fi
done

failed=0
run_check() {
    local name="$1"
    shift
    printf '\nRunning %s\n' "$name"
    if ! "$@"; then
        printf 'FAILED: %s\n' "$name" >&2
        failed=1
    fi
}

run_check 'staged secret scan' bash scripts/check-secrets.sh --staged
run_check 'history secret scan' bash scripts/check-secrets.sh
run_check 'secret hook tests' bash scripts/test-secret-hooks.sh
run_check 'Ruff lint' "$env_dir/bin/ruff" check
run_check 'Ruff format' "$env_dir/bin/ruff" format --check
run_check 'Pyright' "$env_dir/bin/pyright" --pythonpath "$env_dir/bin/python"
run_check 'pytest' "$env_dir/bin/pytest"

if [[ "$failed" -eq 0 ]]; then
    printf '\nAll local checks passed.\n'
else
    printf '\nLocal checks failed. See the results above.\n' >&2
fi
exit "$failed"
