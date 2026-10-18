#!/usr/bin/env bash
set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../scripts/runfile-common.sh" || exit 1
use_environment "${1:-${RAG_ENV_DIR:-$repo_root/.data}}" || exit 1
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
    if ! run_step "$name" "$@"; then
        failed=1
    fi
}

run_check 'staged secret scan' bash scripts/check-secrets.sh --staged
run_check 'history secret scan' bash scripts/check-secrets.sh
run_check 'secret hook tests' bash scripts/test-secret-hooks.sh
run_check 'runfile rules and shell syntax' bash scripts/check-runfiles.sh
run_check 'Ruff lint' "$env_dir/bin/ruff" check
run_check 'Ruff format' "$env_dir/bin/ruff" format --check
run_check 'Pyright' "$env_dir/bin/pyright" --pythonpath "$env_dir/bin/python"
run_check 'offline pytest' "$env_dir/bin/pytest" -m 'not integration'

if [[ "$failed" -eq 0 ]]; then
    printf '\nAll local checks passed.\n'
else
    printf '\nLocal checks failed. See the results above.\n' >&2
fi
exit "$failed"
