#!/usr/bin/env bash

runfile_finish() {
    local status=$?
    trap - EXIT
    if [[ "$status" -ne 0 ]]; then
        printf '\nRunfile failed with exit status %s. See the error above.\n' "$status" >&2
    fi
    if [[ -t 0 && -t 1 && "${RAG_NONINTERACTIVE:-0}" != "1" ]]; then
        printf '\nPress Enter to close this runfile. '
        read -r _runfile_answer || true
    fi
    exit "$status"
}
trap runfile_finish EXIT

if ! repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); then
    echo 'Could not locate the project directory.' >&2
    exit 1
fi
if ! cd "$repo_root"; then
    echo "Could not enter project directory: $repo_root" >&2
    exit 1
fi

use_environment() {
    local requested="${1:-${RAG_ENV_DIR:-$repo_root/.data}}"
    if [[ ! -x "$requested/bin/python" ]]; then
        echo "Existing Python environment not found: $requested" >&2
        echo 'Set RAG_ENV_DIR to the existing environment you chose.' >&2
        return 1
    fi
    if ! env_dir=$(cd "$requested" && pwd); then
        return 1
    fi
    export UV_PROJECT_ENVIRONMENT="$env_dir"
}

run_step() {
    local name="$1"
    local status
    shift
    printf '\nRunning %s\n' "$name"
    if "$@"; then
        return 0
    else
        status=$?
        printf 'FAILED: %s, exit status %s\n' "$name" "$status" >&2
        return "$status"
    fi
}
