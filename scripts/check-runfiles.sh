#!/usr/bin/env bash
set -uo pipefail

if ! repo_root=$(git rev-parse --show-toplevel); then
    exit 1
fi
failed=0
for script in "$repo_root"/runs/*.sh; do
    if ! bash -n "$script"; then
        failed=1
    fi
    if ! awk '
        $1 == "set" {
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^-[a-zA-Z]*e/ || ($i == "-o" && $(i+1) == "errexit")) {
                    printf "%s:%d: runfiles must handle failures explicitly instead of set -e\n", FILENAME, NR
                    failed = 1
                }
            }
        }
        END {exit failed}
    ' "$script"; then
        failed=1
    fi
done
for script in "$repo_root"/scripts/*.sh "$repo_root"/.githooks/*; do
    if ! bash -n "$script"; then
        failed=1
    fi
done
exit "$failed"
