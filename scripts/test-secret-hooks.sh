#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
fixture="$tmp_dir/repo"
mkdir -p "$fixture/scripts" "$fixture/.githooks" "$fixture/.tools/bin"
cp "$repo_root/scripts/check-secrets.sh" "$fixture/scripts/"
cp "$repo_root/.githooks/pre-commit" "$repo_root/.githooks/pre-push" "$fixture/.githooks/"
ln -s "$repo_root/.tools/bin/gitleaks" "$fixture/.tools/bin/gitleaks"
cd "$fixture"
git init --quiet
git config user.name 'Secret scan test'
git config user.email 'secret-scan@example.test'
git config core.hooksPath .githooks
printf 'A clean file\n' > clean.txt
git add clean.txt
git commit --quiet -m 'Clean fixture'
clean_oid=$(git rev-parse HEAD)
zero_oid=$(printf '%040d' 0)
printf 'refs/heads/main %s refs/heads/main %s\n' "$clean_oid" "$zero_oid" | bash .githooks/pre-push

# This token is generated locally for the test and is not a real credential.
printf 'token = "%s%s"\n' 'ghp_' '81b7d90e5f3a6240c9d8b701fe4a3c65209de8ab' > secret.txt
git add secret.txt
if bash .githooks/pre-commit > "$tmp_dir/scan.log" 2>&1; then
    echo 'FAIL: pre-commit accepted a synthetic secret' >&2
    exit 1
fi
# Bypass only inside the disposable fixture to test the pre-push hook.
git -c core.hooksPath=/dev/null commit --quiet -m 'Synthetic secret fixture'
secret_oid=$(git rev-parse HEAD)
if printf 'refs/heads/main %s refs/heads/main %s\n' "$secret_oid" "$clean_oid" | bash .githooks/pre-push > "$tmp_dir/scan.log" 2>&1; then
    echo 'FAIL: pre-push accepted a synthetic secret' >&2
    exit 1
fi
if ! grep -q REDACTED "$tmp_dir/scan.log"; then
    echo 'FAIL: synthetic secret was not reported with redaction' >&2
    exit 1
fi
if grep -q '81b7d90e5f3a6240c9d8b701fe4a3c65209de8ab' "$tmp_dir/scan.log"; then
    echo 'FAIL: scanner exposed the synthetic secret in its output' >&2
    exit 1
fi
printf 'PASS: clean commits pass; synthetic secrets block commits and pushes with redacted output\n'
