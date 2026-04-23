#!/usr/bin/env bash
# Unit tests for judge/docker/lib.sh preflight logic. No docker, no network.
# Usage: bash tests/test_preflight.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=../judge/docker/lib.sh
. "$REPO_ROOT/judge/docker/lib.sh"

PASS=0
FAIL=0
fail_msgs=()

assert() {
    local desc="$1"; shift
    if "$@"; then
        PASS=$((PASS + 1))
        echo "  PASS: $desc"
    else
        FAIL=$((FAIL + 1))
        fail_msgs+=("$desc")
        echo "  FAIL: $desc"
    fi
}

# Runs a test function in-process so the global PASS/FAIL counters update.
# Each test is responsible for cleaning up its own env via trap.
run_test() {
    local name="$1"; shift
    echo "--- $name"
    "$@"
}

# ---- tests ----

test_missing_model_path_fails() {
    unset MODEL_PATH
    local out rc
    out=$(check_env_vars 2>&1); rc=$?
    assert "check_env_vars fails when MODEL_PATH unset" [ "$rc" -ne 0 ]
    assert "error mentions MODEL_PATH" grep -q "MODEL_PATH" <<<"$out"
}

test_model_path_missing_dir_fails() {
    export MODEL_PATH="/does/not/exist/model"
    local rc
    check_env_vars >/dev/null 2>&1; rc=$?
    assert "check_env_vars fails when MODEL_PATH dir missing" [ "$rc" -ne 0 ]
    unset MODEL_PATH
}

test_happy_path_absolute() {
    local tmp; tmp=$(mktemp -d)
    trap "rm -rf '$tmp'; unset MODEL_PATH" RETURN
    export MODEL_PATH="$tmp"
    local rc
    check_env_vars >/dev/null 2>&1; rc=$?
    assert "check_env_vars passes with existing MODEL_PATH directory" [ "$rc" -eq 0 ]
}

test_load_env_file_missing_is_noop() {
    unset MODEL_PATH
    load_env_file "/does/not/exist/.env"
    assert "load_env_file on missing path leaves MODEL_PATH unset" [ -z "${MODEL_PATH:-}" ]
}

test_load_env_file_sets_vars() {
    local tmp; tmp=$(mktemp -d)
    trap "rm -rf '$tmp'; unset MODEL_PATH" RETURN
    cat >"$tmp/.env" <<EOF
MODEL_PATH=/mnt/hdd2/google/gemma-4-E4B-it
EOF
    unset MODEL_PATH
    load_env_file "$tmp/.env"
    assert "load_env_file exports MODEL_PATH" [ "$MODEL_PATH" = "/mnt/hdd2/google/gemma-4-E4B-it" ]
}

test_preflight_rolls_it_all_up() {
    local tmp; tmp=$(mktemp -d)
    trap "rm -rf '$tmp'; unset MODEL_PATH" RETURN
    mkdir -p "$tmp/model"
    cat >"$tmp/.env" <<EOF
MODEL_PATH=$tmp/model
EOF
    unset MODEL_PATH
    local rc
    preflight "$tmp/.env" >/dev/null 2>&1; rc=$?
    assert "preflight succeeds with valid .env + model dir" [ "$rc" -eq 0 ]
    assert "preflight exports MODEL_PATH" [ "$MODEL_PATH" = "$tmp/model" ]
}

run_test "missing MODEL_PATH"         test_missing_model_path_fails
run_test "MODEL_PATH dir missing"     test_model_path_missing_dir_fails
run_test "happy path (absolute)"      test_happy_path_absolute
run_test "load_env_file missing"      test_load_env_file_missing_is_noop
run_test "load_env_file sets vars"    test_load_env_file_sets_vars
run_test "preflight end-to-end"       test_preflight_rolls_it_all_up

echo
echo "========================================"
echo "Passed: $PASS   Failed: $FAIL"
if [[ $FAIL -gt 0 ]]; then
    for m in "${fail_msgs[@]}"; do echo "  $m"; done
    exit 1
fi
