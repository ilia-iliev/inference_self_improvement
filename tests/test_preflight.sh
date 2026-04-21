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

make_fake_cache() {
    # $1 = cache dir, $2 = repo id (e.g. google/gemma-3-1b-it)
    local dir="$1" repo="$2"
    local safe="${repo//\//--}"
    mkdir -p "$dir/hub/models--$safe/snapshots/deadbeef"
    mkdir -p "$dir/hub/models--$safe/blobs"
}

# ---- tests ----

test_missing_env_vars_fails() {
    unset HF_CACHE_DIR MODEL_PATH
    local out rc
    out=$(check_env_vars 2>&1); rc=$?
    assert "check_env_vars fails when unset" [ "$rc" -ne 0 ]
    assert "error mentions both vars" grep -q "HF_CACHE_DIR and MODEL_PATH" <<<"$out"
}

test_cache_dir_missing_fails() {
    export HF_CACHE_DIR="/does/not/exist/xyz"
    export MODEL_PATH="google/gemma-3-1b-it"
    local rc
    check_env_vars >/dev/null 2>&1; rc=$?
    assert "check_env_vars fails when cache dir missing" [ "$rc" -ne 0 ]
}

test_cache_dir_exists_but_model_missing() {
    local tmp; tmp=$(mktemp -d)
    trap "rm -rf '$tmp'" RETURN
    export HF_CACHE_DIR="$tmp"
    export MODEL_PATH="google/gemma-3-1b-it"
    local rc
    check_env_vars >/dev/null 2>&1; rc=$?
    assert "check_env_vars fails when cached snapshot missing" [ "$rc" -ne 0 ]
}

test_happy_path_repo_id() {
    local tmp; tmp=$(mktemp -d)
    trap "rm -rf '$tmp'" RETURN
    make_fake_cache "$tmp" "google/gemma-3-1b-it"
    export HF_CACHE_DIR="$tmp"
    export MODEL_PATH="google/gemma-3-1b-it"
    local rc
    check_env_vars >/dev/null 2>&1; rc=$?
    assert "check_env_vars passes with valid cache layout" [ "$rc" -eq 0 ]
}

test_absolute_path_model_skips_cache_check() {
    local tmp; tmp=$(mktemp -d)
    trap "rm -rf '$tmp'" RETURN
    export HF_CACHE_DIR="$tmp"
    export MODEL_PATH="/models/local"
    local rc
    check_env_vars >/dev/null 2>&1; rc=$?
    assert "check_env_vars accepts absolute path MODEL_PATH without cache check" [ "$rc" -eq 0 ]
}

test_load_env_file_missing_is_noop() {
    unset HF_CACHE_DIR MODEL_PATH
    load_env_file "/does/not/exist/.env"
    assert "load_env_file on missing path leaves vars unset" [ -z "${HF_CACHE_DIR:-}" ]
}

test_load_env_file_sets_vars() {
    local tmp; tmp=$(mktemp -d)
    trap "rm -rf '$tmp'" RETURN
    cat >"$tmp/.env" <<EOF
HF_CACHE_DIR=/some/cache
MODEL_PATH=org/name
TEXT_ONLY=1
EOF
    unset HF_CACHE_DIR MODEL_PATH TEXT_ONLY
    load_env_file "$tmp/.env"
    assert "load_env_file exports HF_CACHE_DIR" [ "$HF_CACHE_DIR" = "/some/cache" ]
    assert "load_env_file exports MODEL_PATH" [ "$MODEL_PATH" = "org/name" ]
    assert "load_env_file exports TEXT_ONLY" [ "$TEXT_ONLY" = "1" ]
}

test_preflight_rolls_it_all_up() {
    local tmp; tmp=$(mktemp -d)
    trap "rm -rf '$tmp'" RETURN
    make_fake_cache "$tmp/hfcache" "google/gemma-3-1b-it"
    cat >"$tmp/.env" <<EOF
HF_CACHE_DIR=$tmp/hfcache
MODEL_PATH=google/gemma-3-1b-it
TEXT_ONLY=1
EOF
    unset HF_CACHE_DIR MODEL_PATH TEXT_ONLY
    local rc
    preflight "$tmp/.env" >/dev/null 2>&1; rc=$?
    assert "preflight succeeds with valid .env + cache" [ "$rc" -eq 0 ]
    assert "preflight exports HF_CACHE_DIR" [ "$HF_CACHE_DIR" = "$tmp/hfcache" ]
    assert "preflight exports TEXT_ONLY" [ "$TEXT_ONLY" = "1" ]
}

run_test "missing env vars"           test_missing_env_vars_fails
run_test "cache dir missing"          test_cache_dir_missing_fails
run_test "model missing from cache"   test_cache_dir_exists_but_model_missing
run_test "happy path (repo id)"       test_happy_path_repo_id
run_test "absolute path MODEL_PATH"   test_absolute_path_model_skips_cache_check
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
