# Sourceable helpers for run_challenge.sh. Pure shell, no docker side effects.
# Each function takes inputs as arguments and/or env vars (no hidden globals).

# Find whichever Compose CLI is available. On success exports COMPOSE_CMD
# as a bash array. On failure prints to stderr and returns non-zero.
detect_compose() {
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD=(docker compose)
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE_CMD=(docker-compose)
    else
        echo "ERROR: neither 'docker compose' (plugin) nor 'docker-compose' (standalone) is installed." >&2
        echo "Install one, e.g. on Ubuntu: sudo apt install docker-compose-plugin" >&2
        return 1
    fi
}

# Source the given env file if it exists. Safe to call with a nonexistent path.
load_env_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        set -a; . "$path"; set +a
    fi
}

# Validate that HF_CACHE_DIR and MODEL_PATH are set and sensible.
# If MODEL_PATH looks like an HF repo id (org/name with no leading slash),
# check that the cache contains its snapshot directory.
check_env_vars() {
    local env_file_hint="${1:-judge/docker/.env}"
    if [[ -z "${HF_CACHE_DIR:-}" || -z "${MODEL_PATH:-}" ]]; then
        echo "ERROR: HF_CACHE_DIR and MODEL_PATH must both be set." >&2
        echo "Copy judge/docker/.env.example to $env_file_hint and edit." >&2
        return 1
    fi
    if [[ ! -d "$HF_CACHE_DIR" ]]; then
        echo "ERROR: HF_CACHE_DIR=$HF_CACHE_DIR is not an existing directory." >&2
        return 1
    fi
    if [[ "$MODEL_PATH" == */* && "$MODEL_PATH" != /* ]]; then
        local cache_subdir="$HF_CACHE_DIR/hub/models--${MODEL_PATH//\//--}"
        if [[ ! -d "$cache_subdir" ]]; then
            echo "ERROR: expected cached model at $cache_subdir (derived from MODEL_PATH=$MODEL_PATH), but it does not exist." >&2
            echo "Either download via huggingface-cli or fix MODEL_PATH/HF_CACHE_DIR." >&2
            return 1
        fi
    fi
}

# Full preflight: load env file then validate. Exports the resulting env vars
# so docker compose picks them up. Returns non-zero on any failure.
preflight() {
    local env_file="$1"
    load_env_file "$env_file"
    check_env_vars "$env_file" || return 1
    export HF_CACHE_DIR MODEL_PATH TEXT_ONLY
}
