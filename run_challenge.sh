#!/usr/bin/env bash
# Inference throughput challenge — orchestration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/judge/docker/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/judge/docker/.env"

# shellcheck source=judge/docker/lib.sh
. "$SCRIPT_DIR/judge/docker/lib.sh"

SUBMIT_PY="$SCRIPT_DIR/judge/submit.py"
UV_DEPS=(--with httpx --with transformers --with sentencepiece)

SERVE_COMPOSE_FILE="$SCRIPT_DIR/serve/compose.yml"
AGENT_RUN_GPU="${AGENT_RUN_GPU:-1}"
AGENT_LLM_URL="${AGENT_LLM_URL:-http://localhost:${AGENT_LLM_PORT:-8001}/v1}"
AGENT_LLM_MODEL="${AGENT_LLM_MODEL:-google/gemma-4-e4b-it}"

agent_build() {
    docker build -t gemma4-agent:latest "$SCRIPT_DIR/tools"
}

agent_docker_run() {
    # $1 = extra docker-run flags (e.g. -it), $@[2..] = command
    local extra_flags="$1"; shift
    # shellcheck disable=SC2086  # extra_flags is intentionally word-split
    docker run --rm $extra_flags \
        --network host \
        --gpus "\"device=${AGENT_RUN_GPU}\"" \
        -v "$SCRIPT_DIR/solution":/workspace/solution \
        -v "$SCRIPT_DIR/data/agent":/workspace/data/agent:ro \
        -v "$SCRIPT_DIR/judge":/workspace/judge:ro \
        -v "$SCRIPT_DIR/PROMPT.md":/workspace/PROMPT.md:ro \
        -v "$HF_CACHE_DIR":/root/.cache/huggingface:ro \
        -e MODEL_PATH="$MODEL_PATH" \
        -e TEXT_ONLY="${TEXT_ONLY:-1}" \
        -e HF_HUB_OFFLINE=1 \
        -e AGENT_LLM_URL="$AGENT_LLM_URL" \
        -e AGENT_LLM_MODEL="$AGENT_LLM_MODEL" \
        gemma4-agent:latest "$@"
}

serve_dc() {
    detect_compose
    "${COMPOSE_CMD[@]}" -f "$SERVE_COMPOSE_FILE" "$@"
}

usage() {
    cat >&2 <<'EOF'
Usage: run_challenge.sh <command>

Setup:
  preflight       Validate .env + HF cache layout. No docker invoked.
  build           Build the base challenge image (heavy, one-time).
  data            Generate synthetic prompts + assets, then HF greedy references
                  for dev + eval sets.

Online loop:
  submit          Build /solution/, evaluate on all prompts, score, promote if >= 1.05x.
                  Auto-initializes baseline on first run (delete judge/baseline.json to force).
  dev [N]         Build /solution/, run on first N dev prompts (default 10),
                  verify tokens. Fast iteration; no baseline compare.

Agent:
  agent           Drop into an interactive bash shell in the sandbox (human/debug).
                  /workspace/solution is rw; data/judge is not mounted.
  agent-run       Run the LLM agent loop non-interactively in the sandbox.
                  Talks to AGENT_LLM_URL (default http://localhost:8001/v1).
                  Exits on `submit` tool call or when the model stops.
  loop [N]        Host-side iterate: agent-run → submit, up to N times (default 5).
                  Stops early on promotion or submit-less agent exit.
  serve-up        Start the local LLM endpoint (serve/compose.yml, GPU AGENT_LLM_GPU).
  serve-down      Stop it.

Misc:
  shell           Interactive bash shell in the base image.

Env knobs:
  AGENT_LLM_GPU   (default 0)    GPU for the LLM serving container
  AGENT_RUN_GPU   (default 1)    GPU exposed to the agent sandbox
  AGENT_LLM_URL   (default http://localhost:8001/v1)
  AGENT_LLM_MODEL (default google/gemma-4-e4b-it)
EOF
}

dc()  {
    detect_compose
    "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" "$@"
}
run() {
    preflight "$ENV_FILE"
    dc run --rm challenge "$@"
}

case "${1:-}" in
    "")             usage; exit 1 ;;
    preflight)      preflight "$ENV_FILE" && echo "preflight OK: HF_CACHE_DIR=$HF_CACHE_DIR MODEL_PATH=$MODEL_PATH" ;;
    build)          preflight "$ENV_FILE"; dc build challenge ;;
    data)           run python3 /data/generate_prompts.py
                    run python3 /judge/baseline/inference.py /data/agent/dev_prompts.jsonl  /data/agent/dev_reference.jsonl
                    run python3 /judge/baseline/inference.py /data/judge/eval_prompts.jsonl /data/judge/eval_reference.jsonl ;;
    submit)         preflight "$ENV_FILE"
                    uv run "${UV_DEPS[@]}" python3 "$SUBMIT_PY" submit ;;
    dev)            preflight "$ENV_FILE"
                    shift
                    uv run "${UV_DEPS[@]}" python3 "$SUBMIT_PY" dev ${1:+-n "$1"} ;;
    agent)          preflight "$ENV_FILE"
                    agent_build
                    agent_docker_run "-it" ;;
    agent-run)      preflight "$ENV_FILE"
                    agent_build
                    agent_docker_run "" python3 /workspace/agent/runner.py ;;
    loop)           preflight "$ENV_FILE"
                    agent_build
                    iters="${2:-5}"
                    for i in $(seq 1 "$iters"); do
                        echo "=== loop iter $i/$iters ==="
                        agent_docker_run "" python3 /workspace/agent/runner.py \
                            || { echo "agent-run failed (iter $i)"; exit 1; }
                        marker="$SCRIPT_DIR/solution/.submit-request"
                        if [ ! -f "$marker" ]; then
                            echo "agent exited without calling submit; stopping loop."
                            exit 0
                        fi
                        rm -f "$marker"
                        uv run "${UV_DEPS[@]}" python3 "$SUBMIT_PY" submit \
                            || echo "submit reported non-zero (continuing)"
                        if python3 -c "import json,sys; sys.exit(0 if json.load(open('$SCRIPT_DIR/solution/last_result.json')).get('promoted') else 1)"; then
                            echo "solution promoted — stopping loop."
                            exit 0
                        fi
                    done ;;
    serve-up)       preflight "$ENV_FILE"; serve_dc up -d ;;
    serve-down)     serve_dc down ;;
    shell)          preflight "$ENV_FILE"; dc run --rm challenge bash ;;
    *)              usage; exit 1 ;;
esac
