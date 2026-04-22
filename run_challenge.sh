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
  agent           Drop the LLM agent into its sandbox (see tools/README.md).
                  /workspace/solution is rw; data/judge is not mounted.

Misc:
  shell           Interactive bash shell in the base image.
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
                    docker build -t gemma4-agent:latest "$SCRIPT_DIR/tools"
                    docker run -it --rm --gpus all \
                        -v "$SCRIPT_DIR/solution":/workspace/solution \
                        -v "$SCRIPT_DIR/data/agent":/workspace/data/agent:ro \
                        -v "$SCRIPT_DIR/judge":/workspace/judge:ro \
                        -v "$SCRIPT_DIR/PROMPT.md":/workspace/PROMPT.md:ro \
                        -v "$HF_CACHE_DIR":/root/.cache/huggingface:ro \
                        -e MODEL_PATH="$MODEL_PATH" \
                        -e TEXT_ONLY="${TEXT_ONLY:-1}" \
                        -e HF_HUB_OFFLINE=1 \
                        gemma4-agent:latest ;;
    shell)          preflight "$ENV_FILE"; dc run --rm challenge bash ;;
    *)              usage; exit 1 ;;
esac
