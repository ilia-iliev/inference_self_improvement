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

usage() {
    cat >&2 <<'EOF'
Usage: run_challenge.sh <command>

Setup:
  preflight          Validate .env + HF cache layout. No docker.
  build              Build the base challenge image (heavy, one-time).
  data               Generate synthetic prompts + assets + HF greedy refs
                     for dev + eval sets.

Online loop:
  submit             Build solution/, evaluate, score, promote if >= 1.05x.
                     On first run, seeds solution/ from judge/baseline/ and
                     initializes baseline timing.

Agent:
  agent              Interactive bash shell in the sandbox (human/debug).
                     solution/ is rw; data/judge is not mounted.
  agent-run          LLM agent loop in the sandbox, non-interactive.
                     Talks to AGENT_LLM_URL.
  loop [N]           Host-side iterate: agent-run → submit, up to N (default 5).
                     Stops on promotion or submit-less agent exit.
  serve-up           Start the local LLM endpoint (serve/compose.yml).
  serve-down         Stop it.

Misc:
  shell              Interactive bash shell in the base image.

Env knobs (see judge/docker/.env.example):
  AGENT_LLM_GPU, AGENT_RUN_GPU, AGENT_LLM_URL, AGENT_LLM_MODEL
EOF
}

dc() {
    detect_compose
    "${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" "$@"
}

serve_dc() {
    detect_compose
    "${COMPOSE_CMD[@]}" -f "$SERVE_COMPOSE_FILE" "$@"
}

run() {
    preflight "$ENV_FILE"
    dc run --rm challenge "$@"
}

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
        --user "$(id -u):$(id -g)" \
        -v "$SCRIPT_DIR/solution":/workspace/solution \
        -v "$SCRIPT_DIR/data/agent":/workspace/data/agent:ro \
        -v "$SCRIPT_DIR/judge":/workspace/judge:ro \
        -v "$SCRIPT_DIR/PROMPT.md":/workspace/PROMPT.md:ro \
        -e MODEL_PATH="$MODEL_PATH" \
        -e HF_HUB_OFFLINE=1 \
        -e AGENT_LLM_URL="$AGENT_LLM_URL" \
        -e AGENT_LLM_MODEL="$AGENT_LLM_MODEL" \
        gemma4-agent:latest "$@"
}

case "${1:-}" in
    "")             usage; exit 1 ;;
    preflight)      preflight "$ENV_FILE" && echo "preflight OK: MODEL_PATH=$MODEL_PATH" ;;
    build)          preflight "$ENV_FILE"; dc build challenge ;;
    data)           run python3 /data/generate_prompts.py
                    run python3 /judge/baseline/inference.py /data/agent/dev_prompts.jsonl  /data/agent/dev_reference.jsonl
                    run python3 /judge/baseline/inference.py /data/judge/eval_prompts.jsonl /data/judge/eval_reference.jsonl ;;
    submit)         preflight "$ENV_FILE"
                    uv run "${UV_DEPS[@]}" python3 "$SUBMIT_PY" ;;
    agent)          preflight "$ENV_FILE"
                    agent_build
                    agent_docker_run "-it" ;;
    agent-run)      preflight "$ENV_FILE"
                    agent_build
                    agent_docker_run "" python3 /workspace/agent/runner.py \
                        2>&1 | tee "$SCRIPT_DIR/solution/agent_run.log" ;;
    loop)           preflight "$ENV_FILE"
                    iters="${2:-5}"
                    _snap="$SCRIPT_DIR/solution/.loop-snapshot"
                    for i in $(seq 1 "$iters"); do
                        echo "=== loop iter $i/$iters ==="

                        # Snapshot code files before agent runs so a failed/diverged
                        # run can be rolled back. Excludes dot-files, last_result.json,
                        # agent_run.log, and NOTES.md (notes accumulate across runs).
                        rm -rf "$_snap"
                        mkdir -p "$_snap"
                        find "$SCRIPT_DIR/solution" -maxdepth 1 -type f \
                            ! -name ".*" \
                            ! -name "last_result.json" \
                            ! -name "agent_run.log" \
                            ! -name "NOTES.md" \
                            -exec cp {} "$_snap/" \;

                        agent_build
                        agent_docker_run "" python3 /workspace/agent/runner.py \
                            2>&1 | tee "$SCRIPT_DIR/solution/agent_run.log" \
                            || echo "agent-run failed (iter $i), continuing"
                        marker="$SCRIPT_DIR/solution/.submit-request"
                        if [ ! -f "$marker" ]; then
                            echo "no submit marker for iter $i — restoring pre-run snapshot"
                            find "$_snap" -maxdepth 1 -type f -exec cp -f {} "$SCRIPT_DIR/solution/" \;
                            continue
                        fi
                        rm -f "$marker"
                        uv run "${UV_DEPS[@]}" python3 "$SUBMIT_PY" \
                            || echo "submit reported non-zero (continuing)"
                        # Append eval result to NOTES.md so the agent sees external outcomes.
                        if [ -f "$SCRIPT_DIR/solution/last_result.json" ]; then
                            _note="[eval iter $i] $(python3 -c "
import json, sys
r = json.load(open('$SCRIPT_DIR/solution/last_result.json'))
if r.get('error'):
    print(f'FAILED: {r[\"error\"]}')
elif r.get('promoted'):
    print(f'PROMOTED score={r[\"score\"]:.3f}')
else:
    print(f'score={r.get(\"score\",0):.3f} agent={r.get(\"agent_time\",\"?\"):.1f}s baseline={r.get(\"baseline_time\",\"?\"):.1f}s mismatched={r.get(\"num_mismatched\",\"?\")}')
")"
                            echo "$_note" >> "$SCRIPT_DIR/solution/NOTES.md"
                        fi
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
