#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Gemma 4 2B Inference Throughput Challenge — Orchestration
#
# Runs all stages inside Docker:
#   1. Build the container image
#   2. Generate synthetic multimodal data (prompts + assets)
#   3. Generate HF greedy reference outputs (canonical ground truth)
#   4. Copy eval prompts to /judge (kept separate from /data for isolation)
#   5. Run vLLM baseline, record timing
#   6. Run agent solution, record timing
#   7. Judge: verify outputs + compute score
#
# Usage:
#   ./run_challenge.sh              # Full pipeline
#   ./run_challenge.sh build        # Only build the image
#   ./run_challenge.sh data         # Only generate data
#   ./run_challenge.sh reference    # Only generate references
#   ./run_challenge.sh baseline     # Only run baseline
#   ./run_challenge.sh solution     # Only run agent solution
#   ./run_challenge.sh judge        # Only run judge
#   ./run_challenge.sh shell        # Drop into an interactive shell
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker/docker-compose.yml"

dc() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

run_service() {
    echo "=========================================="
    echo "  Running: $1"
    echo "=========================================="
    dc run --rm "$1"
}

step_build() {
    echo "Building Docker image..."
    dc build challenge
}

step_data() {
    run_service generate-data
}

step_reference() {
    run_service generate-references
}

step_copy_eval() {
    # Copy eval prompts into /judge so they're isolated from /data
    echo "Copying eval prompts to judge directory..."
    cp "$SCRIPT_DIR/data/eval_prompts.jsonl" "$SCRIPT_DIR/judge/eval_prompts.jsonl"
}

step_baseline() {
    step_copy_eval
    run_service run-baseline
}

step_solution() {
    step_copy_eval
    run_service run-solution
}

step_judge() {
    step_copy_eval
    run_service judge
}

step_shell() {
    dc run --rm challenge bash
}

step_full() {
    step_build
    step_data
    step_reference
    step_copy_eval
    step_baseline
    step_solution
    step_judge
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
STAGE="${1:-full}"

case "$STAGE" in
    build)     step_build ;;
    data)      step_data ;;
    reference) step_reference ;;
    baseline)  step_baseline ;;
    solution)  step_solution ;;
    judge)     step_judge ;;
    shell)     step_shell ;;
    full)      step_full ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Usage: $0 {build|data|reference|baseline|solution|judge|shell|full}"
        exit 1
        ;;
esac

echo ""
echo "Done: $STAGE"
