#!/usr/bin/env bash
# Gemma 4 2B Inference Throughput Challenge — orchestration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="$SCRIPT_DIR/docker/docker-compose.yml"

dc()  { docker compose -f "$COMPOSE" "$@"; }
run() { dc run --rm challenge "$@"; }
copy_eval() { cp "$SCRIPT_DIR/data/eval_prompts.jsonl" "$SCRIPT_DIR/judge/eval_prompts.jsonl"; }

case "${1:-full}" in
    build)     dc build challenge ;;
    data)      run python3 /data/generate_prompts.py ;;
    reference) run python3 /baseline/inference.py /data/dev_prompts.jsonl  /data/dev_reference.jsonl
               run python3 /baseline/inference.py /data/eval_prompts.jsonl /data/eval_reference.jsonl ;;
    baseline)  copy_eval
               run python3 /baseline/inference.py /judge/eval_prompts.jsonl /judge/baseline_outputs.jsonl /judge/baseline_timing.json ;;
    solution)  copy_eval
               run python3 /solution/run.py /judge/eval_prompts.jsonl ;;
    judge)     copy_eval
               run python3 /judge/judge.py ;;
    shell)     dc run --rm challenge bash ;;
    full)      for s in build data reference baseline solution judge; do "$0" "$s"; done ;;
    *)         echo "Usage: $0 {build|data|reference|baseline|solution|judge|shell|full}"; exit 1 ;;
esac
