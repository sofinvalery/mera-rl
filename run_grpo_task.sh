#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <task> [base_model_path]" >&2
  exit 1
fi

REPO_DIR="/workspace/mera-rl"
PY="$REPO_DIR/_deps/prime-rl/.venv/bin/python"

TASK="$1"
shift
BASE_MODEL="$REPO_DIR/outputs/full_01_sft_prime_bs4x2_ga2_len1536/latest"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  BASE_MODEL="$1"
  shift
fi
CLI_EXTRA_ARGS=("$@")
OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/grpo_prime/${TASK}_$(date +%Y%m%d_%H%M%S)}"
ORCH_BATCH_SIZE="${ORCH_BATCH_SIZE:-32}"
ORCH_ROLLOUTS="${ORCH_ROLLOUTS:-8}"
ORCH_MAX_STEPS="${ORCH_MAX_STEPS:-400}"
MAX_TOKENS="${MAX_TOKENS:-192}"

cd "$REPO_DIR"
source "$REPO_DIR/env.sh"
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
if [[ "$TASK" == "rucodeeval" ]]; then
  EXTRA_ARGS+=(--allow-test-split)
fi

"$PY" "$REPO_DIR/mera/scripts/grpo.py" "$TASK" \
  --trainer-gpu-ids "[1]" \
  --inference-gpu-ids "[0]" \
  --trainer.model.name "$BASE_MODEL" \
  --orchestrator.model.name "$BASE_MODEL" \
  --inference.model.name "$BASE_MODEL" \
  --orchestrator.batch-size "$ORCH_BATCH_SIZE" \
  --orchestrator.rollouts-per-example "$ORCH_ROLLOUTS" \
  --orchestrator.max-steps "$ORCH_MAX_STEPS" \
  --orchestrator.sampling.max-tokens "$MAX_TOKENS" \
  --output-dir "$OUT_DIR" \
  "${EXTRA_ARGS[@]}" \
  "${CLI_EXTRA_ARGS[@]}"
