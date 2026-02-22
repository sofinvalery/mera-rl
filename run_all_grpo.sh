#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/rl/mera-rl"
PY="$REPO_DIR/_deps/prime-rl/.venv/bin/python"
BASE_MODEL="${1:-$REPO_DIR/outputs/full_01_sft/merged}"

# Full task set supported by mera/scripts/grpo.py
TASKS=(
  chegeka
  lcs
  mamuramu
  mathlogicqa
  multiq
  parus
  rcb
  rucodeeval
  rumodar
  rumultiar
  ruopenbookqa
  rutie
  ruworldtree
  rwsd
  use
)

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$REPO_DIR/outputs/grpo_all_${STAMP}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PY" ]]; then
  echo "Prime-RL Python not found at: $PY" >&2
  exit 1
fi

if [[ ! -d "$BASE_MODEL" ]]; then
  echo "Base model path not found: $BASE_MODEL" >&2
  exit 1
fi

if [[ ! -f "$REPO_DIR/env.sh" ]]; then
  echo "Missing env file: $REPO_DIR/env.sh" >&2
  exit 1
fi

cd "$REPO_DIR"
source "$REPO_DIR/env.sh"
export CUDA_VISIBLE_DEVICES=0,1

echo "BASE_MODEL=$BASE_MODEL"
echo "RUN_ROOT=$RUN_ROOT"

for TASK in "${TASKS[@]}"; do
  OUT_DIR="$RUN_ROOT/$TASK"
  LOG_FILE="$LOG_DIR/${TASK}.log"

  EXTRA_ARGS=()
  # rucodeeval uses non-fair split in current config.
  if [[ "$TASK" == "rucodeeval" ]]; then
    EXTRA_ARGS+=(--allow-test-split)
  fi

  echo "=== [$TASK] starting ==="
  set -x
  "$PY" "$REPO_DIR/mera/scripts/grpo.py" "$TASK" \
    --trainer-gpu-ids "[1]" \
    --inference-gpu-ids "[0]" \
    --trainer.model.name "$BASE_MODEL" \
    --orchestrator.model.name "$BASE_MODEL" \
    --inference.model.name "$BASE_MODEL" \
    --orchestrator.batch_size 16 \
    --orchestrator.rollouts_per_example 8 \
    --orchestrator.max_steps 400 \
    --sampling.max_tokens 192 \
    --output-dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
  set +x
  echo "=== [$TASK] done ==="
done

echo "All tasks complete. Outputs: $RUN_ROOT"
