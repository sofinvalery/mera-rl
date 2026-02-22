#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/mera-rl"
PY="$REPO_DIR/_deps/prime-rl/.venv/bin/python"

cd "$REPO_DIR"
source "$REPO_DIR/env.sh"

export WANDB_PROJECT="${WANDB_PROJECT:-rl}"
export WANDB_ENTITY="${WANDB_ENTITY:-sofinvalery}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "$PY" ]]; then
  echo "Prime-RL python not found at $PY" >&2
  exit 1
fi

count_visible_gpus() {
  local devices="$1"
  if [[ -z "$devices" ]]; then
    echo 1
    return
  fi
  IFS=',' read -r -a ids <<< "$devices"
  local n=0
  for id in "${ids[@]}"; do
    [[ -n "${id// }" ]] && ((n += 1))
  done
  if (( n < 1 )); then
    n=1
  fi
  echo "$n"
}

NPROC_PER_NODE="${NPROC_PER_NODE:-$(count_visible_gpus "$CUDA_VISIBLE_DEVICES")}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1536}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
LR="${LR:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-200}"
SEED="${SEED:-42}"
USE_LORA="${USE_LORA:-0}"
RUN_TAG="bs${BATCH_SIZE}x${NPROC_PER_NODE}_ga${GRAD_ACCUM}_len${MAX_SEQ_LEN}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/full_01_sft_prime_${RUN_TAG}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sft_qwen3_4b_instruct_prime_${RUN_TAG}}"
LOG_DIR="$REPO_DIR/outputs/logs"
mkdir -p "$LOG_DIR"

STEP_ARGS=(--epochs "$EPOCHS")
if [[ -n "$MAX_STEPS" ]]; then
  STEP_ARGS=(--max-steps "$MAX_STEPS")
fi

LORA_ARGS=()
if [[ "$USE_LORA" == "1" ]]; then
  LORA_ARGS+=(--use-lora)
fi

"$PY" "$REPO_DIR/mera/scripts/sft.py" \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --tasks chegeka mamuramu mathlogicqa multiq parus rcb rumultiar ruopenbookqa rutie ruworldtree rwsd use \
  "${STEP_ARGS[@]}" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --warmup-ratio "$WARMUP_RATIO" \
  --save-steps "$SAVE_STEPS" \
  --seed "$SEED" \
  --nproc-per-node "$NPROC_PER_NODE" \
  "${LORA_ARGS[@]}" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-run-name "$WANDB_RUN_NAME" \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$LOG_DIR/${WANDB_RUN_NAME}.log"
