#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/mera-rl"
PY="$REPO_DIR/_deps/prime-rl/.venv/bin/python"

MODEL_PATH="${1:-Qwen/Qwen3-4B-Instruct-2507}"
OUT_DIR="${2:-$REPO_DIR/outputs/eval_$(date +%Y%m%d_%H%M%S)}"
SPLIT="${3:-test}"
TASK_SET="${4:-benchmark}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

cd "$REPO_DIR"
source "$REPO_DIR/env.sh"
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$REPO_DIR/outputs/logs"

"$PY" "$REPO_DIR/mera/scripts/eval.py" \
  --model "$MODEL_PATH" \
  --task-set "$TASK_SET" \
  --split "$SPLIT" \
  --tensor-parallel 2 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$REPO_DIR/outputs/logs/eval_$(date +%Y%m%d_%H%M%S).log"
