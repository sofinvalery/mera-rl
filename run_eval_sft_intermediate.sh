#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/mera-rl"
SFT_RUN_DIR="${SFT_RUN_DIR:-$REPO_DIR/outputs/full_01_sft_prime_bs4x2_ga2_len1536}"
MODEL_DIR="${MODEL_DIR:-}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/full_02_eval_sft_$(date +%Y%m%d_%H%M%S)}"
SPLIT="${SPLIT:-test}"
TASK_SET="${TASK_SET:-benchmark}"

resolve_latest_step_dir() {
  local run_dir="$1"
  local weights_dir="$run_dir/weights"
  if [[ ! -d "$weights_dir" ]]; then
    return 1
  fi

  local best_step=""
  local d
  for d in "$weights_dir"/step_*; do
    [[ -d "$d" ]] || continue
    [[ -f "$d/STABLE" ]] || continue
    local step="${d##*/step_}"
    if [[ "$step" =~ ^[0-9]+$ ]]; then
      if [[ -z "$best_step" || "$step" -gt "$best_step" ]]; then
        best_step="$step"
      fi
    fi
  done

  if [[ -n "$best_step" ]]; then
    echo "$weights_dir/step_$best_step"
    return 0
  fi
  return 1
}

if [[ -z "$MODEL_DIR" ]]; then
  if [[ -L "$SFT_RUN_DIR/latest" || -d "$SFT_RUN_DIR/latest" ]]; then
    MODEL_DIR="$SFT_RUN_DIR/latest"
  elif [[ -f "$SFT_RUN_DIR/LATEST_WEIGHT_STEP" ]]; then
    STEP="$(tr -d '[:space:]' < "$SFT_RUN_DIR/LATEST_WEIGHT_STEP")"
    MODEL_DIR="$SFT_RUN_DIR/weights/step_${STEP}"
  else
    MODEL_DIR="$(resolve_latest_step_dir "$SFT_RUN_DIR" || true)"
  fi
fi

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: ${MODEL_DIR:-<unset>}" >&2
  echo "Set MODEL_DIR, or point SFT_RUN_DIR to a Prime-RL SFT output containing weights/step_<N>." >&2
  exit 1
fi

echo "Resolved SFT model dir: $MODEL_DIR"

cd "$REPO_DIR"
exec "$REPO_DIR/run_eval.sh" "$MODEL_DIR" "$OUT_DIR" "$SPLIT" "$TASK_SET"
