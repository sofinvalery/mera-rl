#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/mera-rl}"
OUTPUTS_DIR="$REPO_DIR/outputs"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$OUTPUTS_DIR/full_rerun_${STAMP}}"

RUN_BASE_EVAL="${RUN_BASE_EVAL:-1}"
RUN_SFT="${RUN_SFT:-1}"
RUN_EVAL_SFT="${RUN_EVAL_SFT:-1}"
RUN_GRPO_ALL="${RUN_GRPO_ALL:-1}"

SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
DRY_CLEANUP="${DRY_CLEANUP:-0}"
CLEANUP_ONLY="${CLEANUP_ONLY:-0}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo directory not found: $REPO_DIR" >&2
  exit 1
fi

if [[ ! -x "$REPO_DIR/run_eval.sh" || ! -x "$REPO_DIR/run_sft_fast.sh" || ! -x "$REPO_DIR/run_eval_sft_intermediate.sh" || ! -x "$REPO_DIR/run_all_grpo.sh" ]]; then
  echo "Missing required runner scripts in $REPO_DIR" >&2
  exit 1
fi

cleanup_previous_artifacts() {
  mkdir -p "$OUTPUTS_DIR"
  shopt -s nullglob

  local candidates=()
  local p
  for p in \
    "$OUTPUTS_DIR/eval_"* \
    "$OUTPUTS_DIR/full_01_sft_prime_"* \
    "$OUTPUTS_DIR/full_02_eval_sft_"* \
    "$OUTPUTS_DIR/grpo_all_"* \
    "$OUTPUTS_DIR/grpo_prime" \
    "$OUTPUTS_DIR/full_rerun_"* \
    "$OUTPUTS_DIR/logs/eval_"*.log \
    "$OUTPUTS_DIR/logs/sft_"*.log \
    "$OUTPUTS_DIR/logs/grpo_"*.log; do
    [[ -e "$p" ]] || continue
    candidates+=("$p")
  done

  shopt -u nullglob

  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "Cleanup: nothing to remove under $OUTPUTS_DIR"
    return
  fi

  echo "Cleanup targets:"
  printf '  %s\n' "${candidates[@]}"

  if [[ "$DRY_CLEANUP" == "1" ]]; then
    echo "Cleanup dry-run only (DRY_CLEANUP=1)."
    return
  fi

  rm -rf "${candidates[@]}"
  echo "Cleanup complete."
}

if [[ "$SKIP_CLEANUP" != "1" ]]; then
  cleanup_previous_artifacts
else
  echo "Cleanup skipped (SKIP_CLEANUP=1)."
fi

if [[ "$CLEANUP_ONLY" == "1" ]]; then
  echo "Cleanup-only mode finished."
  exit 0
fi

BASE_EVAL_DIR="$RUN_ROOT/01_eval_base"
SFT_DIR="$RUN_ROOT/02_sft"
EVAL_SFT_DIR="$RUN_ROOT/03_eval_sft"
GRPO_ALL_DIR="$RUN_ROOT/04_grpo_all"

mkdir -p "$RUN_ROOT"

echo "Run root: $RUN_ROOT"
echo "Base model: $BASE_MODEL"

if [[ "$RUN_BASE_EVAL" == "1" ]]; then
  echo "[1/4] Base eval"
  "$REPO_DIR/run_eval.sh" "$BASE_MODEL" "$BASE_EVAL_DIR" "test" "benchmark"
fi

if [[ "$RUN_SFT" == "1" ]]; then
  echo "[2/4] SFT"
  OUT_DIR="$SFT_DIR" "$REPO_DIR/run_sft_fast.sh"
fi

if [[ "$RUN_EVAL_SFT" == "1" ]]; then
  echo "[3/4] Eval after SFT"
  SFT_RUN_DIR="$SFT_DIR" OUT_DIR="$EVAL_SFT_DIR" "$REPO_DIR/run_eval_sft_intermediate.sh"
fi

if [[ "$RUN_GRPO_ALL" == "1" ]]; then
  echo "[4/4] GRPO all tasks"
  RUN_ROOT="$GRPO_ALL_DIR" "$REPO_DIR/run_all_grpo.sh" "$SFT_DIR/latest"
fi

cat <<EOF
Done.
Artifacts:
  Base eval:    $BASE_EVAL_DIR
  SFT:          $SFT_DIR
  Eval after SFT: $EVAL_SFT_DIR
  GRPO all:     $GRPO_ALL_DIR
EOF
