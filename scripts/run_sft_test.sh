#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_sft_test.sh [extra run_sft_local.py args...]

Runs a tiny MERA SFT smoke test on a small task subset.

Environment overrides:
  EXPERIMENT            Experiment name. Default: mera_sft_test
  OUTPUT_DIR            Output directory. Default: outputs/runs/<experiment>/sft-test
  MODEL                 Base model for smoke SFT. Default: Qwen/Qwen3-0.6B
  TASKS                 Space-separated tasks. Default: "chegeka mathlogicqa rcb"
  LIMIT                 Per-task example cap. Default: 16
  MAX_STEPS             Default: 8
  SFT_CONFIG            SFT config file. Default: configs/sft/mera-smoke-no-lora.toml
  MAX_SEQ_LEN           Default: 1024
  BATCH_SIZE            Default: 2
  GRAD_ACCUM            Default: 1
  MICRO_BATCH_SIZE      Default: 1
  WANDB_PROJECT         Optional W&B project. Default: mera
  WANDB_ENTITY          Optional W&B entity
  WANDB_RUN_NAME        Optional W&B run name. Default: sft-test-<experiment>
  HF_ADAPTER_REPO_ID    Optional HF repo to upload the final adapter
  HF_MERGED_REPO_ID     Optional HF repo to upload a merged handoff model

Examples:
  scripts/run_sft_test.sh
  LIMIT=8 MAX_STEPS=4 scripts/run_sft_test.sh
  TASKS="chegeka rcb" scripts/run_sft_test.sh --dry-run
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi
EXPERIMENT="${EXPERIMENT:-mera_sft_test}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/runs/$EXPERIMENT/sft-test}"
MANIFEST_PATH="${MANIFEST_PATH:-$REPO_DIR/outputs/runs/$EXPERIMENT/manifest.json}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
TASKS="${TASKS:-chegeka mathlogicqa rcb}"
LIMIT="${LIMIT:-16}"
MAX_STEPS="${MAX_STEPS:-8}"
SFT_CONFIG="${SFT_CONFIG:-$REPO_DIR/configs/sft/mera-smoke-no-lora.toml}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-mera}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sft-test-$EXPERIMENT}"

CMD=(
  "$PYTHON_BIN" "$REPO_DIR/scripts/run_sft_local.py"
  --config "$SFT_CONFIG"
  --output-dir "$OUTPUT_DIR"
  --manifest "$MANIFEST_PATH"
  --experiment "$EXPERIMENT"
  --model "$MODEL"
  --tasks
)

for task in $TASKS; do
  CMD+=("$task")
done

CMD+=(
  --limit "$LIMIT"
  --max-steps "$MAX_STEPS"
  --max-seq-len "$MAX_SEQ_LEN"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --no-lora
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$WANDB_RUN_NAME"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  CMD+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "${HF_ADAPTER_REPO_ID:-}" ]]; then
  CMD+=(--hf-adapter-repo-id "$HF_ADAPTER_REPO_ID")
fi
if [[ -n "${HF_MERGED_REPO_ID:-}" ]]; then
  CMD+=(--hf-merged-repo-id "$HF_MERGED_REPO_ID")
fi

CMD+=("$@")

exec "${CMD[@]}"
