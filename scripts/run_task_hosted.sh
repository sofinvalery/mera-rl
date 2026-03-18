#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_task_hosted.sh <task> [owner] [prime-rl-run-args...]

Examples:
  scripts/run_task_hosted.sh mathlogicqa my-team
  PRIME_ENV_OWNER=my-team scripts/run_task_hosted.sh lcs

Environment overrides:
  MODEL, MAX_STEPS, BATCH_SIZE, ROLLOUTS_PER_EXAMPLE, MAX_TOKENS
  LEARNING_RATE, LORA_ALPHA
  CHECKPOINT_ID, CHECKPOINT_INTERVAL, CHECKPOINT_KEEP_CLOUD
  EVAL_INTERVAL, EVAL_NUM_EXAMPLES, EVAL_ROLLOUTS_PER_EXAMPLE, EVAL_BASE_MODEL=1
  RUN_NAME, WANDB_PROJECT, WANDB_ENTITY, WANDB_NAME
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="$1"
shift

OWNER="${PRIME_ENV_OWNER:-}"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  OWNER="$1"
  shift
fi

if [[ -z "$OWNER" ]]; then
  echo "Owner slug is required. Pass [owner] or set PRIME_ENV_OWNER." >&2
  exit 1
fi

EXTRA_RUN_ARGS=("$@")
TS="$(date +%Y%m%d_%H%M%S)"
GENERATED_DIR="${GENERATED_DIR:-$REPO_DIR/outputs/generated_configs}"
CONFIG_PATH="${CONFIG_PATH:-$GENERATED_DIR/${TASK}_${TS}.toml}"

RENDER_ARGS=(
  --task "$TASK"
  --owner "$OWNER"
  --output "$CONFIG_PATH"
)

if [[ -n "${MODEL:-}" ]]; then
  RENDER_ARGS+=(--model "$MODEL")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  RENDER_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  RENDER_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${LEARNING_RATE:-}" ]]; then
  RENDER_ARGS+=(--learning-rate "$LEARNING_RATE")
fi
if [[ -n "${LORA_ALPHA:-}" ]]; then
  RENDER_ARGS+=(--lora-alpha "$LORA_ALPHA")
fi
if [[ -n "${ROLLOUTS_PER_EXAMPLE:-}" ]]; then
  RENDER_ARGS+=(--rollouts-per-example "$ROLLOUTS_PER_EXAMPLE")
fi
if [[ -n "${MAX_TOKENS:-}" ]]; then
  RENDER_ARGS+=(--max-tokens "$MAX_TOKENS")
fi
if [[ -n "${CHECKPOINT_ID:-}" ]]; then
  RENDER_ARGS+=(--checkpoint-id "$CHECKPOINT_ID")
fi
if [[ -n "${CHECKPOINT_INTERVAL:-}" ]]; then
  RENDER_ARGS+=(--checkpoint-interval "$CHECKPOINT_INTERVAL")
fi
if [[ -n "${CHECKPOINT_KEEP_CLOUD:-}" ]]; then
  RENDER_ARGS+=(--checkpoint-keep-cloud "$CHECKPOINT_KEEP_CLOUD")
fi
if [[ -n "${EVAL_INTERVAL:-}" ]]; then
  RENDER_ARGS+=(--eval-interval "$EVAL_INTERVAL")
fi
if [[ -n "${EVAL_NUM_EXAMPLES:-}" ]]; then
  RENDER_ARGS+=(--eval-num-examples "$EVAL_NUM_EXAMPLES")
fi
if [[ -n "${EVAL_ROLLOUTS_PER_EXAMPLE:-}" ]]; then
  RENDER_ARGS+=(--eval-rollouts-per-example "$EVAL_ROLLOUTS_PER_EXAMPLE")
fi
if [[ "${EVAL_BASE_MODEL:-0}" == "1" ]]; then
  RENDER_ARGS+=(--eval-base-model)
fi
if [[ -n "${RUN_NAME:-}" ]]; then
  RENDER_ARGS+=(--run-name "$RUN_NAME")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  RENDER_ARGS+=(--wandb-project "$WANDB_PROJECT")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  RENDER_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi
if [[ -n "${WANDB_NAME:-}" ]]; then
  RENDER_ARGS+=(--wandb-name "$WANDB_NAME")
fi

python3 "$REPO_DIR/scripts/render_hosted_config.py" "${RENDER_ARGS[@]}"

echo "Generated config: $CONFIG_PATH"
prime rl run "$CONFIG_PATH" "${EXTRA_RUN_ARGS[@]}"
