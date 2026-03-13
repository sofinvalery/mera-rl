#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_all_hosted.sh [fair|all] [owner] [prime-rl-run-args...]

Examples:
  scripts/run_all_hosted.sh fair my-team
  scripts/run_all_hosted.sh all my-team --skip-action-check
  PRIME_ENV_OWNER=my-team scripts/run_all_hosted.sh fair

Environment:
  CONTINUE_ON_ERROR=1   Continue launching next task if one run command fails.
EOF
}

MODE="fair"
if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    fair|all)
      MODE="$1"
      shift
      ;;
    *)
      MODE="fair"
      ;;
  esac
fi

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

FAIR_TASKS=(
  chegeka
  lcs
  mamuramu
  mathlogicqa
  multiq
  parus
  rcb
  rumodar
  rumultiar
  ruopenbookqa
  rutie
  ruworldtree
  rwsd
  use
)

ALL_TASKS=(
  bps
  chegeka
  lcs
  mamuramu
  mathlogicqa
  multiq
  parus
  rcb
  rucodeeval
  rudetox
  rummlu
  rumodar
  rumultiar
  ruopenbookqa
  rutie
  ruworldtree
  rwsd
  simplear
  use
)

case "$MODE" in
  fair)
    TASKS=("${FAIR_TASKS[@]}")
    ;;
  all)
    TASKS=("${ALL_TASKS[@]}")
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TASK_SCRIPT="$REPO_DIR/scripts/run_task_hosted.sh"

for task in "${TASKS[@]}"; do
  echo
  echo "=== Launching task: $task ==="

  if [[ "$task" == "rucodeeval" ]]; then
    if CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}" ALLOW_NONFAIR=1 "$RUN_TASK_SCRIPT" "$task" "$OWNER" "${EXTRA_RUN_ARGS[@]}"; then
      true
    elif [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
      echo "Task failed (continuing): $task" >&2
      continue
    else
      exit 1
    fi
    continue
  fi

  if CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}" "$RUN_TASK_SCRIPT" "$task" "$OWNER" "${EXTRA_RUN_ARGS[@]}"; then
    true
  elif [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
    echo "Task failed (continuing): $task" >&2
    continue
  else
    exit 1
  fi
done
