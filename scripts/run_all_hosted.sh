#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_all_hosted.sh [owner] [prime-rl-run-args...]

Examples:
  scripts/run_all_hosted.sh my-team
  scripts/run_all_hosted.sh my-team --skip-action-check
  PRIME_ENV_OWNER=my-team scripts/run_all_hosted.sh

Environment:
  CONTINUE_ON_ERROR=1   Continue launching next task if one run command fails.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
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

TASKS=(
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

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TASK_SCRIPT="$REPO_DIR/scripts/run_task_hosted.sh"

for task in "${TASKS[@]}"; do
  echo
  echo "=== Launching task: $task ==="

  if CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}" "$RUN_TASK_SCRIPT" "$task" "$OWNER" "${EXTRA_RUN_ARGS[@]}"; then
    true
  elif [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
    echo "Task failed (continuing): $task" >&2
    continue
  else
    exit 1
  fi
done
