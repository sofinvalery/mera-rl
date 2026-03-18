#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/check_env_actions.sh [owner] [task1 task2 ...]

Examples:
  scripts/check_env_actions.sh my-team
  scripts/check_env_actions.sh my-team mathlogicqa rcb
  PRIME_ENV_OWNER=my-team scripts/check_env_actions.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="$REPO_DIR/environments"

OWNER="${PRIME_ENV_OWNER:-}"
if [[ $# -gt 0 ]]; then
  candidate="$1"
  if [[ ! -d "$ENV_ROOT/$candidate" ]]; then
    OWNER="$candidate"
    shift
  fi
fi

if [[ -z "$OWNER" ]]; then
  echo "Owner slug is required. Pass [owner] or set PRIME_ENV_OWNER." >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
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
fi

for task in "${TASKS[@]}"; do
  echo
  echo "=== $OWNER/$task ==="
  prime env action list "$OWNER/$task" -l 5
done
