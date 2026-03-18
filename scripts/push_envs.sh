#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/push_envs.sh [owner] [task1 task2 ...]

Examples:
  scripts/push_envs.sh my-team
  scripts/push_envs.sh my-team mathlogicqa rcb rwsd
  PRIME_ENV_OWNER=my-team scripts/push_envs.sh

Environment:
  PUSH_VISIBILITY=PUBLIC|PRIVATE
  PUSH_TEAM=<team-slug>
  PUSH_AUTO_BUMP=1
  PUSH_RC=1
  PUSH_POST=1
EOF
}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="$REPO_DIR/environments"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

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
  env_dir="$ENV_ROOT/$task"
  if [[ ! -d "$env_dir" ]]; then
    echo "Skipping missing environment: $task" >&2
    continue
  fi

  cmd=(prime env push --path "$env_dir" --owner "$OWNER")
  if [[ -n "${PUSH_TEAM:-}" ]]; then
    cmd+=(--team "$PUSH_TEAM")
  fi
  if [[ -n "${PUSH_VISIBILITY:-}" ]]; then
    cmd+=(--visibility "$PUSH_VISIBILITY")
  fi
  if [[ "${PUSH_AUTO_BUMP:-0}" == "1" ]]; then
    cmd+=(--auto-bump)
  fi
  if [[ "${PUSH_RC:-0}" == "1" ]]; then
    cmd+=(--rc)
  fi
  if [[ "${PUSH_POST:-0}" == "1" ]]; then
    cmd+=(--post)
  fi

  echo
  echo "=== Pushing: $task ==="
  "${cmd[@]}"
done
