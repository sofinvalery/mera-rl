#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${EXPERIMENT:-mera_rlvr_main_sequential}"

python3 "$REPO_DIR/scripts/run_rlvr_local_sequential.py" train --experiment "$EXPERIMENT" "$@"
