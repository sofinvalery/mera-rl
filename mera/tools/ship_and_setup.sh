#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ship_and_setup.sh --host user@host [options]

Required:
  --host USER@HOST       SSH destination.

Options:
  --port PORT            SSH port (default: 22)
  --key PATH             SSH private key
  --proxyjump HOST       SSH ProxyJump
  --target DIR           Target install directory on remote (default: /workspace/rl/mera-rl)
  --torch-index-url URL  PyTorch wheel index URL (default: https://download.pytorch.org/whl/cu124)
  --data-dir DIR         MERA_DATA_DIR value to write into env.sh
  --wandb-project NAME   WANDB_PROJECT value to write into env.sh
  --wandb-entity NAME    WANDB_ENTITY value to write into env.sh
  --wandb-run-group NAME WANDB_RUN_GROUP value to write into env.sh
  --prime-rl-repo-url URL Prime-RL repo URL (default: https://github.com/PrimeIntellect-ai/prime-rl.git)
  --prime-rl-dir DIR      Prime-RL install directory on remote
  --keep-remote-bundle   Keep bundle on remote after extraction
EOF
}

HOST=""
PORT="22"
KEY=""
PROXYJUMP=""
TARGET_DIR="/workspace/rl/mera-rl"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
DATA_DIR=""
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_GROUP=""
PRIME_RL_REPO_URL=""
PRIME_RL_DIR=""
KEEP_REMOTE_BUNDLE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --key)
      KEY="$2"
      shift 2
      ;;
    --proxyjump)
      PROXYJUMP="$2"
      shift 2
      ;;
    --target)
      TARGET_DIR="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb-run-group)
      WANDB_RUN_GROUP="$2"
      shift 2
      ;;
    --prime-rl-repo-url)
      PRIME_RL_REPO_URL="$2"
      shift 2
      ;;
    --prime-rl-dir)
      PRIME_RL_DIR="$2"
      shift 2
      ;;
    --keep-remote-bundle)
      KEEP_REMOTE_BUNDLE="true"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "--host is required" >&2
  usage
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_PATH="$(mktemp -t mera_bundle.XXXXXX.tgz)"
REMOTE_BUNDLE="/tmp/mera_bundle.tgz"
REMOTE_SETUP="/tmp/mera_remote_setup.sh"

BUNDLE_ITEMS=("mera")
if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
  BUNDLE_ITEMS+=("pyproject.toml")
fi
if [[ -f "$REPO_ROOT/uv.lock" ]]; then
  BUNDLE_ITEMS+=("uv.lock")
fi

tar -czf "$BUNDLE_PATH" -C "$REPO_ROOT" "${BUNDLE_ITEMS[@]}"

SCP_ARGS=("-P" "$PORT")
SSH_ARGS=("-p" "$PORT")
if [[ -n "$KEY" ]]; then
  SCP_ARGS+=("-i" "$KEY")
  SSH_ARGS+=("-i" "$KEY")
fi
if [[ -n "$PROXYJUMP" ]]; then
  SCP_ARGS+=("-o" "ProxyJump=$PROXYJUMP")
  SSH_ARGS+=("-o" "ProxyJump=$PROXYJUMP")
fi

scp "${SCP_ARGS[@]}" "$BUNDLE_PATH" "$HOST:$REMOTE_BUNDLE"
scp "${SCP_ARGS[@]}" "$REPO_ROOT/mera/tools/remote_setup.sh" "$HOST:$REMOTE_SETUP"

REMOTE_CMD=(
  "bash" "$REMOTE_SETUP"
  "--bundle" "$REMOTE_BUNDLE"
  "--target" "$TARGET_DIR"
  "--torch-index-url" "$TORCH_INDEX_URL"
)

if [[ -n "$DATA_DIR" ]]; then
  REMOTE_CMD+=("--data-dir" "$DATA_DIR")
fi
if [[ -n "$WANDB_PROJECT" ]]; then
  REMOTE_CMD+=("--wandb-project" "$WANDB_PROJECT")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  REMOTE_CMD+=("--wandb-entity" "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_RUN_GROUP" ]]; then
  REMOTE_CMD+=("--wandb-run-group" "$WANDB_RUN_GROUP")
fi
if [[ -n "$PRIME_RL_REPO_URL" ]]; then
  REMOTE_CMD+=("--prime-rl-repo-url" "$PRIME_RL_REPO_URL")
fi
if [[ -n "$PRIME_RL_DIR" ]]; then
  REMOTE_CMD+=("--prime-rl-dir" "$PRIME_RL_DIR")
fi
if [[ "$KEEP_REMOTE_BUNDLE" == "true" ]]; then
  REMOTE_CMD+=("--keep-bundle")
fi

ssh "${SSH_ARGS[@]}" "$HOST" "${REMOTE_CMD[*]}"

rm -f "$BUNDLE_PATH"

echo "Done. Project installed at $HOST:$TARGET_DIR"
