#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: remote_setup.sh --bundle /tmp/mera_bundle.tgz [options]

Required:
  --bundle PATH           Path to the tarball transferred via scp.

Options:
  --target DIR            Target install directory (default: ~/mera)
  --torch-index-url URL   PyTorch wheel index URL (default: https://download.pytorch.org/whl/cu124)
  --mera-repo-url URL     MERA repo URL (default: https://github.com/MERA-Evaluation/MERA.git)
  --data-dir DIR          MERA_DATA_DIR value to write into env.sh
  --wandb-project NAME    WANDB_PROJECT value to write into env.sh
  --wandb-entity NAME     WANDB_ENTITY value to write into env.sh
  --wandb-run-group NAME  WANDB_RUN_GROUP value to write into env.sh
  --keep-bundle           Do not delete bundle after extraction
EOF
}

BUNDLE=""
TARGET_DIR="$HOME/mera"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
MERA_REPO_URL="https://github.com/MERA-Evaluation/MERA.git"
DATA_DIR=""
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_GROUP=""
KEEP_BUNDLE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      BUNDLE="$2"
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
    --mera-repo-url)
      MERA_REPO_URL="$2"
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
    --keep-bundle)
      KEEP_BUNDLE="true"
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

if [[ -z "$BUNDLE" ]]; then
  echo "--bundle is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$BUNDLE" ]]; then
  echo "Bundle not found: $BUNDLE" >&2
  exit 1
fi

APT_GET="apt-get"
if command -v sudo >/dev/null 2>&1; then
  APT_GET="sudo apt-get"
fi

DEBIAN_FRONTEND=noninteractive $APT_GET update -y
DEBIAN_FRONTEND=noninteractive $APT_GET install -y \
  python3 python3-venv build-essential git curl rsync

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p "$TARGET_DIR"
tar -xzf "$BUNDLE" -C "$TARGET_DIR"

if [[ "$KEEP_BUNDLE" != "true" ]]; then
  rm -f "$BUNDLE"
fi

cd "$TARGET_DIR"
uv venv .venv --python python3
source .venv/bin/activate

uv pip install --index-url "$TORCH_INDEX_URL" torch
uv pip install \
  transformers \
  accelerate \
  bitsandbytes \
  peft \
  trl \
  datasets \
  verifiers \
  vllm \
  scikit-learn \
  scipy \
  omegaconf \
  multiprocess \
  wandb \
  sentencepiece \
  tiktoken \
  protobuf

mkdir -p "$TARGET_DIR/_deps"
if [[ ! -d "$TARGET_DIR/_deps/MERA/.git" ]]; then
  git clone --recurse-submodules "$MERA_REPO_URL" "$TARGET_DIR/_deps/MERA"
else
  git -C "$TARGET_DIR/_deps/MERA" pull --recurse-submodules
fi

ln -sfn "$TARGET_DIR/_deps/MERA" "$TARGET_DIR/MERA_repo"

ENV_FILE="$TARGET_DIR/env.sh"
{
  echo "export WANDB_LOG_MODEL=false"
  echo "export WANDB_WATCH=false"
  echo "export WANDB_DISABLE_CODE=true"
  echo "export WANDB_SILENT=true"
  echo "export NCCL_CUMEM_ENABLE=0"
  echo "export NCCL_CUMEM_HOST_ENABLE=0"
  if [[ -n "$DATA_DIR" ]]; then
    echo "export MERA_DATA_DIR=\"$DATA_DIR\""
  fi
  if [[ -n "$WANDB_PROJECT" ]]; then
    echo "export WANDB_PROJECT=\"$WANDB_PROJECT\""
  fi
  if [[ -n "$WANDB_ENTITY" ]]; then
    echo "export WANDB_ENTITY=\"$WANDB_ENTITY\""
  fi
  if [[ -n "$WANDB_RUN_GROUP" ]]; then
    echo "export WANDB_RUN_GROUP=\"$WANDB_RUN_GROUP\""
  fi
} > "$ENV_FILE"

cat <<EOF
Setup complete.

Next steps:
  source "$TARGET_DIR/.venv/bin/activate"
  source "$ENV_FILE"

Smoke tests:
  python mera/scripts/sft.py --limit 200 --epochs 1 --batch-size 1 --grad-accum 8
  python mera/scripts/grpo.py --limit 100 --num-generations 1 --max-new-tokens 32 --fast
EOF
