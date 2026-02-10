#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: remote_setup.sh [options]

Sets up an Ubuntu machine (with NVIDIA GPUs) to run MERA SFT, GRPO, and evals.
Creates:
  - <repo>/.venv (Python 3.11) for MERA scripts
  - <repo>/_deps/prime-rl/.venv (Python 3.12) for Prime-RL (GRPO)

Options:
  --repo-dir DIR           Repo root (default: inferred from script location)
  --python VER             Python version for MERA venv (default: 3.11)
  --prime-rl-python VER    Python version for Prime-RL venv (default: 3.12)
  --torch-index-url URL    PyTorch index URL (default: https://download.pytorch.org/whl/cu124)
  --recreate-venvs         Recreate both venvs from scratch

  --data-dir DIR           MERA_DATA_DIR value to write into env.sh
  --hf-home DIR            HF_HOME value to write into env.sh (default: <repo>/.hf)
  --wandb-project NAME     WANDB_PROJECT value to write into env.sh
  --wandb-entity NAME      WANDB_ENTITY value to write into env.sh
  --wandb-run-group NAME   WANDB_RUN_GROUP value to write into env.sh
USAGE
}

REPO_DIR=""
PYTHON_VERSION="3.11"
PRIME_RL_PYTHON_VERSION="3.12"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
RECREATE_VENVS="false"

DATA_DIR=""
HF_HOME_DIR=""
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_GROUP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --prime-rl-python)
      PRIME_RL_PYTHON_VERSION="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --recreate-venvs)
      RECREATE_VENVS="true"
      shift 1
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --hf-home)
      HF_HOME_DIR="$2"
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

if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

APT_GET="apt-get"
if command -v sudo >/dev/null 2>&1; then
  APT_GET="sudo apt-get"
fi

DEBIAN_FRONTEND=noninteractive $APT_GET update -y
DEBIAN_FRONTEND=noninteractive $APT_GET install -y \
  python3 python3-venv build-essential git curl ca-certificates

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
  echo "pyproject.toml not found in $REPO_DIR" >&2
  exit 1
fi

if [[ -e "$REPO_DIR/.git" && -f "$REPO_DIR/.gitmodules" ]]; then
  git -C "$REPO_DIR" submodule sync --recursive || true
  git -C "$REPO_DIR" submodule update --init --recursive
fi

MERA_DIR="$REPO_DIR/_deps/MERA"
PRIME_RL_DIR="$REPO_DIR/_deps/prime-rl"

if [[ -z "$HF_HOME_DIR" ]]; then
  HF_HOME_DIR="$REPO_DIR/.hf"
fi
mkdir -p "$HF_HOME_DIR" "$HF_HOME_DIR/datasets" "$HF_HOME_DIR/hub"

VENV_FLAGS=(--allow-existing --seed)
if [[ "$RECREATE_VENVS" == "true" ]]; then
  VENV_FLAGS=(--clear --seed)
fi

uv python install "$PYTHON_VERSION"
uv venv "$REPO_DIR/.venv" --python "$PYTHON_VERSION" "${VENV_FLAGS[@]}"

if [[ -f "$REPO_DIR/uv.lock" ]]; then
  uv sync --frozen --index "$TORCH_INDEX_URL" --index-strategy unsafe-best-match
else
  uv sync --index "$TORCH_INDEX_URL" --index-strategy unsafe-best-match
fi

if [[ -d "$PRIME_RL_DIR" ]]; then
  (
    cd "$PRIME_RL_DIR"
    uv python install "$PRIME_RL_PYTHON_VERSION"
    uv venv .venv --python "$PRIME_RL_PYTHON_VERSION" "${VENV_FLAGS[@]}"
    if [[ -f "uv.lock" ]]; then
      uv sync --frozen
    else
      uv sync
    fi
  )
fi

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf 'export HF_HOME=%q\n' "$HF_HOME_DIR"
  printf 'export HF_DATASETS_CACHE=%q\n' "$HF_HOME_DIR/datasets"
  printf 'export HF_HUB_CACHE=%q\n' "$HF_HOME_DIR/hub"
  if [[ -d "$MERA_DIR" ]]; then
    printf 'export MERA_REPO_DIR=%q\n' "$MERA_DIR"
  fi
  if [[ -n "$DATA_DIR" ]]; then
    printf 'export MERA_DATA_DIR=%q\n' "$DATA_DIR"
  fi
  if [[ -n "$WANDB_PROJECT" ]]; then
    printf 'export WANDB_PROJECT=%q\n' "$WANDB_PROJECT"
  fi
  if [[ -n "$WANDB_ENTITY" ]]; then
    printf 'export WANDB_ENTITY=%q\n' "$WANDB_ENTITY"
  fi
  if [[ -n "$WANDB_RUN_GROUP" ]]; then
    printf 'export WANDB_RUN_GROUP=%q\n' "$WANDB_RUN_GROUP"
  fi
} > "$REPO_DIR/env.sh"
chmod +x "$REPO_DIR/env.sh"

cat <<EOF2
Setup complete.

MERA venv:
  source "$REPO_DIR/.venv/bin/activate"

Prime-RL venv:
  $PRIME_RL_DIR/.venv

Environment:
  source "$REPO_DIR/env.sh"

Smoke tests:
  python mera/scripts/sft.py --limit 50 --epochs 1 --batch-size 1 --grad-accum 4 --output-dir outputs/smoke_sft
  python mera/scripts/grpo.py bps --dry-run
  python mera/scripts/eval_base.py --limit 1 --tensor-parallel 1 --skip-scoring --output-dir outputs/smoke_eval_base
EOF2
