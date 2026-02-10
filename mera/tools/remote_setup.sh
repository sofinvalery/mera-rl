#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: remote_setup.sh [--repo-dir DIR] [--bundle PATH --target DIR] [options]

Installs / updates dependencies and prepares a Prime-RL venv (Python 3.12).

Options:
  --repo-dir DIR            Repo root (default: inferred when running inside the repo)
  --bundle PATH             Tarball produced by ship_and_setup.sh
  --target DIR              Extraction directory for --bundle
  --keep-bundle             Keep the tarball after extraction

  --prime-rl-python VER     Python version for Prime-RL venv (default: 3.12)
  --prime-rl-repo-url URL   Prime-RL repo URL (default: https://github.com/PrimeIntellect-ai/prime-rl.git)
  --prime-rl-dir DIR        Prime-RL install directory (default: <repo>/_deps/prime-rl)
  --recreate-venv           Recreate Prime-RL .venv from scratch (slow)

  --mera-repo-url URL       MERA repo URL (default: https://github.com/MERA-Evaluation/MERA.git)
  --mera-dir DIR            MERA install directory (default: <repo>/_deps/MERA)

  --data-dir DIR            MERA_DATA_DIR value to write into env.sh
  --hf-home DIR             HF_HOME value to write into env.sh (default: /workspace/rl/.hf)
  --wandb-project NAME      WANDB_PROJECT value to write into env.sh
  --wandb-entity NAME       WANDB_ENTITY value to write into env.sh
  --wandb-run-group NAME    WANDB_RUN_GROUP value to write into env.sh

Compatibility (ignored):
  --torch-index-url URL
  --torch-version VER
USAGE
}

REPO_DIR=""
BUNDLE_PATH=""
TARGET_DIR=""
KEEP_BUNDLE="false"

PRIME_RL_PYTHON_VERSION="3.12"
MERA_REPO_URL="https://github.com/MERA-Evaluation/MERA.git"
PRIME_RL_REPO_URL="https://github.com/PrimeIntellect-ai/prime-rl.git"
PRIME_RL_DIR=""
MERA_DIR=""
RECREATE_VENV="false"

DATA_DIR=""
HF_HOME_DIR="/workspace/rl/.hf"
WANDB_PROJECT=""
WANDB_ENTITY=""
WANDB_RUN_GROUP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --bundle)
      BUNDLE_PATH="$2"
      shift 2
      ;;
    --target)
      TARGET_DIR="$2"
      shift 2
      ;;
    --keep-bundle|--keep-remote-bundle)
      KEEP_BUNDLE="true"
      shift 1
      ;;
    --torch-index-url)
      shift 2
      ;;
    --torch-version)
      shift 2
      ;;
    --prime-rl-python)
      PRIME_RL_PYTHON_VERSION="$2"
      shift 2
      ;;
    --mera-repo-url)
      MERA_REPO_URL="$2"
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
    --recreate-venv)
      RECREATE_VENV="true"
      shift 1
      ;;
    --mera-dir)
      MERA_DIR="$2"
      shift 2
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

if [[ -n "$BUNDLE_PATH" ]]; then
  if [[ -z "$TARGET_DIR" ]]; then
    echo "--target is required when using --bundle" >&2
    usage
    exit 1
  fi
  mkdir -p "$TARGET_DIR"
  tar -xzf "$BUNDLE_PATH" -C "$TARGET_DIR"
  if [[ "$KEEP_BUNDLE" != "true" ]]; then
    rm -f "$BUNDLE_PATH"
  fi
  REPO_DIR="$TARGET_DIR"
fi

if [[ -z "$REPO_DIR" ]]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

if [[ -z "$PRIME_RL_DIR" ]]; then
  PRIME_RL_DIR="$REPO_DIR/_deps/prime-rl"
fi
if [[ -z "$MERA_DIR" ]]; then
  MERA_DIR="$REPO_DIR/_deps/MERA"
fi

APT_GET="apt-get"
if command -v sudo >/dev/null 2>&1; then
  APT_GET="sudo apt-get"
fi

DEBIAN_FRONTEND=noninteractive $APT_GET update -y
DEBIAN_FRONTEND=noninteractive $APT_GET install -y \
  python3 python3-venv build-essential git curl ca-certificates \
  tmux htop nvtop git-lfs openssh-client

git lfs install || true

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$REPO_DIR"
mkdir -p "$REPO_DIR/_deps"

if [[ -e "$REPO_DIR/.git" && -f "$REPO_DIR/.gitmodules" ]]; then
  git -C "$REPO_DIR" submodule update --init --recursive || true
fi

if [[ -f "$MERA_DIR/.git" ]]; then
  echo "Using MERA submodule at $MERA_DIR"
elif [[ -d "$MERA_DIR/.git" ]]; then
  git -C "$MERA_DIR" pull --recurse-submodules
else
  rm -rf "$MERA_DIR"
  git clone --recurse-submodules "$MERA_REPO_URL" "$MERA_DIR"
fi

if [[ -f "$PRIME_RL_DIR/.git" ]]; then
  echo "Using prime-rl submodule at $PRIME_RL_DIR"
elif [[ -d "$PRIME_RL_DIR/.git" ]]; then
  git -C "$PRIME_RL_DIR" pull
else
  rm -rf "$PRIME_RL_DIR"
  git clone "$PRIME_RL_REPO_URL" "$PRIME_RL_DIR"
fi

(
  cd "$PRIME_RL_DIR"
  uv python install "$PRIME_RL_PYTHON_VERSION"
  if [[ "$RECREATE_VENV" == "true" || ! -d ".venv" ]]; then
    rm -rf ".venv"
    uv venv .venv --python "$PRIME_RL_PYTHON_VERSION"
  fi
  uv sync
)

VENV_PY="$PRIME_RL_DIR/.venv/bin/python"
uv pip install --python "$VENV_PY" multiprocess omegaconf scikit-learn boto3

INSTALL_ENVS=(
  chegeka
  lcs
  mamuramu
  mathlogicqa
  multiq
  parus
  rcb
  rucodeeval
  rumodar
  rumultiar
  ruopenbookqa
  rutie
  ruworldtree
  rwsd
  use
)

for env_name in "${INSTALL_ENVS[@]}"; do
  env_dir="$REPO_DIR/mera/environments/$env_name"
  if [[ ! -f "$env_dir/pyproject.toml" ]]; then
    echo "Missing environment package: $env_dir" >&2
    exit 1
  fi
  uv pip install --python "$VENV_PY" -e "$env_dir"
done

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf 'export HF_HOME=%q\n' "$HF_HOME_DIR"
  printf 'export HF_DATASETS_CACHE=%q\n' "$HF_HOME_DIR/datasets"
  printf 'export HF_HUB_CACHE=%q\n' "$HF_HOME_DIR/hub"
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

mkdir -p "$HF_HOME_DIR" "$HF_HOME_DIR/datasets" "$HF_HOME_DIR/hub"

cat <<SETUP_EOF
Setup complete.

Prime-RL venv:
  $PRIME_RL_DIR/.venv

Next steps:
  if [ -f "$REPO_DIR/env.sh" ]; then source "$REPO_DIR/env.sh"; fi

Smoke tests:
  python mera/scripts/grpo.py mathlogicqa --dry-run
  $PRIME_RL_DIR/.venv/bin/python mera/scripts/eval.py --limit 50 --tensor-parallel 1 --skip-scoring
SETUP_EOF
