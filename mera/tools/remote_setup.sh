#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: remote_setup.sh [--repo-dir /path/to/repo] [options]

Options:
  --repo-dir DIR          Repo root (default: inferred from script location)
  --torch-index-url URL   PyTorch wheel index URL (default: https://download.pytorch.org/whl/cu124)
  --torch-version VER     PyTorch version (default: 2.6.0+cu124)
  --python VER            Python version for the main venv (default: 3.11)
  --prime-rl-python VER   Python version for prime-rl venv (default: 3.12)
  --mera-repo-url URL     MERA repo URL (default: https://github.com/MERA-Evaluation/MERA.git)
  --prime-rl-repo-url URL Prime-RL repo URL (default: https://github.com/PrimeIntellect-ai/prime-rl.git)
  --prime-rl-dir DIR      Prime-RL install directory (default: <repo>/_deps/prime-rl)
USAGE
}

REPO_DIR=""
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
TORCH_VERSION="2.6.0+cu124"
PYTHON_VERSION="3.11"
PRIME_RL_PYTHON_VERSION="3.12"
MERA_REPO_URL="https://github.com/MERA-Evaluation/MERA.git"
PRIME_RL_REPO_URL="https://github.com/PrimeIntellect-ai/prime-rl.git"
PRIME_RL_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
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
if [[ -z "$PRIME_RL_DIR" ]]; then
  PRIME_RL_DIR="$REPO_DIR/_deps/prime-rl"
fi

APT_GET="apt-get"
if command -v sudo >/dev/null 2>&1; then
  APT_GET="sudo apt-get"
fi

DEBIAN_FRONTEND=noninteractive $APT_GET update -y
DEBIAN_FRONTEND=noninteractive $APT_GET install -y \
  python3 python3-venv build-essential git curl

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

rm -rf /tmp/.tmp* /tmp/uv* /tmp/torchinductor_root
rm -rf "$HOME/.cache/uv" "$HOME/.cache/pip" "$HOME/.cache/huggingface" "$HOME/.cache/torch"

cd "$REPO_DIR"
if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
  echo "pyproject.toml not found in $REPO_DIR" >&2
  exit 1
fi

PYTHON_SHORT="$(echo "$PYTHON_VERSION" | cut -d. -f1,2)"
PYTHON_MAJOR="${PYTHON_SHORT%.*}"
PYTHON_MINOR="${PYTHON_SHORT#*.}"
PYTHON_NEXT_MINOR=$((PYTHON_MINOR + 1))
PYTHON_RANGE=">=${PYTHON_MAJOR}.${PYTHON_MINOR},<${PYTHON_MAJOR}.${PYTHON_NEXT_MINOR}"

python3 - <<PY
from pathlib import Path

path = Path("pyproject.toml")
lines = path.read_text().splitlines()
new_line = 'requires-python = "${PYTHON_RANGE}"'

for idx, line in enumerate(lines):
    if line.strip().startswith("requires-python"):
        lines[idx] = new_line
        break
else:
    insert_at = None
    for idx, line in enumerate(lines):
        if line.strip() == "[project]":
            insert_at = idx + 1
            break
    if insert_at is None:
        lines.insert(0, "[project]")
        insert_at = 1
    lines.insert(insert_at, new_line)

path.write_text("\n".join(lines) + "\n")
PY

uv python install "$PYTHON_VERSION"
rm -rf "$REPO_DIR/.venv"
uv venv .venv --python "$PYTHON_VERSION"
source .venv/bin/activate

if [[ -f "$REPO_DIR/uv.lock" ]]; then
  if ! python3 - <<'PY'; then
from __future__ import annotations
from pathlib import Path
import tomllib

path = Path("uv.lock")
text = path.read_text()
if not text.strip():
    raise SystemExit(1)

data = tomllib.loads(text)
if "version" not in data:
    raise SystemExit(1)
PY
    rm -f "$REPO_DIR/uv.lock"
  fi
fi

uv add --index "$TORCH_INDEX_URL" --index-strategy unsafe-best-match "torch==${TORCH_VERSION}"

mapfile -t missing < <(
  python3 - <<'PY'
from __future__ import annotations
from pathlib import Path
import tomllib

path = Path("pyproject.toml")
data = tomllib.loads(path.read_text())
deps = data.get("project", {}).get("dependencies", [])

def dep_name(spec: str) -> str:
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in spec:
            return spec.split(sep, 1)[0].strip()
    return spec.strip()

names = {dep_name(dep) for dep in deps}
required = {
    "transformers",
    "datasets",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "wandb",
    "scikit-learn",
    "tqdm",
    "multiprocess",
    "verifiers",
    "omegaconf",
    "boto3",
    "scipy",
    "sentencepiece",
    "tiktoken",
    "protobuf",
}

missing = sorted(required - names)
for pkg in missing:
    print(pkg)
PY
)

if [[ ${#missing[@]} -gt 0 ]]; then
  uv add "${missing[@]}"
fi

uv add vllm

uv sync

rm -rf /tmp/.tmp* /tmp/uv* /tmp/torchinductor_root
rm -rf "$HOME/.cache/uv" "$HOME/.cache/pip" "$HOME/.cache/huggingface" "$HOME/.cache/torch"

mkdir -p "$REPO_DIR/_deps"
if [[ ! -d "$REPO_DIR/MERA_repo/.git" ]]; then
  git clone --recurse-submodules "$MERA_REPO_URL" "$REPO_DIR/MERA_repo"
else
  git -C "$REPO_DIR/MERA_repo" pull --recurse-submodules
fi

if [[ ! -d "$PRIME_RL_DIR/.git" ]]; then
  git clone "$PRIME_RL_REPO_URL" "$PRIME_RL_DIR"
else
  git -C "$PRIME_RL_DIR" pull
fi

(
  cd "$PRIME_RL_DIR"
  uv python install "$PRIME_RL_PYTHON_VERSION"
  rm -rf "$PRIME_RL_DIR/.venv"
  uv venv .venv --python "$PRIME_RL_PYTHON_VERSION"
  uv sync
  for env_dir in "$REPO_DIR/mera/environments"/*; do
    if [[ -f "$env_dir/pyproject.toml" ]]; then
      uv add --editable "$env_dir"
    fi
  done
)

cat <<SETUP_EOF
Setup complete.

Next steps:
  source "$REPO_DIR/.venv/bin/activate"

Smoke tests:
  python mera/scripts/sft.py --limit 200 --epochs 1 --batch-size 1 --grad-accum 8
  python mera/scripts/grpo.py bps --dry-run
  python mera/scripts/eval.py --limit 50 --tensor-parallel 1
SETUP_EOF
