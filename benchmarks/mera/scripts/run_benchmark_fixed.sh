#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_benchmark_fixed.sh [--mode regular|gen|all|all_local] [--dry-run]

Required env:
  MODEL_ID                 Hugging Face model id, e.g. Qwen/Qwen3-4B-Instruct-2507

Optional env:
  MODEL_CACHE_DIR          Cache root for predownloaded weights (default: HF_HOME)
  MODEL_DIR                Backward-compatible alias for MODEL_CACHE_DIR
  MERA_DATASETS_CACHE      Persistent datasets cache (default: HF_HOME/datasets)
  DOWNLOAD_MODELS          1 to warm the HF cache before benchmark start (default: 1)
  DOWNLOAD_DATASETS        1 to warm the MERA dataset cache before benchmark start (default: 1)
  REBUILD_VENV             1 to force a full venv rebuild (default: 0)
  MERA_FOLDER              Output directory root (default: ./mera_results/<sanitized_model>_<mode>_<timestamp>)
  CUDA_VISIBLE_DEVICES     GPU selection (default: 0,1)
  HF_HOME                  HF cache root (default: ./hf_home)
  MERA_COMMON_SETUP        Override lm_eval common flags
  RUN_LOCAL_SCORING        1 to run the public local scorer on produced zip files (default: 1)
  PYTHON_BIN               Python executable for uv venv creation (default: python3.12)

Examples:
  MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 scripts/run_benchmark_fixed.sh --mode all
  MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 MODEL_CACHE_DIR=/workspace/.hf_home scripts/run_benchmark_fixed.sh --mode regular
EOF
}

MODE="all"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$MODE" in
  regular)
    BENCHMARK_SCRIPT="scripts/run_benchmark.sh"
    ;;
  gen)
    BENCHMARK_SCRIPT="scripts/run_benchmark_gen.sh"
    ;;
  all)
    BENCHMARK_SCRIPT="scripts/run_benchmark_all.sh"
    ;;
  all_local)
    BENCHMARK_SCRIPT="scripts/run_benchmark_all_localscore.sh"
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 1
    ;;
esac

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
MODEL_ID="${MODEL_ID:-}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-1}"
REBUILD_VENV="${REBUILD_VENV:-0}"
RUN_LOCAL_SCORING="${RUN_LOCAL_SCORING:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
HF_HOME="${HF_HOME:-$REPO_DIR/hf_home}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${MODEL_DIR:-$HF_HOME}}"
MERA_DATASETS_CACHE="${MERA_DATASETS_CACHE:-$HF_HOME/datasets}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MERA_FOLDER="${MERA_FOLDER:-$REPO_DIR/mera_results/${MODEL_ID//\//__}_${MODE}_${STAMP}}"
VENV_DIR="${REPO_DIR}/.venv"
STATE_FILE="${VENV_DIR}/.mera_benchmark_state"
BOOTSTRAP_VERSION="v2"

if [[ -z "$MODEL_ID" ]]; then
  echo "MODEL_ID is required." >&2
  exit 1
fi

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_in_dir() {
  local dir="$1"
  shift
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ (cd %q &&' "$dir"
    printf ' %q' "$@"
    printf ')\n'
  else
    (
      cd "$dir"
      "$@"
    )
  fi
}

write_state() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ %s\n' "printf '%s\n' \"$1\" > \"$STATE_FILE\""
  else
    printf '%s\n' "$1" >"$STATE_FILE"
  fi
}

cd "$REPO_DIR"
mkdir -p "$HF_HOME"
mkdir -p "$MODEL_CACHE_DIR"
mkdir -p "$MERA_DATASETS_CACHE"
mkdir -p "$(dirname "$MERA_FOLDER")"

HARNESS_REV="$(git -C "$REPO_DIR/lm-evaluation-harness" rev-parse HEAD 2>/dev/null || echo unknown)"
STATE_VALUE="$(printf '%s\n' \
  "$BOOTSTRAP_VERSION" \
  "$PYTHON_BIN" \
  "$HARNESS_REV" \
  "torch+torchvision@cu128" \
  "lm-eval[vllm]" \
  "transformers<5" \
  "ray" \
  "accelerate" \
  "datasets" \
  "sentencepiece" \
  "protobuf" \
  "evaluate" \
  "sacrebleu" \
  "huggingface_hub" \
  "omegaconf" \
  "boto3" \
  "scikit-learn")"

CURRENT_STATE=""
if [[ -f "$STATE_FILE" ]]; then
  CURRENT_STATE="$(cat "$STATE_FILE")"
fi

if [[ ! -x "$VENV_DIR/bin/python" || "$CURRENT_STATE" != "$STATE_VALUE" || "$REBUILD_VENV" == "1" ]]; then
  run env UV_VENV_CLEAR=1 uv venv "$VENV_DIR" --python "$PYTHON_BIN"
  run uv pip install --python "$VENV_DIR/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  run_in_dir "$REPO_DIR/lm-evaluation-harness" uv pip install --python "$VENV_DIR/bin/python" -e '.[vllm]'
  run uv pip install --python "$VENV_DIR/bin/python" 'transformers<5' ray accelerate datasets sentencepiece protobuf evaluate sacrebleu huggingface_hub omegaconf boto3 scikit-learn
  run "$VENV_DIR/bin/python" -c "import transformers; from transformers import AutoModelForVision2Seq; print('transformers', transformers.__version__); print('vision2seq_ok', AutoModelForVision2Seq.__name__)"
  write_state "$STATE_VALUE"
else
  echo "Reusing existing uv environment at $VENV_DIR"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf '+ %s\n' ". \"$VENV_DIR/bin/activate\""
else
  . "$VENV_DIR/bin/activate"
fi

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  # Warm the HF cache before using vLLM data parallel to avoid worker download races.
  run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_ID', cache_dir='$MODEL_CACHE_DIR', resume_download=True); print('$MODEL_ID')"
fi

if [[ "$DOWNLOAD_DATASETS" == "1" ]]; then
  dataset_warmup_py="${TMPDIR:-/tmp}/mera_dataset_warmup.py"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ %s\n' "cat > \"$dataset_warmup_py\" <<'PY' ... PY"
  else
    cat >"$dataset_warmup_py" <<'PY'
from datasets import load_dataset

TASKS = [
    "simplear",
    "bps",
    "lcs",
    "chegeka",
    "mathlogicqa",
    "multiq",
    "parus",
    "rcb",
    "rudetox",
    "ruethics",
    "ruhatespeech",
    "ruhhh",
    "ruhumaneval",
    "rucodeeval",
    "rummlu",
    "rumodar",
    "rumultiar",
    "ruopenbookqa",
    "rutie",
    "ruworldtree",
    "rwsd",
    "mamuramu",
    "use",
]
SPLITS = ["train", "test", "validation", "public_test"]

for task in TASKS:
    for split in SPLITS:
        try:
            load_dataset("MERA-evaluation/MERA", name=task, split=split, cache_dir="__DATASETS_CACHE_DIR__")
            print(f"cached {task}:{split}")
        except Exception:
            pass
PY
    python3 - <<PY
from pathlib import Path
path = Path("$dataset_warmup_py")
text = path.read_text(encoding="utf-8")
text = text.replace("__DATASETS_CACHE_DIR__", "$MERA_DATASETS_CACHE")
path.write_text(text, encoding="utf-8")
PY
  fi
  run env HF_HOME="$HF_HOME" HF_HUB_CACHE="$MODEL_CACHE_DIR" HF_DATASETS_CACHE="$MERA_DATASETS_CACHE" python "$dataset_warmup_py"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ %s\n' "rm -f \"$dataset_warmup_py\""
  else
    rm -f "$dataset_warmup_py"
  fi
fi

MODEL_SOURCE="$MODEL_ID"

export HF_HOME
export HF_HUB_CACHE="$MODEL_CACHE_DIR"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
export MERA_DATASETS_CACHE
export CUDA_VISIBLE_DEVICES
export MERA_FOLDER
export MERA_MODEL_STRING="${MERA_MODEL_STRING:-pretrained=${MODEL_SOURCE},trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=2,data_parallel_size=1,gpu_memory_utilization=0.50,max_model_len=32768}"
export MERA_COMMON_SETUP="${MERA_COMMON_SETUP:---model vllm --device cuda --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity INFO --apply_chat_template --fewshot_as_multiturn}"

run_in_dir "$REPO_DIR" bash "$BENCHMARK_SCRIPT"

if [[ "$RUN_LOCAL_SCORING" == "1" ]]; then
  if [[ "$MODE" == "regular" || "$MODE" == "all" ]]; then
    REGULAR_ZIP="${MERA_FOLDER}_submission.zip"
    REGULAR_RES="${MERA_FOLDER}_submission_results.json"
    run_in_dir "$REPO_DIR/modules/scoring" python evaluate_submission.py --config_path configs/main.yaml --submission_path "$REGULAR_ZIP" --results_path "$REGULAR_RES"
  fi
  if [[ "$MODE" == "gen" || "$MODE" == "all" ]]; then
    GEN_ZIP="${MERA_FOLDER}_gen_submission.zip"
    GEN_RES="${MERA_FOLDER}_gen_submission_results.json"
    run_in_dir "$REPO_DIR/modules/scoring" python evaluate_submission.py --config_path configs/main.yaml --submission_path "$GEN_ZIP" --results_path "$GEN_RES"
  fi
fi

echo "mode=$MODE"
echo "model_id=$MODEL_ID"
echo "model_source=$MODEL_SOURCE"
echo "model_cache_dir=$MODEL_CACHE_DIR"
echo "mera_datasets_cache=$MERA_DATASETS_CACHE"
echo "mera_folder=$MERA_FOLDER"
