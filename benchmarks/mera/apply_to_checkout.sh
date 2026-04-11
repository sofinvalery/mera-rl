#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/MERA" >&2
  exit 1
fi

TARGET="$(cd "$1" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -d "$TARGET/scripts" ]]; then
  echo "target does not look like a MERA checkout: $TARGET" >&2
  exit 1
fi

mkdir -p "$TARGET/scripts"
mkdir -p "$TARGET/lm-evaluation-harness/lm_eval/models"

cp "$HERE/scripts/run_benchmark.sh" "$TARGET/scripts/"
cp "$HERE/scripts/run_benchmark_all.sh" "$TARGET/scripts/"
cp "$HERE/scripts/run_benchmark_gen.sh" "$TARGET/scripts/"
cp "$HERE/scripts/run_benchmark_all_localscore.sh" "$TARGET/scripts/"
cp "$HERE/scripts/run_benchmark_fixed.sh" "$TARGET/scripts/"
cp "$HERE/lm-eval-patches/lm_eval/models/vllm_causallms.py" \
  "$TARGET/lm-evaluation-harness/lm_eval/models/vllm_causallms.py"

chmod +x "$TARGET/scripts/run_benchmark.sh"
chmod +x "$TARGET/scripts/run_benchmark_all.sh"
chmod +x "$TARGET/scripts/run_benchmark_gen.sh"
chmod +x "$TARGET/scripts/run_benchmark_all_localscore.sh"
chmod +x "$TARGET/scripts/run_benchmark_fixed.sh"

echo "applied MERA benchmark sync to $TARGET"
