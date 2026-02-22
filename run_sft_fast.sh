#!/usr/bin/env bash
set -euo pipefail

cd /workspace/rl/mera-rl
source env.sh

export WANDB_PROJECT=rl
export WANDB_ENTITY=sofinvalery
export CUDA_VISIBLE_DEVICES=0,1

PY=/workspace/rl/mera-rl/_deps/prime-rl/.venv/bin/python

mkdir -p /workspace/rl/mera-rl/outputs/logs

$PY /workspace/rl/mera-rl/mera/scripts/sft.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --tasks chegeka mamuramu mathlogicqa multiq parus rcb rumultiar ruopenbookqa rutie ruworldtree rwsd use \
  --epochs 1 \
  --batch-size 8 \
  --grad-accum 2 \
  --wandb \
  --wandb-project rl \
  --wandb-entity sofinvalery \
  --wandb-run-name sft_qwen3_4b_bs8ga2 \
  --save-merged \
  --output-dir /workspace/rl/mera-rl/outputs/full_01_sft_bs8ga2 \
  2>&1 | tee /workspace/rl/mera-rl/outputs/logs/sft_qwen3_4b_bs8ga2.log
