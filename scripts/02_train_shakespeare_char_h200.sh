#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-/home/wangyufei/anaconda3/envs/torch/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON" train.py config/train_shakespeare_char.py \
  --device=cuda \
  --dtype=bfloat16 \
  --compile=False \
  --wandb_log=False \
  --wandb_project='nanoGPT' \
  --wandb_run_name='gpt2' \
  --batch_size="${BATCH_SIZE:-256}" \
  --max_iters="${MAX_ITERS:-5000}" \
  --eval_interval="${EVAL_INTERVAL:-250}"
