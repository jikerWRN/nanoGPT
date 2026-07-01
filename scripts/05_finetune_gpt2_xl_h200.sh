#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON" train.py config/finetune_shakespeare.py \
  --init_from=gpt2-xl \
  --device=cuda \
  --dtype=bfloat16 \
  --compile=True \
  --batch_size="${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps="${GRAD_ACCUM_STEPS:-8}" \
  --max_iters="${MAX_ITERS:-100}" \
  --eval_interval="${EVAL_INTERVAL:-10}"
