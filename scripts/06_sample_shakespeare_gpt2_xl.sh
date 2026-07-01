#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON" sample.py \
  --out_dir=out-shakespeare \
  --device=cuda \
  --dtype=bfloat16 \
  --compile=True \
  --num_samples="${NUM_SAMPLES:-5}" \
  --max_new_tokens="${MAX_NEW_TOKENS:-500}"
