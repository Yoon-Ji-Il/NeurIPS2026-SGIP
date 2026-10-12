#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="data/HIK/hik_preprocessed/H25F50_nonwalking"
RUN_DIR="${REPO_ROOT}/runs_hik/hik_h25f50_nonwalking_qwen_object_seed7_gpu3"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${REPO_ROOT}"
"${PYTHON_BIN}" train_hik_1s2s.py \
    --gpu 3 \
    --dataroot "${DATA_ROOT}" \
    --batch_size 16 \
    --num_workers 2 \
    --epoch 30 \
    --lr 0.0001 \
    --gamma 0.98 \
    --seed 7 \
    --deterministic \
    --val_fre 1 \
    --early_stop_metric loss_trans \
    --early_stop_patience 7 \
    --early_stop_warmup 5 \
    --early_stop_min_delta 0.0005 \
    --des_trans_weight 1.0 \
    --mpjpe_weight 1.0 \
    --eval_before_train \
    --save_path "${RUN_DIR}" \
    >"${RUN_DIR}.log" 2>&1
