#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONTEXT_ROOT="data/HIK/hik_preprocessed/H25F50_nonwalking"
DIRECT_ROOT="data/HIK/hik_preprocessed/H25F50_nonwalking_direct29"
RUN_DIR="${REPO_ROOT}/runs_hik/hik_h25f50_nonwalking_direct29_disjoint_seed7_gpu4"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}"
"${PYTHON_BIN}" train_hik_direct29_1s2s.py \
    --gpu 4 \
    --dataroot "${CONTEXT_ROOT}" \
    --direct29_root "${DIRECT_ROOT}" \
    --batch_size 16 \
    --num_workers 2 \
    --epoch 30 \
    --lr 0.0001 \
    --gamma 0.98 \
    --seed 7 \
    --deterministic \
    --val_fre 1 \
    --early_stop_patience 0 \
    --direct29_loss_mode disjoint \
    --direct29_destination_weight 1.0 \
    --direct29_root_weight 1.0 \
    --direct29_pose_weight 1.0 \
    --direct29_yaw_weight 0.1 \
    --gradient_audit_batches 3 \
    --eval_before_train \
    --save_path "${RUN_DIR}" \
    >"${RUN_DIR}.log" 2>&1
