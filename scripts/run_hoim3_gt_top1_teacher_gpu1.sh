#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OBJECT_NAME="final_object_gt_top1_interaction_strength_hoim3_points_1024.pth"
OBJECT_MASK_NAME="final_object_gt_top1_interaction_strength_hoim3_points_1024_mask.pth"
RUN_DIR="${REPO_ROOT}/runs_hoim3/hoim3_hikscale_direct24_gt_top1_teacher_seed7"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/build_hoim3_gt_top1_object_cache.py
mkdir -p "${RUN_DIR}"

exec env CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" -u \
  train_hoim3_direct24_1s2s.py \
  --gpu 0 \
  --object_points_name "${OBJECT_NAME}" \
  --object_mask_name "${OBJECT_MASK_NAME}" \
  --use_pose_loss \
  --epoch 30 \
  --val_fre 1 \
  --early_stop_patience 0 \
  --seed 7 \
  --save_path "${RUN_DIR}"
