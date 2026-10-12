#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${REPO_ROOT}/runs_hoim3/hoim3_hikscale_direct24_qwen3vl4b_teacher_seed7"

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}"

exec env CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" -u \
  train_hoim3_direct24_1s2s.py \
  --gpu 0 \
  --object_points_name \
    final_object_qwen3_vl_4b_hoim3_all_objects_no_cone_points_1024.pth \
  --object_mask_name \
    final_object_qwen3_vl_4b_hoim3_all_objects_no_cone_points_1024_mask.pth \
  --use_pose_loss \
  --epoch 30 \
  --val_fre 1 \
  --early_stop_patience 0 \
  --seed 7 \
  --save_path "${RUN_DIR}"
