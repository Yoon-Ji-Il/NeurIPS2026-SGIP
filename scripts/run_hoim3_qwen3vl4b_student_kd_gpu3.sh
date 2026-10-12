#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEACHER_CKPT="${REPO_ROOT}/runs_hoim3/hoim3_hikscale_direct24_qwen3vl4b_teacher_seed7/best_mpjpe_mean.pth"
RUN_DIR="${REPO_ROOT}/runs_hoim3/hoim3_hikscale_direct24_qwen3vl4b_student_kd_seed7"

cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}"

exec env CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" -u \
  train_hoim3_direct24_1s2s.py \
  --gpu 0 \
  --use_kd \
  --teacher_load_model_dir "${TEACHER_CKPT}" \
  --teacher_object_points_name \
    final_object_qwen3_vl_4b_hoim3_all_objects_no_cone_points_1024.pth \
  --teacher_object_mask_name \
    final_object_qwen3_vl_4b_hoim3_all_objects_no_cone_points_1024_mask.pth \
  --object_points_name \
    final_object_student_hoim3_all_objects_no_cone_points_1024.pth \
  --object_mask_name \
    final_object_student_hoim3_all_objects_no_cone_points_1024_mask.pth \
  --kd_feat_out_tia_weight 1.0 \
  --kd_feat_out_sca_weight 0.0 \
  --kd_feat_object_weight 0.0 \
  --kd_feat_motion_enc_weight 0.0 \
  --kd_out_traj_weight 0.5 \
  --kd_out_traj_dest_weight 1.0 \
  --kd_out_pose_weight 0.0 \
  --kd_feat_kind smooth_l1 \
  --use_pose_loss \
  --epoch 30 \
  --val_fre 1 \
  --early_stop_patience 0 \
  --seed 7 \
  --save_path "${RUN_DIR}"
