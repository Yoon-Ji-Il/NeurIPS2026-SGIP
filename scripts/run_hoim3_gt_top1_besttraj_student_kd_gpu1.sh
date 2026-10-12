#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="data/HOIm3/ours_hikscale_direct24_h30f60/SLICES_888s"

TEACHER_OBJECT_NAME="final_object_gt_top1_interaction_strength_hoim3_points_1024.pth"
TEACHER_OBJECT_MASK_NAME="final_object_gt_top1_interaction_strength_hoim3_points_1024_mask.pth"
STUDENT_OBJECT_NAME="final_object_student_hoim3_all_objects_no_cone_points_1024.pth"
STUDENT_OBJECT_MASK_NAME="final_object_student_hoim3_all_objects_no_cone_points_1024_mask.pth"

TEACHER_RUN="${REPO_ROOT}/runs_hoim3/hoim3_hikscale_direct24_gt_top1_teacher_seed7"
TEACHER_CKPT="${TEACHER_RUN}/best_trajectory_mean.pth"
STUDENT_RUN="${REPO_ROOT}/runs_hoim3/hoim3_hikscale_direct24_gt_top1_besttraj_student_kd_seed7"

cd "${REPO_ROOT}"

if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "Missing teacher checkpoint: ${TEACHER_CKPT}" >&2
  exit 1
fi

for split in train test; do
  for name in \
    "${TEACHER_OBJECT_NAME}" \
    "${TEACHER_OBJECT_MASK_NAME}" \
    "${STUDENT_OBJECT_NAME}" \
    "${STUDENT_OBJECT_MASK_NAME}"; do
    if [[ ! -f "${CACHE_ROOT}/${split}/${name}" ]]; then
      echo "Missing cache: ${CACHE_ROOT}/${split}/${name}" >&2
      exit 1
    fi
  done
done

mkdir -p "${STUDENT_RUN}"

exec env CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" -u \
  train_hoim3_direct24_1s2s.py \
  --gpu 0 \
  --use_kd \
  --teacher_load_model_dir "${TEACHER_CKPT}" \
  --teacher_object_points_name "${TEACHER_OBJECT_NAME}" \
  --teacher_object_mask_name "${TEACHER_OBJECT_MASK_NAME}" \
  --object_points_name "${STUDENT_OBJECT_NAME}" \
  --object_mask_name "${STUDENT_OBJECT_MASK_NAME}" \
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
  --save_path "${STUDENT_RUN}"
