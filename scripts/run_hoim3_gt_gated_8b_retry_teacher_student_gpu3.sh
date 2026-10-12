#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="data/HOIm3/ours_hikscale_direct24_h30f60/SLICES_888s"
SELECTION_ROOT="${REPO_ROOT}/runs_hoim3/qwen3vl8b_gt_gated_retry"
OBJECT_NAME="final_object_qwen3_vl_4b_gtmatch_8b_retry_gtgated_hoim3_all_objects_no_cone_points_1024.pth"
OBJECT_MASK_NAME="final_object_qwen3_vl_4b_gtmatch_8b_retry_gtgated_hoim3_all_objects_no_cone_points_1024_mask.pth"
STUDENT_OBJECT_NAME="final_object_student_hoim3_all_objects_no_cone_points_1024.pth"
STUDENT_OBJECT_MASK_NAME="final_object_student_hoim3_all_objects_no_cone_points_1024_mask.pth"
TEACHER_RUN="${REPO_ROOT}/runs_hoim3/hoim3_hikscale_direct24_qwen4b_gtmatch_8b_retry_gtgated_teacher_seed7"
STUDENT_RUN="${REPO_ROOT}/runs_hoim3/hoim3_hikscale_direct24_qwen4b_gtmatch_8b_retry_gtgated_student_kd_seed7"

cd "${REPO_ROOT}"

echo "waiting_for_hybrid_cache=${SELECTION_ROOT}/teacher_object_cache_summary.json"
while [[ ! -f "${SELECTION_ROOT}/teacher_object_cache_summary.json" ]]; do
  sleep 10
done

for split in train test; do
  for name in \
    "${OBJECT_NAME}" \
    "${OBJECT_MASK_NAME}" \
    "${STUDENT_OBJECT_NAME}" \
    "${STUDENT_OBJECT_MASK_NAME}"; do
    if [[ ! -f "${CACHE_ROOT}/${split}/${name}" ]]; then
      echo "Missing cache: ${CACHE_ROOT}/${split}/${name}" >&2
      exit 1
    fi
  done
done

mkdir -p "${TEACHER_RUN}" "${STUDENT_RUN}"

echo "starting_gt_gated_hybrid_teacher"
env CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" -u \
  train_hoim3_direct24_1s2s.py \
  --gpu 0 \
  --object_points_name "${OBJECT_NAME}" \
  --object_mask_name "${OBJECT_MASK_NAME}" \
  --use_pose_loss \
  --epoch 30 \
  --val_fre 1 \
  --early_stop_patience 0 \
  --seed 7 \
  --save_path "${TEACHER_RUN}" \
  >"${TEACHER_RUN}/train.log" 2>&1

TEACHER_CKPT="${TEACHER_RUN}/best_mpjpe_mean.pth"
if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "Missing teacher checkpoint: ${TEACHER_CKPT}" >&2
  exit 1
fi

echo "starting_gt_gated_hybrid_student_kd"
env CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" -u \
  train_hoim3_direct24_1s2s.py \
  --gpu 0 \
  --use_kd \
  --teacher_load_model_dir "${TEACHER_CKPT}" \
  --teacher_object_points_name "${OBJECT_NAME}" \
  --teacher_object_mask_name "${OBJECT_MASK_NAME}" \
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
  --save_path "${STUDENT_RUN}" \
  >"${STUDENT_RUN}/train.log" 2>&1

echo "gt_gated_hybrid_teacher_student_complete"
