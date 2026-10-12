#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <noise_condition>" >&2
  exit 2
fi

CONDITION="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
SEED="${SEED:-7}"
EPOCHS="${EPOCHS:-100}"
SAVE_FRE="${SAVE_FRE:-1000}"
TEACHER_RUN_ROOT="${TEACHER_RUN_ROOT:-${ROOT_DIR}/new_runs/segmentation_noise/teacher}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/new_runs/segmentation_noise/student}"
TEACHER_CKPT="${TEACHER_CKPT:-${TEACHER_RUN_ROOT}/${CONDITION}_seed${SEED}/best_loss_des_trans.pth}"
RUN_DIR="${RUN_ROOT}/${CONDITION}_seed${SEED}"
OBJECT_STEM="final_object_qwen3_14b_segmentation_noise_points_1024"
TEACHER_OBJECT_POINTS="../segmentation_noise/${CONDITION}/{split}/${OBJECT_STEM}.pth"
TEACHER_OBJECT_MASK="../segmentation_noise/${CONDITION}/{split}/${OBJECT_STEM}_mask.pth"
STUDENT_OBJECT_POINTS="final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_points_1024.pth"
STUDENT_OBJECT_MASK="final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_points_1024_mask.pth"

if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "missing teacher checkpoint: ${TEACHER_CKPT}" >&2
  exit 1
fi
if [[ -d "${RUN_DIR}" ]] && [[ -n "$(ls -A "${RUN_DIR}")" ]]; then
  echo "refusing to reuse non-empty run directory: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_dual_old_0351_kd_third1423.py \
  --use_kd \
  --teacher_load_model_dir "${TEACHER_CKPT}" \
  --teacher_object_points_name "${TEACHER_OBJECT_POINTS}" \
  --teacher_object_mask_name "${TEACHER_OBJECT_MASK}" \
  --object_points_name "${STUDENT_OBJECT_POINTS}" \
  --object_mask_name "${STUDENT_OBJECT_MASK}" \
  --kd_feat_out_tia_weight 1.0 \
  --kd_feat_out_sca_weight 0.0 \
  --kd_feat_object_weight 0.0 \
  --kd_feat_motion_enc_weight 0.0 \
  --kd_out_traj_weight 0.5 \
  --kd_out_traj_dest_weight 1.0 \
  --kd_out_pose_weight 0.0 \
  --kd_feat_kind smooth_l1 \
  --use_pose_loss \
  --val_fre 1 \
  --save_fre "${SAVE_FRE}" \
  --compact_checkpoints \
  --epoch "${EPOCHS}" \
  --seed "${SEED}" \
  --save_path "${RUN_DIR}"
