#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_RUN_NAME="ours_student_train_dual_old_0351_kd_third1423"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to a trained teacher .pth file}"
SEEDS=(7 42 100 984 1004)
BATCH_TAG="${BATCH_TAG:-$(date -u +'%Y%m%d_%H%M%SZ')}"
BATCH_DIR="${ROOT_DIR}/new_runs/${BASE_RUN_NAME}_${BATCH_TAG}"
LAUNCH_LOG="${BATCH_DIR}/launcher.log"
LOCK_DIR="${ROOT_DIR}/new_runs/${BASE_RUN_NAME}_multi_seed_gpu0.lock"

mkdir -p "${ROOT_DIR}/new_runs"
mkdir -p "${BATCH_DIR}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Lock exists at ${LOCK_DIR}. Another launcher may already be running." >> "${LAUNCH_LOG}"
  exit 1
fi

cleanup() {
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] launcher start"
  echo "root_dir=${ROOT_DIR}"
  echo "python=${PYTHON_BIN}"
  echo "gpu=0"
  echo "seeds=${SEEDS[*]}"
  echo "batch_tag=${BATCH_TAG}"
  echo "batch_dir=${BATCH_DIR}"
} >> "${LAUNCH_LOG}"

cd "${ROOT_DIR}" || exit 1

for seed in "${SEEDS[@]}"; do
  run_dir="${BATCH_DIR}/seed${seed}_gpu0"
  mkdir -p "${run_dir}"

  if [ -n "$(ls -A "${run_dir}" 2>/dev/null)" ]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] seed=${seed} run_dir_not_empty=${run_dir}" >> "${LAUNCH_LOG}"
    exit 1
  fi

  {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] seed=${seed} start run_dir=${run_dir}"
  } >> "${LAUNCH_LOG}"

  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -u train_dual_old_0351_kd_third1423.py \
    --use_kd \
    --teacher_load_model_dir "${TEACHER_CHECKPOINT}" \
    --teacher_object_points_name final_object_qwen3_14b_motion_c_final_points_1024.pth \
    --object_points_name final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_points_1024.pth \
    --object_mask_name final_object_student_baseline16_torso_cone_w2_t25_gt29_seed7_points_1024_mask.pth \
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
    --save_fre 1 \
    --seed "${seed}" \
    --save_path "${run_dir}"

  status=$?
  {
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] seed=${seed} exit_status=${status}"
    if [ -d "${run_dir}" ]; then
      ls -1 "${run_dir}"/best*.pth 2>/dev/null || true
    fi
  } >> "${LAUNCH_LOG}"
done

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] launcher done" >> "${LAUNCH_LOG}"
