#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATAROOT="${DATAROOT:-data/GIMO}"
SMPLX_PATH="${SMPLX_PATH:-data/body_models}"
VPOSER_PATH="${VPOSER_PATH:-vposer_v1_0}"
OBJECT_POINTS_NAME="${OBJECT_POINTS_NAME:-final_object_qwen3_14b_motion_c_final_points_1024.pth}"
SAVE_PATH="${SAVE_PATH:-runs_dh/teacher_motion_c_final_seed7_retrain}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-3e-4}"
SEED="${SEED:-7}"

"${PYTHON_BIN}" train_dual.py \
  --dataroot "${DATAROOT}" \
  --smplx_path "${SMPLX_PATH}" \
  --vposer_path "${VPOSER_PATH}" \
  --object_points_name "${OBJECT_POINTS_NAME}" \
  --save_path "${SAVE_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --seed "${SEED}" \
  "$@"
