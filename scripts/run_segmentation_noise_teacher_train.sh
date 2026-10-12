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
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/new_runs/segmentation_noise/teacher}"
RUN_DIR="${RUN_ROOT}/${CONDITION}_seed${SEED}"
OBJECT_STEM="final_object_qwen3_14b_segmentation_noise_points_1024"
OBJECT_POINTS="../segmentation_noise/${CONDITION}/{split}/${OBJECT_STEM}.pth"
OBJECT_MASK="../segmentation_noise/${CONDITION}/{split}/${OBJECT_STEM}_mask.pth"

if [[ -d "${RUN_DIR}" ]] && [[ -n "$(ls -A "${RUN_DIR}")" ]]; then
  echo "refusing to reuse non-empty run directory: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u train_dual.py \
  --dataroot data/GIMO \
  --smplx_path data/body_models \
  --object_points_name "${OBJECT_POINTS}" \
  --object_mask_name "${OBJECT_MASK}" \
  --save_path "${RUN_DIR}" \
  --batch_size 8 \
  --lr 3e-4 \
  --seed "${SEED}" \
  --epoch "${EPOCHS}" \
  --val_fre 1 \
  --save_fre "${SAVE_FRE}" \
  --compact_checkpoints \
  --use_pose_loss
