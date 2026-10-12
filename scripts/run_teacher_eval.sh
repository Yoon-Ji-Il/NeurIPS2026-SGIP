#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_DIR="${RUN_DIR:-runs_dh/teacher_motion_c_final_seed7_retrain}"
RESULTS_FILE="${RESULTS_FILE:-${RUN_DIR}/best_eval_results.txt}"
BEST_OVERALL="${BEST_OVERALL:-}"
DATAROOT="${DATAROOT:-data/GIMO}"
SMPLX_PATH="${SMPLX_PATH:-data/body_models}"
VPOSER_PATH="${VPOSER_PATH:-vposer_v1_0}"
OBJECT_POINTS_NAME="${OBJECT_POINTS_NAME:-final_object_qwen3_14b_motion_c_final_points_1024.pth}"
SEED="${SEED:-7}"

COMMON_ARGS=(
  --dataroot "${DATAROOT}"
  --smplx_path "${SMPLX_PATH}"
  --vposer_path "${VPOSER_PATH}"
  --object_points_name "${OBJECT_POINTS_NAME}"
  --batch_size 1
  --seed "${SEED}"
)

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}. Train a model first or set RUN_DIR." >&2
  exit 1
fi

mapfile -t CANDIDATES < <(
  find "${RUN_DIR}" -maxdepth 1 -type f -name 'best_*.pth' ! -name 'best_overall*.pth' | sort
)

if [[ "${#CANDIDATES[@]}" -eq 0 ]]; then
  echo "No best_*.pth checkpoints found in ${RUN_DIR}" >&2
  exit 1
fi

: > "${RESULTS_FILE}"
best_ckpt=""
best_total=""

for ckpt in "${CANDIDATES[@]}"; do
  echo "===== Evaluating ${ckpt} =====" | tee -a "${RESULTS_FILE}"
  output="$(
    "${PYTHON_BIN}" eval_dual.py \
      "${COMMON_ARGS[@]}" \
      --load_model_dir "${ckpt}" \
      "$@"
  )"
  printf '%s\n' "${output}" | tee -a "${RESULTS_FILE}"

  metrics_line="$(printf '%s\n' "${output}" | grep -E '^\{.*\}$' | tail -n 1)"
  loss_total="$(
    "${PYTHON_BIN}" - "${metrics_line}" <<'PY_INNER'
import ast
import sys

metrics = ast.literal_eval(sys.argv[1])
print(metrics["loss_total"])
PY_INNER
  )"
  echo "loss_total=${loss_total}" | tee -a "${RESULTS_FILE}"

  if [[ -z "${best_total}" ]] || "${PYTHON_BIN}" - "${loss_total}" "${best_total}" <<'PY_INNER'
import sys
sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
PY_INNER
  then
    best_total="${loss_total}"
    best_ckpt="${ckpt}"
  fi
done

echo "===== Best Overall =====" | tee -a "${RESULTS_FILE}"
echo "best_ckpt=${best_ckpt}" | tee -a "${RESULTS_FILE}"
echo "best_total=${best_total}" | tee -a "${RESULTS_FILE}"
if [[ -n "${BEST_OVERALL}" ]]; then
  cp "${best_ckpt}" "${BEST_OVERALL}"
  echo "saved_to=${BEST_OVERALL}" | tee -a "${RESULTS_FILE}"
fi
