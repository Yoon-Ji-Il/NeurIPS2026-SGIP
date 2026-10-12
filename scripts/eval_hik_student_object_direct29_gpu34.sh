#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONTEXT_ROOT="data/HIK/hik_preprocessed/H25F50_nonwalking"
DIRECT_ROOT="data/HIK/hik_preprocessed/H25F50_nonwalking_direct29"
STUDENT_OBJECT="final_object_student_hik_motion_c_kl_h25f50_nonwalking_seed7_points_1024.pth"

cd "${REPO_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

evaluate_variant() {
  local mode="$1"
  local gpu="$2"
  local teacher_run="$3"
  local output_root="$4"

  mkdir -p "${output_root}"
  for checkpoint_name in \
    best_trajectory_mean.pth \
    best_mpjpe_mean.pth \
    best_total_loss.pth \
    last.pth
  do
    checkpoint_path="${teacher_run}/${checkpoint_name}"
    if [[ ! -f "${checkpoint_path}" ]]; then
      continue
    fi
    checkpoint_stem="${checkpoint_name%.pth}"
    eval_dir="${output_root}/${checkpoint_stem}"
    mkdir -p "${eval_dir}"
    "${PYTHON}" train_hik_direct29_1s2s.py \
      --gpu "${gpu}" \
      --dataroot "${CONTEXT_ROOT}" \
      --direct29_root "${DIRECT_ROOT}" \
      --object_points_name "${STUDENT_OBJECT}" \
      --batch_size 16 \
      --num_workers 2 \
      --seed 7 \
      --deterministic \
      --early_stop_patience 0 \
      --direct29_loss_mode "${mode}" \
      --direct29_destination_weight 1.0 \
      --direct29_root_weight 1.0 \
      --direct29_pose_weight 1.0 \
      --direct29_yaw_weight 0.1 \
      --eval_only \
      --load_model_dir "${checkpoint_path}" \
      --save_path "${eval_dir}" \
      > "${eval_dir}.log" 2>&1
  done

  "${PYTHON}" - "${output_root}" "${teacher_run}" "${mode}" <<'PY'
import json
from pathlib import Path
import sys

output_root = Path(sys.argv[1])
teacher_run = Path(sys.argv[2])
mode = sys.argv[3]
results = {}
for checkpoint_name in (
    "best_trajectory_mean.pth",
    "best_mpjpe_mean.pth",
    "best_total_loss.pth",
    "last.pth",
):
    path = output_root / checkpoint_name.removesuffix(".pth") / "baseline_eval.json"
    if path.is_file():
        results[checkpoint_name] = json.loads(path.read_text(encoding="utf-8"))
payload = {
    "evaluation": "fixed_direct29_checkpoint_with_student_object_cache",
    "loss_mode": mode,
    "teacher_checkpoint_root": str(teacher_run.resolve()),
    "student_object_points_name": (
        "final_object_student_hik_motion_c_kl_h25f50_nonwalking_"
        "seed7_points_1024.pth"
    ),
    "results": results,
}
(output_root / "final_test_metrics_by_checkpoint.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, ensure_ascii=False))
PY
}

evaluate_variant \
  legacy_overlap \
  3 \
  "${REPO_ROOT}/runs_hik/hik_h25f50_nonwalking_direct29_legacy_overlap_seed7_gpu3" \
  "${REPO_ROOT}/runs_hik/hik_student_object_eval_direct29_legacy_overlap_seed7_gpu3" &
legacy_pid=$!

evaluate_variant \
  disjoint \
  4 \
  "${REPO_ROOT}/runs_hik/hik_h25f50_nonwalking_direct29_disjoint_seed7_gpu4" \
  "${REPO_ROOT}/runs_hik/hik_student_object_eval_direct29_disjoint_seed7_gpu4" &
disjoint_pid=$!

echo "legacy_overlap gpu3 pid=${legacy_pid}"
echo "disjoint gpu4 pid=${disjoint_pid}"

wait "${legacy_pid}"
wait "${disjoint_pid}"

echo "HIK Student-object direct29 checkpoint evaluation completed."
