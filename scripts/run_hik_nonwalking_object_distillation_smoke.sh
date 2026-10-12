#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SMOKE_ROOT="${SMOKE_ROOT:-${REPO_ROOT}/runs_hik/_smoke_nonwalking_object_distillation}"
GPU="${GPU:-3}"
PYTHON="${PYTHON:-python}"
LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:-4}"

"${PYTHON}" scripts/run_hik_qwen3_14b_candidate_scores.py \
  --backend mock \
  --split all \
  --limit-per-split "${LIMIT_PER_SPLIT}" \
  --no-resume \
  --output-root "${SMOKE_ROOT}/semantics"

"${PYTHON}" scripts/build_hik_nonwalking_teacher_posteriors.py \
  --semantic-root "${SMOKE_ROOT}/semantics" \
  --output-dir "${SMOKE_ROOT}/teacher_soft" \
  --limit-per-split "${LIMIT_PER_SPLIT}"

STUDENT_FEATURES="geometry_score,root_bbox_distance_3d,head_bbox_distance_3d,hand_bbox_distance_3d,root_horizontal_alignment,torso_horizontal_alignment,face_horizontal_alignment,torso_cone_fraction,face_cone_fraction,root_heading_cone_fraction,root_line_distance_xz,torso_line_distance_xz,face_line_distance_xz,label_sittable,label_whiteboard,label_sink,label_cupboard,label_coffee_machine,label_occluder,label_table,label_dishwasher,label_microwave,label_kettle,label_trash,label_drawer"

"${PYTHON}" scripts/train_same_space_student_selector.py \
  --input-csv "${SMOKE_ROOT}/teacher_soft/teacher_candidate_scores.csv" \
  --output-dir "${SMOKE_ROOT}/student" \
  --feature-columns "${STUDENT_FEATURES}" \
  --teacher-target-field posterior \
  --teacher-target-mode probability \
  --teacher-selected-field is_hard_teacher_selected \
  --epochs 2 \
  --batch-size 4 \
  --hidden-dim 32 \
  --lambda-kl 1.0 \
  --lambda-teacher-ce 0.8 \
  --lambda-rank 0.7 \
  --lambda-gt-ce 0.0 \
  --device "cuda:${GPU}" \
  --checkpoint-metric train_kl

"${PYTHON}" dataset/create_hik_nonwalking_student_object_points.py \
  --student-summary-csv "${SMOKE_ROOT}/student/sample_summary.csv" \
  --output-root "${SMOKE_ROOT}/cache" \
  --output-name smoke_student_points_1024.pth \
  --limit-per-split "${LIMIT_PER_SPLIT}"

echo "Smoke test passed: ${SMOKE_ROOT}"
