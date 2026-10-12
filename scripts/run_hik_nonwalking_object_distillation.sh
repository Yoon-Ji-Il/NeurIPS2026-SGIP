#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-3}"
PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
SEMANTIC_ROOT="${SEMANTIC_ROOT:-${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics}"
TEACHER_SOFT_DIR="${TEACHER_SOFT_DIR:-${REPO_ROOT}/runs_hik/hik_h25f50_nonwalking_teacher_soft_motion_c}"
STUDENT_RUN="${STUDENT_RUN:-${REPO_ROOT}/runs_hik/hik_h25f50_nonwalking_student_selector_kl_seed7}"
ALIGNMENT_ROOT="${ALIGNMENT_ROOT:-data/HIK/hik_preprocessed/H25F50_nonwalking/SLICES_888s}"
STUDENT_OBJECT_NAME="${STUDENT_OBJECT_NAME:-final_object_student_hik_motion_c_kl_h25f50_nonwalking_seed7_points_1024.pth}"
LIMIT_PER_SPLIT="${LIMIT_PER_SPLIT:--1}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-300}"
SKIP_QWEN="${SKIP_QWEN:-0}"

STUDENT_FEATURES="geometry_score,root_bbox_distance_3d,head_bbox_distance_3d,hand_bbox_distance_3d,root_horizontal_alignment,torso_horizontal_alignment,face_horizontal_alignment,torso_cone_fraction,face_cone_fraction,root_heading_cone_fraction,root_line_distance_xz,torso_line_distance_xz,face_line_distance_xz,label_sittable,label_whiteboard,label_sink,label_cupboard,label_coffee_machine,label_occluder,label_table,label_dishwasher,label_microwave,label_kettle,label_trash,label_drawer"

if [[ "${SKIP_QWEN}" != "1" ]]; then
  "${PYTHON}" scripts/run_hik_qwen3_14b_candidate_scores.py \
    --backend qwen \
    --model-name "${MODEL_NAME}" \
    --device "cuda:${GPU}" \
    --quantization 4bit \
    --torch-dtype bfloat16 \
    --split all \
    --limit-per-split "${LIMIT_PER_SPLIT}" \
    --output-root "${SEMANTIC_ROOT}"
fi

"${PYTHON}" scripts/build_hik_nonwalking_teacher_posteriors.py \
  --semantic-root "${SEMANTIC_ROOT}" \
  --output-dir "${TEACHER_SOFT_DIR}" \
  --limit-per-split "${LIMIT_PER_SPLIT}" \
  --semantic-gap 4.0 \
  --temperature 1.0 \
  --minimum-logit-margin 1.0

"${PYTHON}" scripts/train_same_space_student_selector.py \
  --input-csv "${TEACHER_SOFT_DIR}/teacher_candidate_scores.csv" \
  --output-dir "${STUDENT_RUN}" \
  --feature-columns "${STUDENT_FEATURES}" \
  --teacher-target-field posterior \
  --teacher-target-mode probability \
  --teacher-temperature 1.0 \
  --distillation-temperature 1.0 \
  --teacher-selected-field is_hard_teacher_selected \
  --hidden-dim 256 \
  --batch-size 32 \
  --epochs "${STUDENT_EPOCHS}" \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --lambda-kl 1.0 \
  --lambda-teacher-ce 0.8 \
  --lambda-rank 0.7 \
  --lambda-pairgap 0.0 \
  --lambda-gt-ce 0.0 \
  --rank-topk 3 \
  --seed 7 \
  --device "cuda:${GPU}" \
  --checkpoint-metric train_kl

"${PYTHON}" dataset/create_hik_nonwalking_student_object_points.py \
  --student-summary-csv "${STUDENT_RUN}/sample_summary.csv" \
  --alignment-root "${ALIGNMENT_ROOT}" \
  --output-root "${ALIGNMENT_ROOT}" \
  --output-name "${STUDENT_OBJECT_NAME}" \
  --num-object-points 1024 \
  --limit-per-split "${LIMIT_PER_SPLIT}"

echo "HIK non-walking object distillation completed."
echo "Teacher posterior: ${TEACHER_SOFT_DIR}/teacher_candidate_scores.csv"
echo "Student selector: ${STUDENT_RUN}/best_model.pt"
echo "Student object cache: ${ALIGNMENT_ROOT}/{train,test}/${STUDENT_OBJECT_NAME}"
