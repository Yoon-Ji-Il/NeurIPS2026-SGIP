#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="data/GIMO/SLICES_888s/segmentation_noise"
SOFT_ROOT="$ROOT/object_selector/teacher_targets/segmentation_noise_soft"
RUN_ROOT="$ROOT/object_selector/same_space_runs/segmentation_noise_soft_kl"
PYTHON="${PYTHON:-python}"
FEATURES="gaze_prompt_score,bbox_hit_ratio,last_bbox_hit,mean_bbox_distance,last_bbox_distance,mean_centroid_distance,last_centroid_distance,distance,head_centroid_distance_3d,root_alignment,torso_alignment,face_normal_alignment,root_torso_alignment,bbox_size_x,bbox_size_y,bbox_size_z,bbox_volume"
OBJECT_NAME="final_object_qwen3_14b_soft_kl_student_points_1024.pth"

REQUESTED_CONDITION="${1:-all}"
DEVICE="${2:-cpu}"
CONDITIONS=(
  random_instance_deletion_10pct
  random_instance_deletion_20pct
  target_neighbor_merge
  target_two_way_split
  partial_noisy_target_mask
)

cd "$ROOT"

for CONDITION in "${CONDITIONS[@]}"; do
  if [[ "$REQUESTED_CONDITION" != "all" && "$REQUESTED_CONDITION" != "$CONDITION" ]]; then
    continue
  fi

  INPUT_CSV="$SOFT_ROOT/$CONDITION/teacher_candidate_scores_hard_consistent.csv"
  OUTPUT_DIR="$RUN_ROOT/$CONDITION"
  if [[ ! -f "$INPUT_CSV" ]]; then
    echo "Missing Qwen soft distribution: $INPUT_CSV" >&2
    exit 1
  fi

  "$PYTHON" scripts/train_same_space_student_selector.py \
    --input-csv "$INPUT_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --feature-columns "$FEATURES" \
    --teacher-target-field posterior_hard_consistent \
    --teacher-target-mode probability \
    --teacher-temperature 1.0 \
    --distillation-temperature 4.0 \
    --checkpoint-metric train_kl \
    --epochs 200 \
    --batch-size 32 \
    --hidden-dim 256 \
    --dropout 0.0 \
    --lambda-kl 1.0 \
    --lambda-rank 0.0 \
    --lambda-pairgap 0.0 \
    --lambda-teacher-ce 0.0 \
    --lambda-gt-ce 0.0 \
    --seed 7 \
    --device "$DEVICE"

  "$PYTHON" dataset/create_samplewise_noise_object_points.py \
    --selector-summary-csv "$OUTPUT_DIR/sample_summary.csv" \
    --scene-graph-root "$ROOT/object_selector/scene_graphs/segmentation_noise/$CONDITION" \
    --output-root "$DATA_ROOT/$CONDITION" \
    --gimo-dataset-csv data/GIMO/dataset.csv \
    --gimo-root data/GIMO \
    --num-object-points 1024 \
    --output-name "$OBJECT_NAME" \
    --train-reference-manifest "$DATA_ROOT/$CONDITION/train/final_object_qwen3_14b_motion_c_final_points_1024_manifest.json" \
    --test-reference-manifest "$DATA_ROOT/$CONDITION/test/final_object_qwen3_14b_motion_c_final_points_1024_manifest.json"
done
