#!/bin/bash
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python}"

BASE_CSV="runs/student/loo/base/candidate_scores_dedup.csv"
SUPPORT_CSV="object_selector/same_space_feature_inputs/teacher_candidate_scores_support_context_geometry.csv"
MOTION_CSV="object_selector/same_space_feature_inputs/teacher_candidate_scores_input_motion_temporal_attention6.csv"
OUT_ROOT="runs/residual/single_feature_sweep_v1"

B2="baseline_student_score,baseline_student_posterior"

declare -a FEATURES=(
    "object_support_on_larger_overlap_max"
    "object_support_on_larger_size_ratio_log"
    "object_support_on_larger_vertical_gap_abs_min"
    "object_support_on_larger_score"
    "object_support_child_on_top_log_count"
    "object_support_child_on_top_overlap_sum"
    "object_support_child_on_top_volume_ratio_sum"
    "object_support_max_child_score"
    "input_torso_object_forward_coord"
    "input_torso_object_lateral_coord"
    "obs_hand_last_bbox_distance_3d"
    "input_hand_heading_object_fraction"
    "input_face_ray_perp_mean"
    "input_head_object_cone_fraction"
    "input_torso_object_cone_fraction"
)

mkdir -p "$OUT_ROOT"

echo "Reference baseline: b2_only from feature_group_sweep_v1 = 19/37 GT, 14/37 teacher"
echo ""

for feat in "${FEATURES[@]}"; do
    name="b2_plus_${feat}"
    out="$OUT_ROOT/$name"
    cols="$B2,$feat"
    mkdir -p "$out"

    echo "==========================================="
    echo "[$(date +%H:%M:%S)] Running: $name"
    echo "  feature: $feat"
    echo "==========================================="

    $PYTHON scripts/train_same_space_residual_reranker.py \
        --baseline-candidate-csv "$BASE_CSV" \
        --feature-csv "$SUPPORT_CSV" \
        --feature-csv "$MOTION_CSV" \
        --output-dir "$out" \
        --feature-columns "$cols" \
        --hidden-dim 128 \
        --max-residual 0.6 \
        --dropout 0.0 \
        --epochs 200 \
        --lr 1e-3 \
        --weight-decay 1e-4 \
        --lambda-kl 1.0 \
        --lambda-anchor 0.7 \
        --lambda-residual-l2 0.015 \
        --lambda-rank 0.7 \
        --lambda-teacher-ce 0.8 \
        --seed 7 \
        > "$out/train.log" 2>&1

    $PYTHON - <<PY
import json
from pathlib import Path
path = Path("$out") / "summary.json"
s = json.loads(path.read_text())
print("  best_teacher_epoch=", s["best_epoch"])
print("  best_teacher_test teacher/gt=", f'{s["best_test_teacher_top1_count"]}/{s["best_test_gt_exact_count"]}')
print("  best_gt_epoch=", s["best_gt_epoch"])
print("  best_gt_test teacher/gt=", f'{s["best_gt_test_teacher_top1_count"]}/{s["best_gt_test_gt_exact_count"]}')
PY
done

echo ""
echo "==========================================="
echo "SINGLE FEATURE SWEEP SUMMARY"
echo "==========================================="
$PYTHON - <<'PY'
import csv
import json
from pathlib import Path

root = Path("runs/residual/single_feature_sweep_v1")
print(f"{'run':<44} {'best_gt':<10} {'teacher':<10} {'epoch_g':<8}")
for path in sorted(root.glob("*/summary.json")):
    s = json.loads(path.read_text())
    print(
        f"{path.parent.name:<44} "
        f"{str(s['best_gt_test_gt_exact_count']) + '/37':<10} "
        f"{str(s['best_gt_test_teacher_top1_count']) + '/37':<10} "
        f"{s['best_gt_epoch']:<8}"
    )
PY
