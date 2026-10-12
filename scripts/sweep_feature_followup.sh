#!/bin/bash
# Follow-up sweep based on LOO findings:
# (a) Drop noise feature (torso_alignment)
# (b) Top-8 features only
# (c) Multi-seed variance check on best config
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

COMMON="--hidden-dim 256 --weight-decay 1e-3 --lr 5e-4 --teacher-temperature 0.7 --epochs 80"

ALL_14="distance,root_centroid_distance_3d,head_centroid_distance_3d,root_min_point_distance_3d,head_min_point_distance_3d,face_normal_alignment,face_normal_horizontal_alignment,root_alignment,root_horizontal_alignment,torso_alignment,torso_horizontal_alignment,root_torso_alignment,root_line_distance_xz,torso_line_distance_xz"

DROP_NOISE_13="distance,root_centroid_distance_3d,head_centroid_distance_3d,root_min_point_distance_3d,head_min_point_distance_3d,face_normal_alignment,face_normal_horizontal_alignment,root_alignment,root_horizontal_alignment,torso_horizontal_alignment,root_torso_alignment,root_line_distance_xz,torso_line_distance_xz"

TOP_8="head_min_point_distance_3d,root_horizontal_alignment,head_centroid_distance_3d,root_line_distance_xz,face_normal_alignment,face_normal_horizontal_alignment,root_alignment,torso_line_distance_xz"

declare -a RUNS=(
    "drop_noise_13_s7         | --feature-columns $DROP_NOISE_13 --seed 7"
    "drop_noise_13_s11        | --feature-columns $DROP_NOISE_13 --seed 11"
    "drop_noise_13_s23        | --feature-columns $DROP_NOISE_13 --seed 23"
    "top_8_s7                 | --feature-columns $TOP_8 --seed 7"
    "top_8_s11                | --feature-columns $TOP_8 --seed 11"
    "all_14_s11               | --feature-columns $ALL_14 --seed 11"
    "all_14_s23               | --feature-columns $ALL_14 --seed 23"
    "all_14_s42               | --feature-columns $ALL_14 --seed 42"
)

mkdir -p runs/student/followup

for entry in "${RUNS[@]}"; do
    name=$(echo "$entry" | cut -d'|' -f1 | xargs)
    extra=$(echo "$entry" | cut -d'|' -f2- | xargs)
    out="runs/student/followup/${name}"
    mkdir -p "$out"

    echo "==========================================="
    echo "[$(date +%H:%M:%S)] Running: $name"
    echo "==========================================="

    $PYTHON -m object_selector_new.scripts.train_student \
        --input-csv runs/teacher_v1/teacher_restored.csv \
        --output-dir "$out" \
        $COMMON \
        $extra \
        > "$out/train.log" 2>&1 || echo "  FAILED: $name"
done

echo ""
echo "==========================================="
echo "FOLLOWUP SUMMARY"
echo "==========================================="
echo ""
echo "Group 1: Drop noise (torso_alignment removed, 13 features)"
for s in s7 s11 s23; do
    log="runs/student/followup/drop_noise_13_${s}/train.log"
    [ -f "$log" ] && printf "  %-25s : %s\n" "drop_noise_13_${s}" "$(grep 'Best epoch' $log | tail -1)"
done

echo ""
echo "Group 2: Top-8 features only"
for s in s7 s11; do
    log="runs/student/followup/top_8_${s}/train.log"
    [ -f "$log" ] && printf "  %-25s : %s\n" "top_8_${s}" "$(grep 'Best epoch' $log | tail -1)"
done

echo ""
echo "Group 3: Variance on all_14 (seed sweep)"
for s in s11 s23 s42; do
    log="runs/student/followup/all_14_${s}/train.log"
    [ -f "$log" ] && printf "  %-25s : %s\n" "all_14_${s}" "$(grep 'Best epoch' $log | tail -1)"
done

echo ""
echo "Reference (from previous LOO):"
echo "  base all_14 seed 7: GT pre=18 → post=17 / 37"
