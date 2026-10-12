#!/bin/bash
# Leave-one-out feature ablation: drop each feature one at a time, see impact.
# Uses sweep_v2 winner config (h512_wd_strong) as the base.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

# All 14 default features
ALL_FEATURES=(
    "distance"
    "root_centroid_distance_3d"
    "head_centroid_distance_3d"
    "root_min_point_distance_3d"
    "head_min_point_distance_3d"
    "face_normal_alignment"
    "face_normal_horizontal_alignment"
    "root_alignment"
    "root_horizontal_alignment"
    "torso_alignment"
    "torso_horizontal_alignment"
    "root_torso_alignment"
    "root_line_distance_xz"
    "torso_line_distance_xz"
)

# Common args (h512_wd_strong winner config)
COMMON="--hidden-dim 512 --weight-decay 1e-3 --lr 5e-4 --epochs 80 --seed 7"

mkdir -p runs/student/loo

# (0) Baseline with all 14
echo "=== [base] all 14 ==="
$PYTHON -m object_selector_new.scripts.train_student \
    --input-csv runs/teacher_v1/teacher_restored.csv \
    --output-dir runs/student/loo/base \
    $COMMON \
    > runs/student/loo/base/train.log 2>&1

# (1) Drop one at a time
for drop in "${ALL_FEATURES[@]}"; do
    # Build feature list excluding the dropped one
    cols=""
    for f in "${ALL_FEATURES[@]}"; do
        if [ "$f" != "$drop" ]; then
            if [ -z "$cols" ]; then
                cols="$f"
            else
                cols="$cols,$f"
            fi
        fi
    done

    name="drop_${drop}"
    out="runs/student/loo/${name}"
    mkdir -p "$out"

    echo "==========================================="
    echo "[$(date +%H:%M:%S)] Drop: $drop  (using ${#ALL_FEATURES[@]} - 1 = 13 features)"
    echo "==========================================="

    $PYTHON -m object_selector_new.scripts.train_student \
        --input-csv runs/teacher_v1/teacher_restored.csv \
        --output-dir "$out" \
        --feature-columns "$cols" \
        $COMMON \
        > "$out/train.log" 2>&1 || echo "  FAILED: $name"
done

# Summary
echo ""
echo "==========================================="
echo "LEAVE-ONE-OUT SUMMARY"
echo "==========================================="
echo ""
echo "BASE (all 14):"
grep "Best epoch" runs/student/loo/base/train.log | tail -1
echo ""
echo "Drop one at a time (sorted by GT pre, lower = more important):"
for drop in "${ALL_FEATURES[@]}"; do
    log="runs/student/loo/drop_${drop}/train.log"
    if [ -f "$log" ]; then
        line=$(grep "Best epoch" "$log" | tail -1)
        printf "  drop %-40s : %s\n" "$drop" "$line"
    fi
done
