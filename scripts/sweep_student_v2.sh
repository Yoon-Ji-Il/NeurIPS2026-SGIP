#!/bin/bash
# Stage-2 student sweep: target sharpening + lr + capacity + GT hybrid.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python}"

declare -a RUNS=(
    "baseline                 |"
    "lr_low                   | --lr 3e-4"
    "T_S_sharper              | --teacher-temperature 0.7"
    "T_S_sharper_lr_low       | --teacher-temperature 0.7 --lr 3e-4"
    "h512_wd_strong           | --hidden-dim 512 --weight-decay 1e-3 --lr 5e-4"
    "hybrid_gt                | --lambda-gt-ce 0.3 --lr 3e-4"
    "all_in                   | --teacher-temperature 0.7 --hidden-dim 512 --weight-decay 1e-3 --lambda-gt-ce 0.3 --lr 3e-4"
)

mkdir -p runs/student/sweep_v2

for entry in "${RUNS[@]}"; do
    name=$(echo "$entry" | cut -d'|' -f1 | xargs)
    extra=$(echo "$entry" | cut -d'|' -f2- | xargs)
    out="runs/student/sweep_v2/${name}"
    mkdir -p "$out"

    echo "==========================================="
    echo "[$(date +%H:%M:%S)] Running: $name"
    echo "  args: $extra"
    echo "==========================================="

    $PYTHON -m object_selector_new.scripts.train_student \
        --input-csv runs/teacher_v1/teacher_restored.csv \
        --output-dir "$out" \
        --epochs 200 --seed 7 \
        $extra \
        > "$out/train.log" 2>&1 || echo "  FAILED: $name"
done

echo ""
echo "==========================================="
echo "SWEEP_V2 SUMMARY"
echo "==========================================="
printf "%-25s | %-60s\n" "RUN" "BEST EPOCH RESULT"
printf -- "----------------------------+-------------------------------------------------------------\n"
for entry in "${RUNS[@]}"; do
    name=$(echo "$entry" | cut -d'|' -f1 | xargs)
    log="runs/student/sweep_v2/${name}/train.log"
    if [ -f "$log" ]; then
        best=$(grep "Best epoch" "$log" | tail -1)
        printf "%-25s | %s\n" "$name" "$best"
    fi
done

echo ""
echo "=== Per-run KL trajectory (last 5 epoch lines) ==="
for entry in "${RUNS[@]}"; do
    name=$(echo "$entry" | cut -d'|' -f1 | xargs)
    log="runs/student/sweep_v2/${name}/train.log"
    if [ -f "$log" ]; then
        echo "--- $name ---"
        grep "^\[epoch" "$log" | tail -5
    fi
done
