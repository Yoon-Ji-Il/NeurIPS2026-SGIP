#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="$ROOT/scripts/train_segmentation_noise_soft_students.sh"
LOG_DIR="$ROOT/logs/segmentation_noise_soft_students_5gpu"

mkdir -p "$LOG_DIR"
cd "$ROOT"

CONDITIONS=(
  random_instance_deletion_10pct
  random_instance_deletion_20pct
  target_neighbor_merge
  target_two_way_split
  partial_noisy_target_mask
)
GPUS=(0 1 2 4 5)

PIDS=()
for INDEX in "${!CONDITIONS[@]}"; do
  CONDITION="${CONDITIONS[$INDEX]}"
  GPU="${GPUS[$INDEX]}"
  LOG_PATH="$LOG_DIR/$CONDITION.log"

  echo "Launching $CONDITION on physical GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" \
    bash "$WORKER" "$CONDITION" cuda:0 \
    >"$LOG_PATH" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for INDEX in "${!PIDS[@]}"; do
  CONDITION="${CONDITIONS[$INDEX]}"
  GPU="${GPUS[$INDEX]}"
  if wait "${PIDS[$INDEX]}"; then
    echo "Completed $CONDITION on physical GPU $GPU"
  else
    echo "Failed $CONDITION on physical GPU $GPU; check $LOG_DIR/$CONDITION.log" >&2
    FAILED=1
  fi
done

exit "$FAILED"
