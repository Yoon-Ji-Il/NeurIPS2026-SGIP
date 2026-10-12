#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python dataset/create_gaze_final_nearest_object_points.py \
  --split all \
  --num-object-points 1024 \
  --bbox-margin 0.0 \
  --output-name final_object_gaze_final_nearest_points_1024.pth \
  --analysis-dir "$REPO_ROOT/object_selector/gaze_runs/gaze_final_nearest_object_v1"
