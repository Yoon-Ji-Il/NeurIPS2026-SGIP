#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT/object_selector/teacher_targets/qwen3_14b_gaze_scene_graph_v1}"
OBJECT_NAME="${2:-final_object_qwen3_14b_gaze_scene_graph_points_1024.pth}"


cd "$ROOT"

# Activate an environment with the optional selector dependencies before running.
python scripts/run_gaze_qwen_object_selector.py \
  --backend qwen \
  --model-name Qwen/Qwen3-14B \
  --quantization 4bit \
  --torch-dtype bfloat16 \
  --max-new-tokens 1024 \
  --split all \
  --output-dir "$OUTPUT_DIR"

python dataset/create_final_object_points_from_selector.py \
  --selector-summary-csv "$OUTPUT_DIR/teacher_selection_summary.csv" \
  --gimo-dataset-csv data/GIMO/dataset.csv \
  --gimo-root data/GIMO \
  --openins3d-root data/OpenIns3D/output/results_demo \
  --output-root data/GIMO/SLICES_888s \
  --output-name "$OBJECT_NAME"
