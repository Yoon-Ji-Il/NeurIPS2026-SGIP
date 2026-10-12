#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GIMO_ROOT="${GIMO_ROOT:-data/GIMO}"
SLICE_ROOT="${SLICE_ROOT:-${GIMO_ROOT}/SLICES_888s}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SLICE_ROOT}}"
SCENE_GRAPH_ROOT="${SCENE_GRAPH_ROOT:-${REPO_DIR}/object_selector/scene_graphs}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${REPO_DIR}/object_selector/gaze_runs/gaze_scene_graph_distance_v1}"

"${PYTHON_BIN}" dataset/create_gaze_object_points_from_scene_graph.py \
  --gimo-root "${GIMO_ROOT}" \
  --gimo-dataset-csv "${GIMO_ROOT}/dataset.csv" \
  --slice-root "${SLICE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --scene-graph-root "${SCENE_GRAPH_ROOT}" \
  --analysis-dir "${ANALYSIS_DIR}" \
  "$@"
