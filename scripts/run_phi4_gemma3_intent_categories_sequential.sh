#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/intent_category_models}"

MODELS=(
  "phi4:14b"
  "gemma3:12b"
)

RUN_DIRS=()
for MODEL_NAME in "${MODELS[@]}"; do
  MODEL_SLUG="${MODEL_NAME//:/_}"
  MODEL_SLUG="${MODEL_SLUG//\//_}"
  OUTPUT_DIR="$RUN_ROOT/$MODEL_SLUG"
  RUN_DIRS+=("$OUTPUT_DIR")

  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_remote_intent_category_classifier.py" \
    --model-name "$MODEL_NAME" \
    --ollama-base-url "$OLLAMA_BASE_URL" \
    --output-dir "$OUTPUT_DIR" \
    --unload-after-run
done

"$PYTHON_BIN" "$ROOT_DIR/scripts/compare_intent_category_runs.py" \
  --run-dirs "${RUN_DIRS[@]}" \
  --output-dir "$RUN_ROOT/comparison"
