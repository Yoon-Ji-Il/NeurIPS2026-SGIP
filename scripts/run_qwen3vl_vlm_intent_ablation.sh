#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/vlm_intent_ablation}"

BASELINE_MODEL="qwen3-vl:8b-instruct-q4_K_M"
CANDIDATE_MODELS=(
  "qwen3-vl:4b-instruct-q4_K_M"
  "gemma3:4b-it-q4_K_M"
)

run_model() {
  local model_name="$1"
  local model_slug="${model_name//:/_}"
  model_slug="${model_slug//\//_}"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/run_remote_intent_category_classifier.py" \
    --model-name "$model_name" \
    --ollama-base-url "$OLLAMA_BASE_URL" \
    --output-dir "$RUN_ROOT/$model_slug" \
    --unload-after-run
}

run_model "$BASELINE_MODEL"
for model_name in "${CANDIDATE_MODELS[@]}"; do
  run_model "$model_name"
done

BASELINE_SLUG="${BASELINE_MODEL//:/_}"
BASELINE_SLUG="${BASELINE_SLUG//\//_}"
CANDIDATE_RUN_DIRS=()
for model_name in "${CANDIDATE_MODELS[@]}"; do
  model_slug="${model_name//:/_}"
  model_slug="${model_slug//\//_}"
  CANDIDATE_RUN_DIRS+=("$RUN_ROOT/$model_slug")
done

"$PYTHON_BIN" "$ROOT_DIR/scripts/report_vlm_intent_ablation.py" \
  --baseline-run-dir "$RUN_ROOT/$BASELINE_SLUG" \
  --candidate-run-dirs "${CANDIDATE_RUN_DIRS[@]}" \
  --output-dir "$RUN_ROOT/report"
