#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_PREFIX="${OLLAMA_PREFIX:-.local/ollama}"
OLLAMA_MODELS="${OLLAMA_MODELS:-.ollama_models}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BIN="$OLLAMA_PREFIX/bin/ollama"

"$ROOT_DIR/scripts/start_local_ollama.sh"

MODELS=(
  "qwen3-vl:8b-instruct-q4_K_M"
  "qwen3-vl:4b-instruct-q4_K_M"
  "gemma3:4b-it-q4_K_M"
)

for model_name in "${MODELS[@]}"; do
  echo "Pulling $model_name"
  OLLAMA_HOST="$OLLAMA_HOST" \
    OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$OLLAMA_BIN" pull "$model_name"
done

OLLAMA_HOST="$OLLAMA_HOST" \
  OLLAMA_MODELS="$OLLAMA_MODELS" \
  "$OLLAMA_BIN" list
