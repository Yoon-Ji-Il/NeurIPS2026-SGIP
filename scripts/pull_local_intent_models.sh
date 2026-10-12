#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_PREFIX="${OLLAMA_PREFIX:-.local/ollama}"
OLLAMA_MODELS="${OLLAMA_MODELS:-.ollama_models}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BIN="$OLLAMA_PREFIX/bin/ollama"

"$ROOT_DIR/scripts/start_local_ollama.sh"

MODELS=(
  "phi4:14b"
  "gemma3:12b"
)

for MODEL_NAME in "${MODELS[@]}"; do
  echo "Pulling $MODEL_NAME"
  OLLAMA_HOST="$OLLAMA_HOST" \
    OLLAMA_MODELS="$OLLAMA_MODELS" \
    "$OLLAMA_BIN" pull "$MODEL_NAME"
done

OLLAMA_HOST="$OLLAMA_HOST" \
  OLLAMA_MODELS="$OLLAMA_MODELS" \
  "$OLLAMA_BIN" list
