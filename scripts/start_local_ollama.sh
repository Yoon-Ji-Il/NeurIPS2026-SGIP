#!/usr/bin/env bash
set -euo pipefail

OLLAMA_PREFIX="${OLLAMA_PREFIX:-.local/ollama}"
OLLAMA_MODELS="${OLLAMA_MODELS:-.ollama_models}"
OLLAMA_RUNTIME_DIR="${OLLAMA_RUNTIME_DIR:-.ollama_runtime}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
GPU_INDEX="${GPU_INDEX:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OLLAMA_BIN="$OLLAMA_PREFIX/bin/ollama"
LOG_FILE="$OLLAMA_RUNTIME_DIR/server.log"
PID_FILE="$OLLAMA_RUNTIME_DIR/server.pid"

if [[ ! -x "$OLLAMA_BIN" ]]; then
  echo "Ollama binary not found: $OLLAMA_BIN" >&2
  exit 1
fi

mkdir -p "$OLLAMA_MODELS" "$OLLAMA_RUNTIME_DIR"

if "$PYTHON_BIN" - "$OLLAMA_BASE_URL" 2>/dev/null <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1] + "/api/tags", timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
then
  echo "Ollama is already running at $OLLAMA_BASE_URL"
  exit 0
fi

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  OLLAMA_HOST="$OLLAMA_HOST" \
  OLLAMA_MODELS="$OLLAMA_MODELS" \
  OLLAMA_KEEP_ALIVE="30m" \
  OLLAMA_MAX_LOADED_MODELS="1" \
  OLLAMA_NUM_PARALLEL="1" \
  "$OLLAMA_BIN" serve >"$LOG_FILE" 2>&1 &
echo "$!" >"$PID_FILE"

for _ in $(seq 1 60); do
  if "$PYTHON_BIN" - "$OLLAMA_BASE_URL" 2>/dev/null <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1] + "/api/tags", timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
  then
    echo "Started Ollama at $OLLAMA_BASE_URL (PID $(<"$PID_FILE"), GPU $GPU_INDEX)"
    echo "Log: $LOG_FILE"
    exit 0
  fi
  sleep 1
done

echo "Ollama did not become ready. Check $LOG_FILE" >&2
exit 1
