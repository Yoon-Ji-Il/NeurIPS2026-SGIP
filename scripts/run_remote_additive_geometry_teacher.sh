#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 MODEL_NAME" >&2
  echo "Example: $0 qwen3:8b" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
model_name="$1"
model_slug="${model_name//\//_}"
model_slug="${model_slug//:/_}"

python_bin="${PYTHON_BIN:-python}"
ollama_base_url="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
output_dir="${OUTPUT_DIR:-${repo_dir}/runs/remote_additive_geometry/${model_slug}}"
context_length="${CONTEXT_LENGTH:-16384}"
max_new_tokens="${MAX_NEW_TOKENS:-4096}"
timeout_seconds="${TIMEOUT_SECONDS:-600}"
keep_alive="${KEEP_ALIVE:-10m}"
sample_retries="${SAMPLE_RETRIES:-2}"
max_samples="${MAX_SAMPLES:--1}"

echo "model=${model_name}"
echo "ollama=${ollama_base_url}"
echo "output=${output_dir}"
echo "selection=semantic(match+ease) + fixed geometry/motion weighted sum"
echo "tie_break=false"

exec "${python_bin}" "${script_dir}/run_remote_additive_geometry_teacher.py" \
  --model-name "${model_name}" \
  --ollama-base-url "${ollama_base_url}" \
  --context-length "${context_length}" \
  --max-new-tokens "${max_new_tokens}" \
  --timeout "${timeout_seconds}" \
  --keep-alive "${keep_alive}" \
  --sample-retries "${sample_retries}" \
  --max-samples "${max_samples}" \
  --output-dir "${output_dir}"
