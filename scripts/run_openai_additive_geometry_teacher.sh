#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [OPENAI_MODEL]" >&2
  echo "Example: $0 gpt-5.6-terra" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
model_name="${1:-gpt-5.6-terra}"
model_slug="${model_name//\//_}"
model_slug="${model_slug//:/_}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if [[ ! -t 0 ]]; then
    echo "OPENAI_API_KEY is not set and no interactive terminal is available." >&2
    exit 2
  fi
  read -r -s -p "OpenAI API key (input hidden): " OPENAI_API_KEY
  echo
  if [[ -z "${OPENAI_API_KEY}" ]]; then
    echo "OpenAI API key cannot be empty." >&2
    exit 2
  fi
  export OPENAI_API_KEY
fi

python_bin="${PYTHON_BIN:-python}"
openai_base_url="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
output_dir="${OUTPUT_DIR:-${repo_dir}/runs/openai_additive_geometry/${model_slug}}"
reasoning_effort="${OPENAI_REASONING_EFFORT:-none}"
max_output_tokens="${MAX_NEW_TOKENS:-4096}"
timeout_seconds="${TIMEOUT_SECONDS:-600}"
sample_retries="${SAMPLE_RETRIES:-2}"
max_samples="${MAX_SAMPLES:--1}"

echo "provider=openai"
echo "model=${model_name}"
echo "output=${output_dir}"
echo "reasoning_effort=${reasoning_effort}"
echo "api_response_storage=false"
echo "selection=semantic(match+ease) + fixed geometry/motion weighted sum"
echo "tie_break=false"

exec "${python_bin}" "${script_dir}/run_remote_additive_geometry_teacher.py" \
  --backend openai \
  --model-name "${model_name}" \
  --openai-base-url "${openai_base_url}" \
  --openai-reasoning-effort "${reasoning_effort}" \
  --max-new-tokens "${max_output_tokens}" \
  --timeout "${timeout_seconds}" \
  --sample-retries "${sample_retries}" \
  --max-samples "${max_samples}" \
  --output-dir "${output_dir}"
