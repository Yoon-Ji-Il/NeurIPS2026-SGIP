#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

model_name="${1:-${OLLAMA_MODEL:-qwen3:4b}}"
base_url="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
python_bin="${PYTHON_BIN:-python}"
input_jsonl="${INPUT_JSONL:-${repo_root}/runs/jsonl/dataset_v1.jsonl}"
context_length="${CONTEXT_LENGTH:-16384}"
max_new_tokens="${MAX_NEW_TOKENS:-4096}"
start_index="${START_INDEX:-0}"
end_index="${END_INDEX:--1}"
max_samples="${MAX_SAMPLES:--1}"

output_tag="${model_name//\//_}"
output_tag="${output_tag//:/_}"
output_csv="${OUTPUT_CSV:-${repo_root}/runs/remote_qwen/${output_tag}/teacher.csv}"

cd "${repo_root}"
"${python_bin}" -m object_selector_new.scripts.export_teacher_csv \
  --input-jsonl "${input_jsonl}" \
  --output-csv "${output_csv}" \
  --llm-backend ollama \
  --ollama-base-url "${base_url}" \
  --ollama-model "${model_name}" \
  --ollama-context-length "${context_length}" \
  --ollama-max-new-tokens "${max_new_tokens}" \
  --ollama-timeout 600 \
  --start-index "${start_index}" \
  --end-index "${end_index}" \
  --max-samples "${max_samples}"

if [[ "${start_index}" == "0" && "${end_index}" == "-1" && "${max_samples}" == "-1" ]]; then
  output_dir="$(dirname "${output_csv}")"
  "${python_bin}" -m object_selector_new.scripts.export_teacher_selection_summary \
    --input-jsonl "${input_jsonl}" \
    --teacher-csv "${output_csv}" \
    --output-csv "${output_dir}/selection_summary.csv" \
    --metrics-json "${output_dir}/selection_metrics.json"
fi
