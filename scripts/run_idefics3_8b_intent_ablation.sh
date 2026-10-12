#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
model_dir="${IDEFICS3_MODEL_DIR:-pretrained_models/Idefics3-8B-Llama3}"
model_name="HuggingFaceM4/Idefics3-8B-Llama3"
run_root="${RUN_ROOT:-${root_dir}/runs/vlm_intent_ablation}"
output_dir="${run_root}/idefics3_8b_llama3_bfloat16"
baseline_dir="${run_root}/qwen3_vl_8b_hf_bfloat16"
report_dir="${run_root}/report_size_matched_8b_bfloat16"
qwen_gpu="${QWEN_GPU:-1}"
idefics_gpu="${IDEFICS_GPU:-3}"

mkdir -p "${baseline_dir}" "${output_dir}"

CUDA_VISIBLE_DEVICES="${qwen_gpu}" TOKENIZERS_PARALLELISM=false \
"${python_bin}" "${root_dir}/scripts/run_hf_vlm_intent_predictions.py" \
  --model "pretrained_models/Qwen3-VL-8B-Instruct" \
  --model-name "Qwen/Qwen3-VL-8B-Instruct" \
  --architecture qwen3_vl \
  --output-dir "${baseline_dir}" \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --context-length 4096 \
  --max-new-tokens 256 \
  --seed 0 \
  --resume \
  >"${baseline_dir}/inference.log" 2>&1 &
qwen_pid="$!"

CUDA_VISIBLE_DEVICES="${idefics_gpu}" TOKENIZERS_PARALLELISM=false \
"${python_bin}" "${root_dir}/scripts/run_hf_vlm_intent_predictions.py" \
  --model "${model_dir}" \
  --model-name "${model_name}" \
  --architecture idefics3 \
  --output-dir "${output_dir}" \
  --device cuda:0 \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --context-length 4096 \
  --max-new-tokens 256 \
  --seed 0 \
  --resume \
  >"${output_dir}/inference.log" 2>&1 &
idefics_pid="$!"

failed=0
if ! wait "${qwen_pid}"; then
  echo "Qwen inference failed; inspect ${baseline_dir}/inference.log" >&2
  failed=1
fi
if ! wait "${idefics_pid}"; then
  echo "Idefics3 inference failed; inspect ${output_dir}/inference.log" >&2
  failed=1
fi
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

"${python_bin}" "${root_dir}/scripts/run_remote_intent_category_classifier.py" \
  --model-name "Qwen/Qwen3-VL-8B-Instruct" \
  --output-dir "${baseline_dir}" \
  --dataset-csv "${root_dir}/dataset/dataset_3_with_want.csv" \
  --context-length 4096 \
  --max-new-tokens 256 \
  --seed 0 \
  --skip-server-check

# All 83 predictions now exist, so the established finalizer performs no
# Ollama request and writes only the standard CSVs and metrics JSON.
"${python_bin}" "${root_dir}/scripts/run_remote_intent_category_classifier.py" \
  --model-name "${model_name}" \
  --output-dir "${output_dir}" \
  --dataset-csv "${root_dir}/dataset/dataset_3_with_want.csv" \
  --context-length 4096 \
  --max-new-tokens 256 \
  --seed 0 \
  --skip-server-check

"${python_bin}" "${root_dir}/scripts/report_vlm_intent_ablation.py" \
  --baseline-run-dir "${baseline_dir}" \
  --candidate-run-dirs "${output_dir}" \
  --output-dir "${report_dir}"

echo "Completed: ${report_dir}"
