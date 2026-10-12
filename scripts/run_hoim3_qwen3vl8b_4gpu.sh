#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE="${REPO_ROOT}/scripts/run_hoim3_qwen3vl8b_teacher.py"
RUN_ROOT="${REPO_ROOT}/runs_hoim3/qwen3vl8b_teacher"
MODEL_ROOT="pretrained_models/Qwen3-VL-8B-Instruct"
GPU_IDS=(2 3 4 5)
NUM_SHARDS=4

export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "${REPO_ROOT}"
mkdir -p \
  "${RUN_ROOT}/train/shards" \
  "${RUN_ROOT}/test/shards" \
  "${RUN_ROOT}/logs"

run_shard() {
  local shard_id="$1"
  local gpu_id="$2"
  local train_output="${RUN_ROOT}/train/shards/shard_$(printf '%03d' "${shard_id}").jsonl"
  local test_output="${RUN_ROOT}/test/shards/shard_$(printf '%03d' "${shard_id}").jsonl"

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${PIPELINE}" infer \
    --input-jsonl "${RUN_ROOT}/train/inference_manifest_materialized.jsonl" \
    --output-jsonl "${train_output}" \
    --model "${MODEL_ROOT}" \
    --device cuda:0 \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard_id}" \
    --video-frames 12 \
    --retries 2 \
    --teacher-temperature 20.0 \
    --allow-geometry-fallback \
    --resume

  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${PIPELINE}" infer \
    --input-jsonl "${RUN_ROOT}/test/inference_manifest_materialized.jsonl" \
    --output-jsonl "${test_output}" \
    --model "${MODEL_ROOT}" \
    --device cuda:0 \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${shard_id}" \
    --video-frames 12 \
    --retries 2 \
    --teacher-temperature 20.0 \
    --allow-geometry-fallback \
    --resume
}

pids=()
for shard_id in 0 1 2 3; do
  gpu_id="${GPU_IDS[${shard_id}]}"
  run_shard "${shard_id}" "${gpu_id}" \
    >"${RUN_ROOT}/logs/shard_$(printf '%03d' "${shard_id}").log" 2>&1 &
  shard_pid=$!
  pids+=("${shard_pid}")
  echo "shard=${shard_id} physical_gpu=${gpu_id} pid=${shard_pid}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${PYTHON_BIN}" "${PIPELINE}" merge \
  --input-glob "${RUN_ROOT}/train/shards/shard_*.jsonl" \
  --expected-count 3332 \
  --output-jsonl "${RUN_ROOT}/train/selections.jsonl" \
  --summary-json "${RUN_ROOT}/train/selection_summary.json"

"${PYTHON_BIN}" "${PIPELINE}" merge \
  --input-glob "${RUN_ROOT}/test/shards/shard_*.jsonl" \
  --expected-count 689 \
  --output-jsonl "${RUN_ROOT}/test/selections.jsonl" \
  --summary-json "${RUN_ROOT}/test/selection_summary.json"

"${PYTHON_BIN}" "${PIPELINE}" export \
  --selection-root "${RUN_ROOT}" \
  --split all
