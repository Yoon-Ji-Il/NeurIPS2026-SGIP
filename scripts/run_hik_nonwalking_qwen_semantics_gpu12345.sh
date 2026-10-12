#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
SHARD_ROOT="${SHARD_ROOT:-${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics_shards}"
MERGED_ROOT="${MERGED_ROOT:-${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics}"
GPUS=(1 2 3 4 5)
NUM_SHARDS="${#GPUS[@]}"

mkdir -p "${SHARD_ROOT}/logs"
pids=()
for shard_index in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard_index]}"
  shard_output="${SHARD_ROOT}/shard_$(printf '%03d' "${shard_index}")"
  log_path="${SHARD_ROOT}/logs/shard_$(printf '%03d' "${shard_index}").log"
  "${PYTHON}" scripts/run_hik_qwen3_14b_candidate_scores.py \
    --backend qwen \
    --model-name "${MODEL_NAME}" \
    --device "cuda:${gpu}" \
    --quantization 4bit \
    --torch-dtype bfloat16 \
    --split all \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard_index}" \
    --output-root "${shard_output}" \
    > "${log_path}" 2>&1 &
  pids+=("$!")
  echo "shard=${shard_index} gpu=${gpu} pid=${pids[-1]} log=${log_path}"
done

failure=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failure=1
  fi
done
if [[ "${failure}" -ne 0 ]]; then
  echo "At least one Qwen shard failed. Inspect ${SHARD_ROOT}/logs." >&2
  exit 1
fi

shard_roots=()
for shard_index in "${!GPUS[@]}"; do
  shard_roots+=("${SHARD_ROOT}/shard_$(printf '%03d' "${shard_index}")")
done

"${PYTHON}" scripts/merge_hik_candidate_semantic_shards.py \
  --shard-roots "${shard_roots[@]}" \
  --output-root "${MERGED_ROOT}"

echo "Merged HIK non-walking Qwen semantics: ${MERGED_ROOT}"
