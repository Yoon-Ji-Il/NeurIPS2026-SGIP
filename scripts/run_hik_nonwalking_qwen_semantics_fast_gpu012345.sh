#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
OLD_SHARD_ROOT="${OLD_SHARD_ROOT:-${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics_shards}"
FAST_SHARD_ROOT="${FAST_SHARD_ROOT:-${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics_fast_shards}"
MERGED_ROOT="${MERGED_ROOT:-${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
GPUS=(0 1 2 3 4 5)
NUM_SHARDS="${#GPUS[@]}"

bootstrap_roots=()
for old_index in 0 1 2 3 4; do
  bootstrap_roots+=("${OLD_SHARD_ROOT}/shard_$(printf '%03d' "${old_index}")")
done

mkdir -p "${FAST_SHARD_ROOT}/logs"
pids=()
for shard_index in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard_index]}"
  shard_output="${FAST_SHARD_ROOT}/shard_$(printf '%03d' "${shard_index}")"
  log_path="${FAST_SHARD_ROOT}/logs/shard_$(printf '%03d' "${shard_index}").log"
  "${PYTHON}" scripts/run_hik_qwen3_14b_candidate_scores_fast.py \
    --model-name "${MODEL_NAME}" \
    --device "cuda:${gpu}" \
    --quantization 4bit \
    --torch-dtype bfloat16 \
    --split all \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${shard_index}" \
    --bootstrap-roots "${bootstrap_roots[@]}" \
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
  echo "At least one fast Qwen shard failed. Inspect ${FAST_SHARD_ROOT}/logs." >&2
  exit 1
fi

shard_roots=()
for shard_index in "${!GPUS[@]}"; do
  shard_roots+=("${FAST_SHARD_ROOT}/shard_$(printf '%03d' "${shard_index}")")
done

"${PYTHON}" scripts/merge_hik_candidate_semantic_shards.py \
  --shard-roots "${shard_roots[@]}" \
  --output-root "${MERGED_ROOT}"

echo "Merged fast HIK non-walking Qwen semantics: ${MERGED_ROOT}"
