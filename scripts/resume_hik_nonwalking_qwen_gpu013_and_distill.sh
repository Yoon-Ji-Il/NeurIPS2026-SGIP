#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
SOURCE_SHARD_ROOT="${SOURCE_SHARD_ROOT:-${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics_fast_shards}"
RESUME_SHARD_ROOT="${RESUME_SHARD_ROOT:-${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics_resume_gpu013_shards}"
MERGED_ROOT="${MERGED_ROOT:-${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
GPUS=(0 1 3)
NUM_SHARDS="${#GPUS[@]}"

bootstrap_roots=()
for old_index in 0 1 2 3 4 5; do
  bootstrap_roots+=(
    "${SOURCE_SHARD_ROOT}/shard_$(printf '%03d' "${old_index}")"
  )
done

mkdir -p "${RESUME_SHARD_ROOT}/logs"
pids=()
for shard_index in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard_index]}"
  shard_output="${RESUME_SHARD_ROOT}/shard_$(printf '%03d' "${shard_index}")"
  log_path="${RESUME_SHARD_ROOT}/logs/shard_$(printf '%03d' "${shard_index}").log"
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
  echo "At least one resumed Qwen shard failed. Inspect ${RESUME_SHARD_ROOT}/logs." >&2
  exit 1
fi

shard_roots=()
for shard_index in "${!GPUS[@]}"; do
  shard_roots+=(
    "${RESUME_SHARD_ROOT}/shard_$(printf '%03d' "${shard_index}")"
  )
done

"${PYTHON}" scripts/merge_hik_candidate_semantic_shards.py \
  --shard-roots "${shard_roots[@]}" \
  --output-root "${MERGED_ROOT}"

echo "Merged resumed HIK non-walking Qwen semantics: ${MERGED_ROOT}"

SKIP_QWEN=1 \
GPU=3 \
SEMANTIC_ROOT="${MERGED_ROOT}" \
bash scripts/run_hik_nonwalking_object_distillation.sh

echo "HIK non-walking Qwen resume, merge, distillation, and cache export completed."
