#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
SHARD_ROOT="${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics_resume_gpu013_shards"
MERGED_ROOT="${REPO_ROOT}/runs_hik/qwen3_14b_h25f50_nonwalking_candidate_semantics"
FAILED_SAMPLE_ID="D|1|cupboard|79|36477"

"${PYTHON}" scripts/run_hik_qwen3_14b_candidate_scores_fast.py \
  --model-name Qwen/Qwen3-14B \
  --device cuda:1 \
  --quantization 4bit \
  --torch-dtype bfloat16 \
  --split test \
  --sample-id "${FAILED_SAMPLE_ID}" \
  --batch-size 1 \
  --max-new-tokens 512 \
  --num-shards 3 \
  --shard-index 1 \
  --output-root "${SHARD_ROOT}/shard_001"

"${PYTHON}" scripts/merge_hik_candidate_semantic_shards.py \
  --shard-roots \
    "${SHARD_ROOT}/shard_000" \
    "${SHARD_ROOT}/shard_001" \
    "${SHARD_ROOT}/shard_002" \
  --output-root "${MERGED_ROOT}"

SKIP_QWEN=1 \
GPU=1 \
SEMANTIC_ROOT="${MERGED_ROOT}" \
bash scripts/run_hik_nonwalking_object_distillation.sh

echo "HIK non-walking final sample repair and full distillation completed."
