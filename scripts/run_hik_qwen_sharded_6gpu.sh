#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GRAPH_ROOT="${REPO_ROOT}/runs_hik/scene_graphs_h25f50_action_geometry_v1"
SHARDS_ROOT="${REPO_ROOT}/runs_hik/qwen3_14b_action_geometry_h25f50_shards"
NUM_SHARDS=6

mkdir -p "${SHARDS_ROOT}/logs"
cd "${REPO_ROOT}"

pids=()
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_name=$(printf "shard_%03d" "${shard_index}")
    CUDA_VISIBLE_DEVICES="${shard_index}" "${PYTHON_BIN}" \
        scripts/run_hik_qwen3_14b_object_selector.py \
        --scene-graph-root "${GRAPH_ROOT}" \
        --output-root "${SHARDS_ROOT}/${shard_name}" \
        --split all \
        --backend qwen \
        --model-name Qwen/Qwen3-14B \
        --device cuda:0 \
        --torch-dtype bfloat16 \
        --quantization 4bit \
        --max-new-tokens 128 \
        --max-retries 2 \
        --num-shards "${NUM_SHARDS}" \
        --shard-index "${shard_index}" \
        >"${SHARDS_ROOT}/logs/${shard_name}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
