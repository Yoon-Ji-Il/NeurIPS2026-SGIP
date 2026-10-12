#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

bash scripts/run_hik_direct29_legacy_overlap_gpu3.sh &
legacy_pid=$!
bash scripts/run_hik_direct29_disjoint_gpu4.sh &
disjoint_pid=$!

echo "legacy_overlap gpu3 pid=${legacy_pid}"
echo "disjoint gpu4 pid=${disjoint_pid}"

wait "${legacy_pid}"
wait "${disjoint_pid}"
