#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Comparable 20-step Mini SWE GRPO runs on one eight-GPU node.
# Launch through scripts/entrypoint/ray-job.sh with fresh EXP_ROOT and SAVE_DIR.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
export HICACHE_DEPLOYMENT="${HICACHE_DEPLOYMENT:-regular}"
# Bound the cache consistently across GPU capacities; leave room for training offload.
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-600000}"
export AGENT_SERVER_SEED="${AGENT_SERVER_SEED:-42}"
export ENABLE_EVAL="${ENABLE_EVAL:-0}"

exec bash "${RELAX_ROOT}/examples/mini_swe_agent/run_mini_swe_agent_hicache.sh" \
   --rollout-seed 42 "$@"
