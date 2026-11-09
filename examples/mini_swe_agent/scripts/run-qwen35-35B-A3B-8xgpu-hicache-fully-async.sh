#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5 Mini SWE: four training GPUs and four rollout GPUs, 128K context.
# Launch through scripts/entrypoint/ray-job.sh; reuse the Mini SWE data/server setup.
# AGENTIC_SCHEDULING=group selects the matched whole-Group scheduling control.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export MINI_SWE_TRAIN_MODE=fully_async
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/mini_swe_agent_hicache}"
export EXP_NAME="${EXP_NAME:-qwen35-35B-A3B-mini-swe-8xgpu-fully-async}"
export RUN_ID="${RUN_ID:-${EXP_NAME}-$(date +%Y%m%d-%H%M%S)}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
export MAX_STALENESS="${MAX_STALENESS:-1}"
export AGENT_SERVER_MAX_TURNS="${AGENT_SERVER_MAX_TURNS:-100}"
export AGENT_SERVER_TRAIN_CONCURRENCY="${AGENT_SERVER_TRAIN_CONCURRENCY:-128}"
export AGENT_SERVER_REWARD_CONCURRENCY="${AGENT_SERVER_REWARD_CONCURRENCY:-24}"
export AGENT_SERVER_SUPERVISED_SESSIONS=1
export AGENT_SERVER_SESSION_MEMORY_MIB="${AGENT_SERVER_SESSION_MEMORY_MIB:-8192}"
export AGENT_SERVER_RESUMABLE_GENERATION=1
export AGENT_SERVER_LOCAL_MODEL_COST_MAP=True
export AGENT_CLIENT_RESOURCE_PROTOCOL=1
export AGENT_SERVER_SEED="${AGENT_SERVER_SEED:-42}"
export AGENT_SERVER_SHUFFLE=1
export ENABLE_EVAL="${ENABLE_EVAL:-0}"
export ENABLE_DEBUG_DUMP="${ENABLE_DEBUG_DUMP:-0}"
export ENABLE_CHECKPOINT_SAVE=0
export ENABLE_CLEARML="${ENABLE_CLEARML:-1}"
export HICACHE_DEPLOYMENT=regular
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-4194304}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${MAX_CONTEXT_LEN}}"
LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
NUM_ITERS_PER_TRAIN_UPDATE="${NUM_ITERS_PER_TRAIN_UPDATE:-8}"
AGENTIC_SCHEDULING="${AGENTIC_SCHEDULING:-incremental}"
for value in "${GLOBAL_BATCH_SIZE}" "${MAX_CONTEXT_LEN}" "${AGENT_SERVER_TRAIN_CONCURRENCY}" \
   "${MAX_TOKENS_PER_GPU}" "${LOG_PROBS_MAX_TOKENS_PER_GPU}" "${NUM_ITERS_PER_TRAIN_UPDATE}"; do
   [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Batch, capacity and token budgets must be positive integers" >&2; exit 1; }
done
((GLOBAL_BATCH_SIZE % (8 * NUM_ITERS_PER_TRAIN_UPDATE) == 0 && AGENT_SERVER_TRAIN_CONCURRENCY % 8 == 0)) || {
   echo "GBS must contain whole Groups per streaming batch; physical slots must be divisible by eight" >&2; exit 1;
}
((MAX_TOKENS_PER_GPU >= MAX_CONTEXT_LEN && LOG_PROBS_MAX_TOKENS_PER_GPU >= MAX_CONTEXT_LEN)) || {
   echo "Both TP4/CP1 token budgets must fit the complete context" >&2; exit 1;
}
[[ "${MAX_STALENESS}" =~ ^[0-9]+$ ]] || { echo "MAX_STALENESS must be nonnegative" >&2; exit 1; }
export ROLLOUT_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / 8))
export OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * 3 / 2))}"
SCHEDULING_ARGS=()
case "${AGENTIC_SCHEDULING}" in
   group)
      export AGENTIC_CONCURRENCY=$((AGENT_SERVER_TRAIN_CONCURRENCY / 8))
      unset AGENTIC_SESSION_CAPACITY_PER_NODE
      ;;
   atomic|incremental)
      export AGENTIC_CONCURRENCY="${OVER_SAMPLING_BATCH_SIZE}"
      export AGENTIC_SESSION_CAPACITY_PER_NODE="${AGENT_SERVER_TRAIN_CONCURRENCY}"
      SCHEDULING_ARGS+=(--agentic-session-capacity-per-node "${AGENTIC_SESSION_CAPACITY_PER_NODE}")
      if [[ "${AGENTIC_SCHEDULING}" == incremental ]]; then
         SCHEDULING_ARGS+=(
            --agentic-incremental-session-start
            --agentic-eager-session-generation
         )
      fi
      ;;
   *) echo "AGENTIC_SCHEDULING must be group, atomic or incremental" >&2; exit 1 ;;
esac

# One optimizer update per complete partition; eight TQ batches overlap its
# forward/backward work with remaining rollout, without increasing update count.
exec bash "${EXAMPLE_DIR}/run_mini_swe_agent_hicache.sh" \
   --sglang-config "${EXAMPLE_DIR}/hicache-cpu-2x-tp2-blackwell.yaml" \
   --tensor-model-parallel-size 4 \
   --context-parallel-size 1 \
   --pipeline-model-parallel-size 1 \
   --expert-model-parallel-size 4 \
   --expert-tensor-parallel-size 1 \
   --num-steps-per-rollout 1 \
   --num-iters-per-train-update "${NUM_ITERS_PER_TRAIN_UPDATE}" \
   --linear-cp-mode headwise \
   --seq-length "${MAX_CONTEXT_LEN}" \
   --sglang-context-length "${MAX_CONTEXT_LEN}" \
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
   --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}" \
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}" \
   --balance-data \
   --rollout-seed 42 \
   "${SCHEDULING_ARGS[@]}" \
   "$@"
