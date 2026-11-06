#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Eight large-memory Blackwell GPUs, Qwen3.5-35B-A3B, 128K Mini SWE GRPO.
# Launch through scripts/entrypoint/ray-job.sh. Paths are supplied by the caller.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
export AGENT_SERVER_MAX_TURNS="${AGENT_SERVER_MAX_TURNS:-100}"
export AGENT_SERVER_TRAIN_CONCURRENCY="${AGENT_SERVER_TRAIN_CONCURRENCY:-128}"
export AGENT_SERVER_REWARD_CONCURRENCY="${AGENT_SERVER_REWARD_CONCURRENCY:-24}"
export AGENT_SERVER_SUPERVISED_SESSIONS=1
export AGENT_SERVER_LOCAL_MODEL_COST_MAP=True
export AGENT_CLIENT_RESOURCE_PROTOCOL=1
export AGENT_SERVER_SEED="${AGENT_SERVER_SEED:-42}"
export AGENT_SERVER_SHUFFLE=1
export ENABLE_EVAL="${ENABLE_EVAL:-0}"
export ENABLE_DEBUG_DUMP="${ENABLE_DEBUG_DUMP:-0}"
export ENABLE_CHECKPOINT_SAVE="${ENABLE_CHECKPOINT_SAVE:-0}"
export HICACHE_DEPLOYMENT=regular
# Per TP2 engine: 2M KV tokens, equivalent to 16 full 128K contexts.
# Shorter active trajectories and CPU HiCache support the larger Session pool.
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-2097152}"

# TP4 bounds the large vocabulary logits and their backward buffers at 128K.
TRAIN_TP="${TRAIN_TP:-4}"
TRAIN_CP="${TRAIN_CP:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / TRAIN_CP))}"
LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
AGENTIC_SCHEDULING="${AGENTIC_SCHEDULING:-incremental}"
for value in "${GLOBAL_BATCH_SIZE}" "${MAX_CONTEXT_LEN}" "${AGENT_SERVER_TRAIN_CONCURRENCY}" \
   "${TRAIN_TP}" "${TRAIN_CP}" "${MAX_TOKENS_PER_GPU}" "${LOG_PROBS_MAX_TOKENS_PER_GPU}"; do
   [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Batch, capacity and token budgets must be positive integers" >&2; exit 1; }
done
((GLOBAL_BATCH_SIZE % 8 == 0 && AGENT_SERVER_TRAIN_CONCURRENCY % 8 == 0)) || {
   echo "Mini SWE uses eight members per Group; GBS and physical slots must be divisible by eight" >&2; exit 1;
}
((8 % (TRAIN_TP * TRAIN_CP) == 0 && MAX_TOKENS_PER_GPU * TRAIN_CP >= MAX_CONTEXT_LEN && \
   LOG_PROBS_MAX_TOKENS_PER_GPU * TRAIN_CP >= MAX_CONTEXT_LEN)) || {
   echo "Training parallelism and both token budgets must fit the complete context" >&2; exit 1;
}
export ROLLOUT_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / 8))
SCHEDULING_ARGS=()
case "${AGENTIC_SCHEDULING}" in
   group)
      # Legacy whole-Group admission, with exactly the same physical upper bound.
      export AGENTIC_CONCURRENCY=$((AGENT_SERVER_TRAIN_CONCURRENCY / 8))
      unset AGENTIC_SESSION_CAPACITY_PER_NODE
      ;;
   atomic|incremental)
      export AGENTIC_CONCURRENCY="${ROLLOUT_BATCH_SIZE}"
      export AGENTIC_SESSION_CAPACITY_PER_NODE="${AGENT_SERVER_TRAIN_CONCURRENCY}"
      SCHEDULING_ARGS+=(--agentic-session-capacity-per-node "${AGENTIC_SESSION_CAPACITY_PER_NODE}")
      if [[ "${AGENTIC_SCHEDULING}" == incremental ]]; then
         SCHEDULING_ARGS+=(--agentic-incremental-session-start --agentic-fixed-step-eager)
      fi
      ;;
   *) echo "AGENTIC_SCHEDULING must be group, atomic or incremental" >&2; exit 1 ;;
esac

exec bash "${SCRIPT_DIR}/run_mini_swe_agent_hicache.sh" \
   --sglang-config "${SCRIPT_DIR}/hicache-cpu-4x-tp2-blackwell.yaml" \
   --tensor-model-parallel-size "${TRAIN_TP}" \
   --context-parallel-size "${TRAIN_CP}" \
   --linear-cp-mode "${LINEAR_CP_MODE:-headwise}" \
   --seq-length "${MAX_CONTEXT_LEN}" \
   --sglang-context-length "${MAX_CONTEXT_LEN}" \
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
   --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}" \
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}" \
   --balance-data --rollout-seed 42 \
   "${SCHEDULING_ARGS[@]}" "$@"
