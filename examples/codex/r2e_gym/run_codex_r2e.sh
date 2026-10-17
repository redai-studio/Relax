#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

export NCCL_NVLS_ENABLE=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

CODEX_COMPACT="${CODEX_COMPACT:-0}"
CODEX_CONTEXT_WINDOW="${CODEX_CONTEXT_WINDOW:-32768}"
CODEX_COMPACT_LIMIT="${CODEX_COMPACT_LIMIT:-16384}"
EXTRA_ARGS=()
if [ -n "${CODEX_EVAL_DATA:-}" ]; then
   EXTRA_ARGS+=(
      --eval-prompt-data swe_bench_lite "${CODEX_EVAL_DATA}"
      --eval-interval "${EVAL_INTERVAL:-100}"
      --eval-input-key input
      --n-samples-per-eval-prompt 1
      --eval-temperature 0
      --agentic-eval-concurrency "${AGENTIC_EVAL_CONCURRENCY:-50}"
   )
fi
if [ "${CODEX_COMPACT}" = "1" ]; then
   EXTRA_ARGS+=(--agentic-custom-advantage-path examples.codex.r2e_gym.advantage.session_grpo)
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
   set +u
   source "${RELAX_DIR}/scripts/entrypoint/local.sh"
   set -u
   set +x
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

TIMESTAMP="$(date '+%Y-%m-%d-%H:%M:%S')"
PROJECT_NAME="${PROJECT_NAME:-Relax/dev/wulumeng/codex}"
EXP_NAME="${EXP_NAME:-qwen35-35B-A3B-codex-r2e-${TIMESTAMP}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
mkdir -p "${EXP_ROOT}/dump"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor":[1,8],"rollout":[1,8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "${MODEL_DIR}/Qwen3.5-35B-A3B" \
   --ref-load "${MODEL_DIR}/Qwen3.5-35B-A3B" \
   --megatron-to-hf-mode bridge \
   --prompt-data "${CODEX_R2E_DATA}" \
   --input-key input \
   --metadata-key metadata \
   --rollout-shuffle \
   --dump-details "${EXP_ROOT}/dump" \
   --num-rollout "${NUM_ROLLOUT}" \
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
   --partial-rollout \
   --over-sampling-batch-size 12 \
   --partial-rollout-max-aborted-count 1 \
   --rollout-max-response-len 4096 \
   --rollout-max-context-len "${CODEX_CONTEXT_WINDOW}" \
   --rollout-temperature 1 \
   --global-batch-size "${GLOBAL_BATCH_SIZE}" \
   --use-fault-tolerance \
   --optimizer adam \
   --lr 3e-7 \
   --lr-decay-style constant \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   --optimizer-cpu-offload \
   --overlap-cpu-optimizer-d2h-h2d \
   --use-precision-aware-optimizer \
   --advantage-estimator grpo \
   --use-kl-loss \
   --kl-loss-coef 0.001 \
   --kl-loss-type low_var_kl \
   --entropy-coef 0.0 \
   --eps-clip 0.2 \
   --eps-clip-high 0.28 \
   --use-tis \
   --rollout-num-gpus-per-engine 2 \
   --sglang-mem-fraction-static 0.75 \
   --sglang-max-running-requests 64 \
   --sglang-load-format dummy \
   --use-agentic-rollout \
   --agent-command "${CODEX_AGENT_PYTHON} ${SCRIPT_DIR}/agent.py" \
   --agent-cwd "${SCRIPT_DIR}" \
   --agent-timeout 1200 \
   --agent-env \
      "R2E_SIF_DIR=${R2E_SIF_DIR}" \
      "R2E_WHEEL_DIR=${R2E_DATA_PATH}/wheels" \
      "CODEX_RUNTIME_DIR=${CODEX_RUNTIME_DIR}" \
      "CODEX_COMPACT=${CODEX_COMPACT}" \
      "CODEX_CONTEXT_WINDOW=${CODEX_CONTEXT_WINDOW}" \
      "CODEX_COMPACT_LIMIT=${CODEX_COMPACT_LIMIT}" \
   --agentic-tool-call-parser qwen3_coder \
   --agentic-reasoning-parser qwen3 \
   --tensor-model-parallel-size 2 \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 4 \
   --expert-model-parallel-size 8 \
   --expert-tensor-parallel-size 1 \
   --recompute-granularity full \
   --recompute-method uniform \
   --recompute-num-layers 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 8192 \
   --data-pad-size-multiplier 1024 \
   --log-probs-max-tokens-per-gpu 24576 \
   --log-probs-chunk-size 2048 \
   --recompute-loss-function \
   --calculate-per-token-loss \
   --moe-router-load-balancing-type none \
   --moe-aux-loss-coeff 0.0 \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   --no-rope-fusion \
   --moe-flex-dispatcher-backend deepep \
   --moe-token-dispatcher-type flex \
   --use-clearml \
   --use-metrics-service \
   --tb-project-name "${PROJECT_NAME}" \
   --tb-experiment-name "${EXP_NAME}" \
   "${EXTRA_ARGS[@]}"
