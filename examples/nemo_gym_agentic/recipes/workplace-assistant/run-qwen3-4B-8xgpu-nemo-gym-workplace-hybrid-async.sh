#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU Workplace Assistant hybrid-async training (4 actor + 4 rollout).

set -ex
set -o pipefail

now=$(date -u "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." &>/dev/null && pwd)"
RUN_TRAINING_SCRIPT="${EXAMPLE_DIR}/scripts/run_training.sh"
RUN_AGENT_SCRIPT="${EXAMPLE_DIR}/scripts/run_agent_app.sh"

test -f "${RUN_TRAINING_SCRIPT}" && test -f "${RUN_AGENT_SCRIPT}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${EXAMPLE_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

: "${MODEL_DIR:?MODEL_DIR must contain Qwen3-4B/}"
: "${GYM_HOST:?GYM_HOST must be reachable from every Relax worker}"
: "${RUNTIME_ENV_JSON:?RUNTIME_ENV_JSON must be set by a Relax entrypoint}"
test -s "${MODEL_DIR}/Qwen3-4B/config.json"

NEMO_GYM_SOURCE_DATA="${NEMO_GYM_SOURCE_DATA:-/data/nemo-gym/workplace_assistant_train.jsonl}"
NEMO_GYM_GATEWAY_PORT="${NEMO_GYM_GATEWAY_PORT:-${GYM_PORT:-29000}}"
SAVE_DIR="${SAVE_DIR:-${PWD}/outputs/nemo-gym-workplace-hybrid-async}"
test -s "${NEMO_GYM_SOURCE_DATA}"
DATA_LIMIT="$(awk 'NF { count++ } END { print count + 0 }' "${NEMO_GYM_SOURCE_DATA}")"

RUNTIME_ENV_JSON="$(
   jq -c --arg relax_root "${RELAX_ROOT}" '
      .env_vars.PYTHONPATH = ($relax_root + ":" + (.env_vars.PYTHONPATH // ""))
      | . + {"py_executable": "/usr/bin/python3"}
   ' <<<"${RUNTIME_ENV_JSON}"
)"
export RUNTIME_ENV_JSON

PROJECT_NAME="${PROJECT_NAME:-Relax/dev/nemo-gym}"
EXP_NAME="${EXP_NAME:-workplace-assistant-qwen3-4b-8xgpu-hybrid-async}"
GATEWAY_URL="http://${GYM_HOST}:${NEMO_GYM_GATEWAY_PORT}"
PROMPT_SET="${NEMO_GYM_PROMPT_DATA:-${SAVE_DIR}/data/workplace_assistant_hybrid_train.jsonl}"
SUBMISSION_ID="${RELAX_SUBMISSION_ID:-relax-nemo-gym-workplace-8xgpu-hybrid-async-${now}-${BASHPID}-${RANDOM}}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
if [ -n "${RAY_DASHBOARD_ADDRESS:-}" ]; then
   DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS}"
elif [[ "${RAY_ADDRESS:-auto}" =~ ^([^:/]+):[0-9]+$ ]]; then
   DASHBOARD_ADDRESS="http://${BASH_REMATCH[1]}:${RAY_DASHBOARD_PORT}"
else
   DASHBOARD_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B/"
   --ref-load "${MODEL_DIR}/Qwen3-4B/"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save "${SAVE_DIR}/nemo-gym/workplace/Qwen3-4B_mcore_8xgpu_hybrid_async/"
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key input
   --metadata-key metadata
   --rollout-shuffle

   --use-agentic-rollout
   --agent-command ". ${RUN_AGENT_SCRIPT}"
   --agent-cwd "${RELAX_ROOT}"
   --agent-env
     "NEMO_GYM_URL=${GATEWAY_URL}"
     "NEMO_GYM_ENVIRONMENT=workplace_assistant"
     "NEMO_GYM_CONFIG=workplace-assistant-v1"
     "NEMO_GYM_MODEL=model"
     "NEMO_GYM_INTERRUPT_POLICY=protected"
     "NEMO_GYM_DEADLINE_S=600"
     "NEMO_GYM_LEASE_S=60"
   --agent-timeout 660
   --agentic-reasoning-parser qwen3
   --agentic-tool-call-parser qwen

   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 8191
   --rollout-max-context-len 8192
   --rollout-max-response-len 2048
   --rollout-temperature 0.7
   --global-batch-size 64
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 8192
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-max-bs 4
   --sglang-router-policy consistent_hashing
   --sglang-router-disable-circuit-breaker
)

TRACKING_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --skip-eval-before-train
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

RAY_SUBMIT_ARGS=()
if [ "${RAY_NO_WAIT:-0}" = "1" ]; then
   RAY_SUBMIT_ARGS+=(--no-wait)
fi

mkdir -p log
ray job submit "${RAY_SUBMIT_ARGS[@]}" \
   --submission-id="${SUBMISSION_ID}" \
   --address="${DASHBOARD_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- bash "${RUN_TRAINING_SCRIPT}" \
   "${GATEWAY_URL}" "${NEMO_GYM_SOURCE_DATA}" "${PROMPT_SET}" "${DATA_LIMIT}" \
   --resource '{"actor":[1,4],"rollout":[1,4]}' \
   --max-staleness 2 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 1 \
   --hybrid \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${TRACKING_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen3-4B-8xgpu-nemo-gym-workplace-hybrid-async-${now}.log"
