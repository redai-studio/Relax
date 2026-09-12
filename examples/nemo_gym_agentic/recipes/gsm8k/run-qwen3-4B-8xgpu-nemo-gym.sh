#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU GSM8K NeMo Gym agentic training.

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

NEMO_GYM_SOURCE_DATA="${NEMO_GYM_SOURCE_DATA:-/data/nemo-gym/gsm8k_benchmark.jsonl}"
NEMO_GYM_DATA_LIMIT="${NEMO_GYM_DATA_LIMIT:-32}"
NEMO_GYM_NUM_ROLLOUT="${NEMO_GYM_NUM_ROLLOUT:-3}"
NEMO_GYM_N_SAMPLES_PER_PROMPT="${NEMO_GYM_N_SAMPLES_PER_PROMPT:-4}"
NEMO_GYM_GLOBAL_BATCH_SIZE="${NEMO_GYM_GLOBAL_BATCH_SIZE:-${NEMO_GYM_N_SAMPLES_PER_PROMPT}}"
NEMO_GYM_ROLLOUT_MAX_RESPONSE_LEN="${NEMO_GYM_ROLLOUT_MAX_RESPONSE_LEN:-2048}"
NEMO_GYM_ROLLOUT_MAX_CONTEXT_LEN="${NEMO_GYM_ROLLOUT_MAX_CONTEXT_LEN:-8192}"
NEMO_GYM_ROLLOUT_MAX_PROMPT_LEN="${NEMO_GYM_ROLLOUT_MAX_PROMPT_LEN:-$((NEMO_GYM_ROLLOUT_MAX_CONTEXT_LEN - 1))}"
NEMO_GYM_TOOL_CALL_PARSER="${NEMO_GYM_TOOL_CALL_PARSER:-qwen}"
NEMO_GYM_DEADLINE_S="${NEMO_GYM_DEADLINE_S:-600}"
NEMO_GYM_LEASE_S="${NEMO_GYM_LEASE_S:-60}"
NEMO_GYM_AGENT_TIMEOUT="${NEMO_GYM_AGENT_TIMEOUT:-$((NEMO_GYM_DEADLINE_S + 60))}"
NEMO_GYM_MAX_TOKENS_PER_GPU="${NEMO_GYM_MAX_TOKENS_PER_GPU:-8192}"
NEMO_GYM_LOG_PROBS_CHUNK_SIZE="${NEMO_GYM_LOG_PROBS_CHUNK_SIZE:-1024}"
NEMO_GYM_NUM_GPUS="${NEMO_GYM_NUM_GPUS:-8}"
NEMO_GYM_TENSOR_MODEL_PARALLEL_SIZE="${NEMO_GYM_TENSOR_MODEL_PARALLEL_SIZE:-${NEMO_GYM_NUM_GPUS}}"
NEMO_GYM_ROLLOUT_NUM_GPUS_PER_ENGINE="${NEMO_GYM_ROLLOUT_NUM_GPUS_PER_ENGINE:-${NEMO_GYM_NUM_GPUS}}"
NEMO_GYM_GATEWAY_PORT="${NEMO_GYM_GATEWAY_PORT:-${GYM_PORT:-28100}}"
EXP_DIR="${EXP_DIR:-${PWD}/outputs/nemo-gym-gsm8k}"
test -s "${NEMO_GYM_SOURCE_DATA}"

RUNTIME_ENV_JSON="$(
   jq -c --arg relax_root "${RELAX_ROOT}" '
      .env_vars.PYTHONPATH = ($relax_root + ":" + (.env_vars.PYTHONPATH // ""))
      | . + {"py_executable": "/usr/bin/python3"}
   ' <<<"${RUNTIME_ENV_JSON}"
)"
export RUNTIME_ENV_JSON

PROJECT_NAME="${PROJECT_NAME:-Relax/dev/nemo-gym}"
EXP_NAME="${EXP_NAME:-gsm8k-qwen3-4b-${NEMO_GYM_NUM_GPUS}xgpu}"
GATEWAY_URL="http://${GYM_HOST}:${NEMO_GYM_GATEWAY_PORT}"
PROMPT_SET="${NEMO_GYM_PROMPT_DATA:-${EXP_DIR}/data/gsm8k_smoke.jsonl}"
SUBMISSION_ID="${RELAX_SUBMISSION_ID:-relax-nemo-gym-gsm8k-${NEMO_GYM_NUM_GPUS}xgpu-${now}-${BASHPID}-${RANDOM}}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
if [ -n "${RAY_DASHBOARD_ADDRESS:-}" ]; then
   DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS}"
elif [[ "${RAY_ADDRESS:-auto}" =~ ^([^:/]+):[0-9]+$ ]]; then
   DASHBOARD_ADDRESS="http://${BASH_REMATCH[1]}:${RAY_DASHBOARD_PORT}"
else
   DASHBOARD_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
fi
NEMO_GYM_RESOURCE_JSON="$(
   jq -cn --argjson num_gpus "${NEMO_GYM_NUM_GPUS}" \
      '{actor: [1, $num_gpus], rollout: [1, $num_gpus]}'
)"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B/"
   --ref-load "${MODEL_DIR}/Qwen3-4B/"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save "${EXP_DIR}/Qwen3-4B_mcore_${NEMO_GYM_NUM_GPUS}xgpu/"
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
     "NEMO_GYM_ENVIRONMENT=gsm8k"
     "NEMO_GYM_CONFIG=gsm8k-v1"
     "NEMO_GYM_MODEL=model"
     "NEMO_GYM_INTERRUPT_POLICY=protected"
     "NEMO_GYM_DEADLINE_S=${NEMO_GYM_DEADLINE_S}"
     "NEMO_GYM_LEASE_S=${NEMO_GYM_LEASE_S}"
   --agent-timeout "${NEMO_GYM_AGENT_TIMEOUT}"
   --agentic-reasoning-parser qwen3
   --agentic-tool-call-parser "${NEMO_GYM_TOOL_CALL_PARSER}"

   --num-rollout "${NEMO_GYM_NUM_ROLLOUT}"
   --rollout-batch-size 1
   --n-samples-per-prompt "${NEMO_GYM_N_SAMPLES_PER_PROMPT}"
   --rollout-max-prompt-len "${NEMO_GYM_ROLLOUT_MAX_PROMPT_LEN}"
   --rollout-max-context-len "${NEMO_GYM_ROLLOUT_MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${NEMO_GYM_ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature 0.7
   --global-batch-size "${NEMO_GYM_GLOBAL_BATCH_SIZE}"
)

PERF_ARGS=(
   --tensor-model-parallel-size "${NEMO_GYM_TENSOR_MODEL_PARALLEL_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${NEMO_GYM_MAX_TOKENS_PER_GPU}"
   --log-probs-max-tokens-per-gpu "${NEMO_GYM_MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${NEMO_GYM_LOG_PROBS_CHUNK_SIZE}"
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
   --rollout-num-gpus-per-engine "${NEMO_GYM_ROLLOUT_NUM_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-max-bs 4
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

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --submission-id="${SUBMISSION_ID}" \
   --address="${DASHBOARD_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- bash "${RUN_TRAINING_SCRIPT}" \
   "${GATEWAY_URL}" "${NEMO_GYM_SOURCE_DATA}" "${PROMPT_SET}" "${NEMO_GYM_DATA_LIMIT}" \
   --resource "${NEMO_GYM_RESOURCE_JSON}" \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${TRACKING_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen3-4B-${NEMO_GYM_NUM_GPUS}xgpu-nemo-gym-gsm8k-${now}.log"
