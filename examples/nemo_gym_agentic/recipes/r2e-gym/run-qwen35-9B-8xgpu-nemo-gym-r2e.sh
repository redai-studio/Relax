#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xGPU R2E-Gym training: 32K context, actor TP=2 and CP=1,
# and four SGLang TP=2 rollout engines.

set -ex
set -o pipefail

PROJECT_NAME="${PROJECT_NAME:-Relax/dev/nemo-gym}"
EXP_NAME="${EXP_NAME:-r2e-gym-qwen35-9b-8xgpu-tp2-32k}"
: "${SAVE_DIR:=${PWD}/outputs/${EXP_NAME}}"

: "${NEMO_GYM_SOURCE_DATA:?Set NEMO_GYM_SOURCE_DATA to the prepared R2E-Gym JSONL}"
NEMO_GYM_GATEWAY_PORT="${NEMO_GYM_GATEWAY_PORT:-${GYM_PORT:-${R2E_GYM_PORT_BASE:-28100}}}"
if ! [[ "${NEMO_GYM_GATEWAY_PORT}" =~ ^[1-9][0-9]{0,4}$ ]] || ((NEMO_GYM_GATEWAY_PORT > 65535)); then
   echo "NEMO_GYM_GATEWAY_PORT must be an integer between 1 and 65535" >&2
   exit 2
fi

now=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." &>/dev/null && pwd)"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
   source "${EXAMPLE_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

: "${MODEL_DIR:?MODEL_DIR must contain Qwen3.5-9B/}"
: "${GYM_HOST:?GYM_HOST must be reachable from every Relax worker}"
: "${RUNTIME_ENV_JSON:?RUNTIME_ENV_JSON must be set by a Relax entrypoint}"
test -s "${MODEL_DIR}/Qwen3.5-9B/config.json"
test -s "${MODEL_DIR}/Qwen3.5-9B/preprocessor_config.json"
DATA_LIMIT="$(wc -l < "${NEMO_GYM_SOURCE_DATA}")"

RUNTIME_ENV_JSON="$(
   jq -c --arg relax_root "${RELAX_ROOT}" '
      .env_vars.PYTHONPATH = ($relax_root + ":" + (.env_vars.PYTHONPATH // ""))
      | . + {"py_executable": "/usr/bin/python3"}
   ' <<<"${RUNTIME_ENV_JSON}"
)"
export RUNTIME_ENV_JSON

PROMPT_SET="${NEMO_GYM_PROMPT_DATA:-${NEMO_GYM_SOURCE_DATA%.jsonl}_relax.jsonl}"
GATEWAY_URL="http://${GYM_HOST}:${NEMO_GYM_GATEWAY_PORT}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.5-9B/"
   --ref-load "${MODEL_DIR}/Qwen3.5-9B/"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save "${SAVE_DIR}/nemo-gym/r2e-gym/Qwen3.5-9B/"
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key input
   --metadata-key metadata
   --rollout-shuffle

   --use-agentic-rollout
   --agent-command ". ${RELAX_ROOT}/examples/nemo_gym_agentic/scripts/run_agent_app.sh"
   --agent-cwd "${RELAX_ROOT}"
   --agent-env
     "NEMO_GYM_URL=${GATEWAY_URL}"
     "NEMO_GYM_ENVIRONMENT=r2e_gym"
     "NEMO_GYM_CONFIG=r2e-gym-v1"
     "NEMO_GYM_MODEL=model"
     "NEMO_GYM_INTERRUPT_POLICY=protected"
     "NEMO_GYM_DEADLINE_S=1200"
     "NEMO_GYM_LEASE_S=120"
   --agent-timeout 1260
   --agentic-tool-call-parser qwen3_coder
   --num-rollout 32
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 8192
   --rollout-max-response-len 43008
   --rollout-max-context-len 51200
   --rollout-temperature 0.7
   --global-batch-size 64

   # --debug-rollout-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --sequence-parallel
   --calculate-per-token-loss
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --no-rope-fusion

   --use-dynamic-batch-size
   --max-tokens-per-gpu 12800
   --log-probs-chunk-size 512
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
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-max-bs 4
   --sglang-router-policy consistent_hashing
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
   --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- bash "${RELAX_ROOT}/examples/nemo_gym_agentic/scripts/run_training.sh" \
   "${GATEWAY_URL}" "${NEMO_GYM_SOURCE_DATA}" "${PROMPT_SET}" "${DATA_LIMIT}" \
   --resource '{"actor":[1,8],"rollout":[1,8]}' \
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
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen35-9B-8xgpu-tp2-32k-nemo-gym-${now}.log"
