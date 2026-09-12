#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU agentic training on a mix of math_with_judge and
# workplace_assistant. Per-row `metadata.environment` / `metadata.config`
# (attached by prepare_multienv.sh) selects the Gateway spec — the training
# script itself does not set NEMO_GYM_ENVIRONMENT.

set -ex
set -o pipefail

export NEMO_GYM_GATEWAY_PORT="${NEMO_GYM_GATEWAY_PORT:-${GYM_PORT:-${MULTIENV_PORT_BASE:-29300}}}"
PROJECT_NAME="${PROJECT_NAME:-Relax/dev/nemo-gym}"
EXP_NAME="${EXP_NAME:-multienv-math-workplace-qwen3-4b-8xgpu}"

: "${MODEL_DIR:?MODEL_DIR must contain Qwen3-4B/}"
: "${GYM_HOST:?GYM_HOST must be reachable from every Relax worker}"
: "${DATA_DIR:?DATA_DIR must be set (shared data root)}"
case "${DATA_DIR%/}" in
    */nemo-gym/multienv) ;;
    *) DATA_DIR="${DATA_DIR%/}/nemo-gym/multienv" ;;
esac
: "${NEMO_GYM_SOURCE_DATA:=${DATA_DIR}/multienv_train_raw.jsonl}"

now=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." &>/dev/null && pwd)"
RUN_TRAINING_SCRIPT="${EXAMPLE_DIR}/scripts/run_training.sh"
RUN_AGENT_SCRIPT="${EXAMPLE_DIR}/scripts/run_agent_app.sh"

for required_path in \
   "${MODEL_DIR}/Qwen3-4B/config.json" \
   "${NEMO_GYM_SOURCE_DATA}" \
   "${RUN_TRAINING_SCRIPT}" \
   "${RUN_AGENT_SCRIPT}"; do
   if [ ! -s "${required_path}" ]; then
      echo "ERROR: required file is missing or empty: ${required_path}" >&2
      exit 2
   fi
done
if ! [[ "${NEMO_GYM_GATEWAY_PORT}" =~ ^[0-9]+$ ]] \
   || ((10#${NEMO_GYM_GATEWAY_PORT} < 1 || 10#${NEMO_GYM_GATEWAY_PORT} > 65535)); then
   echo "ERROR: NEMO_GYM_GATEWAY_PORT must be an integer between 1 and 65535" >&2
   exit 2
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
   source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

: "${RUNTIME_ENV_JSON:?RUNTIME_ENV_JSON must be set by a Relax entrypoint}"
DATA_LIMIT="$(wc -l < "${NEMO_GYM_SOURCE_DATA}")"

RUNTIME_ENV_JSON="$(
   jq -c --arg relax_root "${RELAX_ROOT}" '
      .env_vars.PYTHONPATH = ($relax_root + ":" + (.env_vars.PYTHONPATH // ""))
      | . + {"py_executable": "/usr/bin/python3"}
   ' <<<"${RUNTIME_ENV_JSON}"
)"
export RUNTIME_ENV_JSON

PROMPT_SET="${DATA_DIR}/multienv_train.jsonl"
GATEWAY_URL="http://${GYM_HOST}:${NEMO_GYM_GATEWAY_PORT}"
SUBMISSION_ID="${RELAX_SUBMISSION_ID:-relax-nemo-gym-multienv-${now}-${BASHPID}-${RANDOM}}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
if [ -n "${RAY_DASHBOARD_ADDRESS:-}" ]; then
   DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS}"
elif [[ "${RAY_ADDRESS:-auto}" =~ ^([^:/]+):[0-9]+$ ]]; then
   DASHBOARD_ADDRESS="http://${BASH_REMATCH[1]}:${RAY_DASHBOARD_PORT}"
elif [ "${RAY_ADDRESS:-auto}" = "auto" ]; then
   ray_head_ip="$(
      ray list nodes --address=auto --format json |
         jq -er 'map(select(.state == "ALIVE" and .is_head_node == true)) | .[0].node_ip'
   )"
   DASHBOARD_ADDRESS="http://${ray_head_ip}:${RAY_DASHBOARD_PORT}"
else
   echo "ERROR: cannot derive the Ray Dashboard URL; set RAY_DASHBOARD_ADDRESS" >&2
   exit 2
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B/"
   --ref-load "${MODEL_DIR}/Qwen3-4B/"
   --save ${SAVE_DIR}/nemo-gym/multienv/Qwen3-4B
   --save-interval 200
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)

# Per-row metadata routes each session to the right environment. --agent-env
# purposely omits NEMO_GYM_ENVIRONMENT / NEMO_GYM_CONFIG — client.py reads
# metadata.environment / metadata.config first, and would only fall back to
# these variables for a single-env recipe.
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
     "NEMO_GYM_MODEL=model"
     "NEMO_GYM_INTERRUPT_POLICY=protected"
     "NEMO_GYM_DEADLINE_S=1800"
     "NEMO_GYM_LEASE_S=60"
   --agent-timeout 1860
   --agentic-reasoning-parser qwen3
   --agentic-tool-call-parser qwen

   --num-rollout 200
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 6144
   --rollout-max-response-len 16384
   --rollout-max-context-len 24576
   --rollout-temperature 1.0
   --global-batch-size 256
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
   --max-tokens-per-gpu 24576
   --log-probs-max-tokens-per-gpu 24576
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
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-router-policy consistent_hashing

   # --sglang-enable-session-radix-cache
   # --sglang-radix-eviction-policy priority
   # --agentic-session-lifecycle
   # --agentic-program-admission
   # --agentic-admission-headroom 0.90
   # --agentic-admission-pressure-threshold 0.92
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
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen3-4B-8xgpu-nemo-gym-multienv-${now}.log"
