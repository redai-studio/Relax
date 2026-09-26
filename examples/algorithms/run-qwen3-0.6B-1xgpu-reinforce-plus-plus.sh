#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Parameterized Qwen3-0.6B synchronous colocate recipe for GRPO,
# REINFORCE++, and REINFORCE++-baseline.
#
# Required:
#   MODEL_PATH=/path/to/Qwen3-0.6B
#   PROMPT_DATA=/path/to/gsm8k/main/train_clean.parquet
#   OUTPUT_DIR=/path/to/output
#
# Optional:
#   EVAL_DATA=/path/to/gsm8k/main/test-00000-of-00001.parquet
#   ADVANTAGE_ESTIMATOR=reinforce_plus_plus[_baseline] | grpo
#   NUM_ROLLOUT=50 SEED=42 USE_HEALTH_CHECK=1

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    # Cluster launchers should set RELAX_ENTRYPOINT_MODE and initialize their
    # own job-isolated Ray runtime instead of sourcing the local entrypoint.
    source "${REPO_ROOT}/scripts/entrypoint/local.sh"
fi

MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${REPO_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

: "${MODEL_PATH:?MODEL_PATH must point to Qwen3-0.6B}"
: "${PROMPT_DATA:?PROMPT_DATA must point to the GSM8K training parquet}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-reinforce_plus_plus}"
NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
SEED="${SEED:-42}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-4}"
REWARD_MAX_CONCURRENCY="${REWARD_MAX_CONCURRENCY:-16}"
USE_HEALTH_CHECK="${USE_HEALTH_CHECK:-1}"
KL_COEF="${KL_COEF:-0.01}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
PROJECT_NAME="${PROJECT_NAME:-Relax/task29-reinforce-plus-plus}"
now="$(date '+%Y-%m-%d-%H-%M-%S')"
experiment_name="qwen3-0.6b-${ADVANTAGE_ESTIMATOR}-seed-${SEED}"

case "${ADVANTAGE_ESTIMATOR}" in
    reinforce_plus_plus)
        ALGORITHM_ARGS=(
            --advantage-estimator reinforce_plus_plus
            --normalize-advantages
            --gamma 1.0
            --kl-coef "${KL_COEF}"
            --kl-loss-type k1
            --eps-clip 0.2
            --eps-clip-high 0.2
        )
        ;;
    reinforce_plus_plus_baseline)
        ALGORITHM_ARGS=(
            --advantage-estimator reinforce_plus_plus_baseline
            --normalize-advantages
            --kl-coef 0
            --use-kl-loss
            --kl-loss-type k2
            --kl-loss-coef "${KL_LOSS_COEF}"
            --eps-clip 0.2
            --eps-clip-high 0.2
        )
        ;;
    grpo)
        ALGORITHM_ARGS=(
            --advantage-estimator grpo
            --kl-coef 0
            --use-kl-loss
            --kl-loss-type k2
            --kl-loss-coef "${KL_LOSS_COEF}"
            --eps-clip 0.2
            --eps-clip-high 0.2
        )
        ;;
    *)
        echo "Unsupported ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR}" >&2
        exit 2
        ;;
esac

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
    --load "${OUTPUT_DIR}/actor"
    --save "${OUTPUT_DIR}/actor"
    --save-interval "${SAVE_INTERVAL:-50}"
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --reward-num-workers "${REWARD_NUM_WORKERS}"
    --reward-max-concurrency "${REWARD_MAX_CONCURRENCY}"
    --balance-data
    --use-fault-tolerance
)

EVAL_ARGS=(--skip-eval-before-train)
if [[ -n "${EVAL_DATA:-}" ]]; then
    EVAL_ARGS+=(
        --log-passrate
        --eval-interval "${EVAL_INTERVAL:-10}"
        --eval-prompt-data gsm8k "${EVAL_DATA}"
        --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-4}"
        --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    )
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
    --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU:-4096}"
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.45}"
)

LOGGING_ARGS=(
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${experiment_name}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --seed "${SEED}"
)

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"
if [[ -z "${RUNTIME_ENV_JSON:-}" ]]; then
    RUNTIME_ENV_JSON='{}'
fi

CONTROLLER_ARGS=(
    --resource '{"actor": [1, 1], "rollout": [1, 1]}'
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
)
case "${USE_HEALTH_CHECK}" in
    1|true) CONTROLLER_ARGS+=(--use-health-check) ;;
    0|false) ;;
    *) echo "USE_HEALTH_CHECK must be one of: 1, 0, true, false" >&2; exit 2 ;;
esac

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${CONTROLLER_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${ALGORITHM_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/${experiment_name}-${now}.log"
