#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -ex
set -o pipefail

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen25-7B.sh"

SEARCH_R1_MODE="${SEARCH_R1_MODE:?Set SEARCH_R1_MODE to vanilla or multiagent.}"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the model root.}"
SEARCH_R1_DATA_ROOT="${SEARCH_R1_DATA_ROOT:?Set SEARCH_R1_DATA_ROOT to the prepared Search-R1 asset root.}"
SEARCH_R1_RETRIEVER_URL="${SEARCH_R1_RETRIEVER_URL:?Set SEARCH_R1_RETRIEVER_URL to the POST /retrieve endpoint.}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
PROJECT_NAME="${PROJECT_NAME:-Relax/dev/wulumeng/search-r1}"
EXP_NAME="${EXP_NAME:-qwen25-7b-base-search-r1-${SEARCH_R1_MODE}-${TIMESTAMP}}"

TRAIN_DATA="${SEARCH_R1_DATA_ROOT}/qa/nq_hotpotqa_train_relax/train.parquet"
EVAL_DATA="${SEARCH_R1_DATA_ROOT}/qa/nq_hotpotqa_train_relax/test.parquet"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen2.5-7B"
    --ref-load "${MODEL_DIR}/Qwen2.5-7B"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --metadata-key extra_info
    --use-agentic-rollout
    --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
    --agent-cwd "${SCRIPT_DIR}"
    --agent-env
        "SEARCH_R1_MODE=${SEARCH_R1_MODE}"
        "SEARCH_R1_RETRIEVER_URL=${SEARCH_R1_RETRIEVER_URL}"
    --agent-timeout 9999
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-prompt-len 2048
    --rollout-max-response-len 1000
    --rollout-stop-token-ids 151645
    --rollout-max-context-len 16384
    --rollout-temperature 1.0
    --rollout-top-p 0.95
    --eval-max-response-len 500
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-shuffle
    --balance-data
    --use-fault-tolerance
    --agentic-custom-advantage-path examples.search_r1.advantage_search_r1.advantage_func
)

EVAL_ARGS=()
if [ "${EVAL_INTERVAL}" -gt 0 ]; then
    EVAL_ARGS+=(
        --eval-interval "${EVAL_INTERVAL}"
        --eval-prompt-data
            search_r1 "${EVAL_DATA}"
        --custom-eval-rollout-log-function-path examples.search_r1.eval_search_r1.log_eval_rollout_data
        --n-samples-per-eval-prompt 1
        --eval-temperature 0.0
        --eval-top-p 1.0
    )
fi

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.05
    --kl-loss-type low_var_kl
    --entropy-coef 0.001
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-7
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --calculate-per-token-loss
    --max-tokens-per-gpu 16384
    --log-probs-max-tokens-per-gpu 16384
    --no-rope-fusion
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.65
)

LOG_ARGS=(
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}"
    --use-clearml
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

if [ -n "${DUMP_DETAILS_DIR:-}" ]; then
    MISC_ARGS+=(--dump-details "${DUMP_DETAILS_DIR}")
fi

mkdir -p logs
ray job submit --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "logs/${EXP_NAME}.log"
