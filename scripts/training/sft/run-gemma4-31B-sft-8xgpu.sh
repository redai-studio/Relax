#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# gemma-4-31B-it full-parameter SFT, TP=8 PP=1, ray-submit launch.
#
# Usage:
#   MODEL_DIR=<dir with gemma-4-31B-it> \
#   PROMPT_DATA=<sft.jsonl> bash scripts/training/sft/run-gemma4-31B-sft-8xgpu.sh
#
#   # multi-node (QS sets MASTER_ADDR/POD_NAME/WORLD_SIZE):
#   ACTOR_GPUS=16 bash scripts/entrypoint/spmd-multinode.sh \
#     scripts/training/sft/run-gemma4-31B-sft-8xgpu.sh

set -ex
set -o pipefail

unset NCCL_NVLS_ENABLE

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

TP_SIZE="${TP_SIZE:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-8}"

if [ "$((16 % TP_SIZE))" -ne 0 ]; then
    echo "ERROR: TP_SIZE=${TP_SIZE} does not divide the 16 KV heads." >&2
    exit 1
fi
if [ "$((ACTOR_GPUS % TP_SIZE))" -ne 0 ]; then
    echo "ERROR: ACTOR_GPUS=${ACTOR_GPUS} is not divisible by TP_SIZE=${TP_SIZE}." >&2
    exit 1
fi
echo "parallelism: TP=${TP_SIZE} DP=$((ACTOR_GPUS / TP_SIZE)) PP=1 CP=1 on ${ACTOR_GPUS} GPU(s)"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" \
    | jq -c '.env_vars.GEMMA4_CONVERSION_MODE = "text"')

GEMMA4_SFT_THINKING="${GEMMA4_SFT_THINKING:-1}"
if [ -n "${GEMMA4_SFT_THINKING}" ]; then
    RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" \
        | jq -c --arg v "${GEMMA4_SFT_THINKING}" '.env_vars.GEMMA4_SFT_THINKING = $v')
fi

if [ -n "${NVTE_DEBUG:-}" ]; then
    RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" | jq -c \
        --arg d "${NVTE_DEBUG}" --arg l "${NVTE_DEBUG_LEVEL:-2}" \
        '.env_vars.NVTE_DEBUG = $d | .env_vars.NVTE_DEBUG_LEVEL = $l')
fi
export RUNTIME_ENV_JSON

source "${MODEL_CONFIG_DIR}/gemma4-31B.sh"

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the dir containing gemma-4-31B-it}"
CKPT="${CKPT:-${MODEL_DIR}/gemma-4-31B-it}"
PROMPT_DATA="${PROMPT_DATA:?set PROMPT_DATA to an SFT jsonl}"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/gemma4}"
EXP_NAME="${EXP_NAME:-gemma4-31b-sft-smoke-gpu8}"
CKPT_ROOT="${CKPT_ROOT:-/data/temp}"
SAVE_DIR="${SAVE_DIR:-${CKPT_ROOT}/gemma4-31B-sft}"

RAY_ADDRESS="${RAY_ADDRESS:-http://${MASTER_ADDR:-127.0.0.1}:${RAY_DASHBOARD_PORT:-8265}}"

CKPT_ARGS=(
   --hf-checkpoint "${CKPT}"
   --ref-load      "${CKPT}"
   --load          "${CKPT}"
   --save          "${SAVE_DIR}"

   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --save-interval 1000000
   --num-epoch 1
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key   "${INPUT_KEY:-instruction}"
   --label-key   "${LABEL_KEY:-output}"

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-4096}
   --balance-data

   --global-batch-size ${GLOBAL_BATCH_SIZE:-256}
   --seq-length ${SEQ_LENGTH:-4096}
   ${NUM_ROLLOUT:+--num-rollout ${NUM_ROLLOUT}}
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --sequence-parallel

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --data-parallel-sharding-strategy optim
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --min-lr 5e-7
   --lr-decay-style ${LR_DECAY_STYLE:-cosine}
   --lr-warmup-fraction 0.1

   --weight-decay ${WEIGHT_DECAY:-0.1}
   --adam-beta1 0.9
   --adam-beta2 ${ADAM_BETA2:-0.95}

   --adam-eps 1e-8
   --clip-grad 1.0
)

PER_TOKEN_LOSS_FLAG=()
[ "${CALC_PER_TOKEN_LOSS:-1}" != "0" ] && PER_TOKEN_LOSS_FLAG=(--calculate-per-token-loss)

MISC_ARGS=(
   --bf16
   --attention-backend auto
   --cross-entropy-fusion-impl te
   --log-interval 1
   --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES:-30}
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --recompute-loss-function
   --use-health-check
   --no-save-rng

   --seed ${SEED:-42}

   "${PER_TOKEN_LOSS_FLAG[@]}"
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}-${now}
)

mkdir -p "${SAVE_DIR}" log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"sft\": [1, 0], \"actor\": [1, ${ACTOR_GPUS}]}" \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${WANDB_ARGS[@]}" 2>&1 | tee log/${EXP_NAME}-${now}.log
