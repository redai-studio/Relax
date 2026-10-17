#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.6-35B-A3B DPO comparison on one 8-GPU node with fixed TP1/EP4.
# Baseline: PARALLEL_CASE=baseline EXP_NAME=unique-baseline-name bash scripts/entrypoint/ray-job.sh scripts/training/dpo/run-qwen36-35B-A3B-dpo-8xgpu.sh
# CP-only: PARALLEL_CASE=cp EXP_NAME=unique-cp-name bash scripts/entrypoint/ray-job.sh scripts/training/dpo/run-qwen36-35B-A3B-dpo-8xgpu.sh
# PP-only: PARALLEL_CASE=pp EXP_NAME=unique-pp-name bash scripts/entrypoint/ray-job.sh scripts/training/dpo/run-qwen36-35B-A3B-dpo-8xgpu.sh

set -ex
set -o pipefail

export RAY_NO_WAIT="${RAY_NO_WAIT-}"
now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "$(cd -- "${SCRIPT_DIR}/../../entrypoint" && pwd)/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen36-35B-A3B.sh"

PARALLEL_CASE="${PARALLEL_CASE:?Set PARALLEL_CASE to baseline, cp, or pp}"
case "${PARALLEL_CASE}" in
    baseline)
        PP_SIZE=1
        CP_SIZE=1
        ;;
    cp)
        PP_SIZE=1
        CP_SIZE=2
        ;;
    pp)
        PP_SIZE=2
        CP_SIZE=1
        ;;
    *)
        echo "PARALLEL_CASE must be baseline, cp, or pp; got ${PARALLEL_CASE}" >&2
        exit 1
        ;;
esac
ACTOR_WORLD_SIZE=8
EP_SIZE=4
DP_SIZE=$((ACTOR_WORLD_SIZE / (PP_SIZE * CP_SIZE)))
PROJECT_NAME="${PROJECT_NAME:-Relax/dpo/qwen3.6-35B-A3B}"
: "${EXP_NAME:?Set a fresh EXP_NAME for every comparison run}"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3.6-35B-A3B}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/ultrafeedback/ultrafeedback_train.parquet}"
CHECKPOINT_DIR="${SAVE_DIR}/${EXP_NAME}"
if [ -e "${CHECKPOINT_DIR}" ]; then
    echo "Fresh precision runs must not reuse an existing checkpoint directory: ${CHECKPOINT_DIR}" >&2
    exit 1
fi
NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
PACKED_TOKEN_CAPACITY="${PACKED_TOKEN_CAPACITY:-4096}"
if (( PACKED_TOKEN_CAPACITY % CP_SIZE != 0 )); then
    echo "PACKED_TOKEN_CAPACITY must be divisible by CP_SIZE" >&2
    exit 1
fi
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((PACKED_TOKEN_CAPACITY / CP_SIZE))}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --enable-weights-backuper
   --load "${CHECKPOINT_DIR}"
   --save "${CHECKPOINT_DIR}"
   --save-interval "${SAVE_INTERVAL:-100}"
   --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-2}"
)

DPO_ARGS=(
   --loss-type dpo
   --dpo-beta "${DPO_BETA:-0.1}"
   --no-dataset-shuffle
   --seed "${SEED:-42}"
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --preference-chosen-key chosen
   --preference-rejected-key rejected
   --preference-max-length "${SEQ_LENGTH}"
   --preference-max-completion-length "${SEQ_LENGTH}"
   --seq-length "${SEQ_LENGTH}"
   --num-rollout "${NUM_ROLLOUT}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-16}"
   --use-dynamic-batch-size
   # Keep the full-sequence packing capacity constant across CP configurations.
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --use-gloo-process-groups
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-6}"
   --lr-decay-style cosine
   --lr-warmup-fraction "${WARMUP_FRACTION:-0.05}"
   --min-lr 0
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.999
   --clip-grad 1.0
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EP_SIZE}"
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --no-rope-fusion
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
   --colocate
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP:-127.0.0.1}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"sft\":[1,0],\"actor\":[1,${ACTOR_WORLD_SIZE}]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${DPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
