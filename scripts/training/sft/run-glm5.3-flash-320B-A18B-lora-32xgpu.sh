#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Text LoRA SFT with the GLM-5.3-Flash cu13 image and 32 GPUs.
# MODEL_DIR/GLM-5.3-Flash is the original HF checkpoint.
# DATA_DIR/train.jsonl contains OpenAI messages; heldout.jsonl is optional.
# From an existing Ray cluster:
#   MODEL_DIR=... DATA_DIR=... SAVE_DIR=... bash scripts/entrypoint/ray-job.sh \
#       scripts/training/sft/run-glm5.3-flash-320B-A18B-lora-32xgpu.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the shared model root}"
: "${DATA_DIR:?Set DATA_DIR to the shared dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to the shared checkpoint root}"
MODEL_NAME="${MODEL_NAME:-GLM-5.3-Flash}"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
ACTOR_GPUS="${ACTOR_GPUS:-32}"
TP_SIZE="${TP_SIZE:-4}"
EP_SIZE="${EP_SIZE:-16}"
RESOURCE="{\"actor\": [1, ${ACTOR_GPUS}], \"sft\": [1, 0]}"
TRAIN_FILE="${TRAIN_FILE:-train.jsonl}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/glm5.3-flash}"
EXP_NAME="${EXP_NAME:-glm5.3-flash-text-lora-32xgpu}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
now=$(date "+%Y%m%d-%H%M%S")

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
if [[ "${RELAX_ENTRYPOINT_MODE}" == "local" ]]; then
    RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://${MASTER_ADDR}:8265}"
fi
: "${RAY_JOB_ADDRESS:?Set RAY_JOB_ADDRESS to the Ray Jobs HTTP URL}"
source "${SCRIPT_DIR}/../../models/glm5.3-flash-320B-A18B.sh"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-10}"
    --max-actor-ckpt-to-keep 2
    --save-lora-only
)

LORA_ARGS=(
    --lora-rank 16
    --lora-alpha 32
    --lora-dropout 0
    --lora-scope language
    --lora-target-modules linear_fc1 linear_fc2
    --freeze-vision-model
    --freeze-vision-projection
)

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}/${TRAIN_FILE}"
    --input-key messages
    --num-rollout "${NUM_ROLLOUT:-1000}"
    --rollout-batch-size "${GLOBAL_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --seq-length "${SEQ_LENGTH}"
    --sft-oversize-strategy skip
    --sft-chunked-logits
    --sft-logits-chunk-size 512
    --recompute-loss-function
    --balance-data
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --linear-cp-mode headwise
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${SEQ_LENGTH}"
    --empty-unused-memory-level 1
    --calculate-per-token-loss
    --attention-backend flash
    --attention-dropout 0
    --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LEARNING_RATE:-2e-5}"
    --lr-decay-style constant
    --use-checkpoint-opt-param-scheduler
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --no-pin-cpu-grads
    --no-pin-cpu-params
)

EVAL_ARGS=()
if [[ -n "${EVAL_FILE:-}" ]]; then
    EVAL_ARGS=(
        --eval-prompt-data heldout "${DATA_DIR}/${EVAL_FILE}"
        --eval-input-key messages
        --eval-interval "${EVAL_INTERVAL:-10}"
    )
fi

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource "${RESOURCE}" \
    --max-staleness 0 \
    --num-data-storage-units 2 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${METRICS_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
