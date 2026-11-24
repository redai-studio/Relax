#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5-4B text-pair embedding SFT on two data-parallel GPUs.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root containing Qwen3.5-4B}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing pairs.jsonl}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray Jobs HTTP address}"

export RELAX_SKIP_CLUSTER_CLEANUP=1
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
source "${RELAX_ROOT}/scripts/models/qwen35-4B.sh"

EXP_NAME="${EXP_NAME:-qwen35-4b-embedding-2xgpu}"
now=$(date '+%Y%m%d-%H%M%S')
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3.5-4B"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval 2
    --max-actor-ckpt-to-keep 2
)
SFT_ARGS=(
    --loss-type sft
    --task-type embedding
    --custom-dataset-class examples.embedding.text_pairs.TextPairDataset
    --prompt-data "${DATA_DIR}/pairs.jsonl"
    --global-batch-size 4
    --n-samples-per-prompt 1
    --num-rollout "${NUM_UPDATES:-2}"
    --num-steps-per-rollout 1
    --seq-length 256
    --sft-oversize-strategy keep
    --per-rank-fetch
)
LORA_ARGS=(
    --lora-rank 8
    --lora-alpha 16
    --lora-target-modules linear_proj linear_fc2
    --lora-scope language
    --freeze-vision-model
    --freeze-vision-projection
)
TRAIN_ARGS=(
    --bf16
    --optimizer adam
    --lr 1e-4
    --lr-decay-style constant
    --clip-grad 1.0
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --qkv-format thd
    --use-gloo-process-groups
    --use-dynamic-batch-size
    --max-tokens-per-gpu 1024
    --no-rope-fusion
    --attention-backend flash
    --no-use-tensorboard
)

mkdir -p "${SAVE_DIR}/logs/${EXP_NAME}"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft": [1, 0], "actor": [1, 2]}' \
    --num-data-storage-units 2 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "$@" 2>&1 | tee "${SAVE_DIR}/logs/${EXP_NAME}/${now}.log"
