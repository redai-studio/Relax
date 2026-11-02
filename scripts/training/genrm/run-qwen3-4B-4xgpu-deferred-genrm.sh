#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Two-step Qwen3-4B shared deferred genrm smoke run on one four-GPU node.
# Set MODEL_DIR, DATA_DIR, SAVE_DIR and RAY_API_SERVER_ADDRESS (Jobs HTTP).
# On an existing cluster:
#   bash scripts/entrypoint/ray-job.sh scripts/training/genrm/run-qwen3-4B-4xgpu-deferred-genrm.sh

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_NAME="${PROJECT_NAME:-Relax/inference/deferred}"
EXP_NAME="${EXP_NAME:-qwen3-4b-dapomath-4xgpu-deferred-genrm}"
now=$(date "+%Y%m%d-%H%M%S")
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${RAY_API_SERVER_ADDRESS:?Set RAY_API_SERVER_ADDRESS to the existing Ray Jobs HTTP address}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/ray-job.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen3-4B.sh"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-4B"
    --ref-load "${MODEL_DIR}/Qwen3-4B"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval 1
    --max-actor-ckpt-to-keep 2
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rm-type dapo-genrm
    --reward-key score
    --num-rollout "${NUM_ROLLOUT:-2}"
    --rollout-batch-size 2
    --over-sampling-batch-size 2
    --n-samples-per-prompt 2
    --rollout-max-response-len 1024
    --rollout-temperature 0.8
    --global-batch-size 4
)

SCORER_ARGS=(
    --defer-reward-to-post-process
    --genrm-model-path "${MODEL_DIR}/Qwen3-4B"
    --genrm-num-gpus-per-engine 1
    --genrm-engine-config '{"mem_fraction_static": 0.4}'
    --genrm-sampling-config '{"temperature": 0.1, "max_response_len": 256}'
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
    --max-tokens-per-gpu 4096
)

TRAIN_ARGS=(
    --advantage-estimator grpo
    --eps-clip 0.2
    --eps-clip-high 0.28
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

SGLANG_ARGS=(
    --rollout-num-gpus 4
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.4
)

WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

# Each role uses the same four bundles in a separate phase.
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_API_SERVER_ADDRESS}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 4], "rollout": [1, 4], "genrm": [1, 4]}' \
    --num-gpus-per-node 4 \
    --actor-num-gpus-per-node 4 \
    --colocate --offload \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${SCORER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "$@" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
