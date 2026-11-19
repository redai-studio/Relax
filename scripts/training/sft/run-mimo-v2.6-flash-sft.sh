#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Four-layer text SFT on a converted MiMo-V2.6-Flash checkpoint.
# Default layout: one 8-GPU node, TP4, PP1 and EP4.
# Usage: bash run-mimo-v2.6-flash-sft.sh BF16_CHECKPOINT MESSAGES_JSONL SAVE_DIR [train args...]
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HF_CHECKPOINT=${1:?provide the four-layer BF16 checkpoint directory}
PROMPT_DATA=${2:?provide a messages dataset}
SAVE_DIR=${3:?provide the checkpoint output directory}
shift 3

source "${MODEL_CONFIG_DIR:-${SCRIPT_DIR}/../../models}/mimo-v2.6-flash.sh"
if [[ "$layers" != 4 ]]; then
    echo "Expected a four-layer checkpoint, found $layers layers in $HF_CHECKPOINT/config.json" >&2
    exit 1
fi

NODES=${NODES:-1}
NUM_GPUS=${GPUS_PER_NODE:-${NUM_GPUS:-8}}
TP=${TP:-4}
PP=${PP:-1}
EP=${EP:-4}

CKPT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT"
    --ref-load "$HF_CHECKPOINT"
    --megatron-to-hf-mode bridge
    --save "$SAVE_DIR"
    --save-interval "${SAVE_INTERVAL:-100}"
)
SFT_ARGS=(
    --loss-type sft
    --prompt-data "$PROMPT_DATA"
    --input-key messages
    --num-rollout "${STEPS:-1}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-2}"
    --qkv-format thd
    --use-dynamic-batch-size
    --seq-length "${SEQ_LENGTH:-1024}"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-1024}"
    --sft-max-in-flight-steps 1
    --num-data-storage-units 1
)
PERF_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --expert-model-parallel-size "$EP"
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --attention-backend fused
    --attention-softmax-in-fp32
)
if (( TP > 1 )); then
    PERF_ARGS+=(--sequence-parallel)
fi
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
    --optimizer-cpu-offload
    --optimizer-offload-fraction 1.0
    --use-precision-aware-optimizer
    --accumulate-allreduce-grads-in-fp32
)
TRAIN_CMD=(
    "${PYTHON:-python3}" -m relax.entrypoints.train
    --resource "{\"sft\":[1,0],\"actor\":[$NODES,$NUM_GPUS]}"
    --num-gpus-per-node "$NUM_GPUS"
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${PERF_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "$@"
)

# A script used as the entrypoint of an existing Ray Job is already on the driver.
if [[ -n "${RAY_JOB_CONFIG_JSON_ENV_VAR:-}" ]]; then
    exec "${TRAIN_CMD[@]}"
fi
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" -- "${TRAIN_CMD[@]}"
