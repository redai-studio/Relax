#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# QSA + GDN + PLE, experimental native-prefix text-only SFT. DATA_DIR/train.jsonl contains a messages column.
# Usage: bash scripts/entrypoint/ray-job.sh examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh
# Preview without Ray or GPUs: DRY_RUN=1 bash examples/chunkflow/scripts/run-qwen38-flash-next-sft-32xgpu.sh

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"
source "${SCRIPT_DIR}/_common.sh"

# Prefix replay calls the shipped native QSA bitset path. Optimized recipes
# patch these methods with different layout contracts and must stay disabled.
export MCORE_QSA_SPARSE_BACKEND=flex
export RELAX_QWEN38_QSA_COMMUNITY_BACKEND=""
export RELAX_QWEN38_QSA_FUSED_SELECTOR=0
export RELAX_QWEN38_QSA_FUSED_ATTENTION=0
export RELAX_QWEN38_QSA_FIXED_POOL=0
export RELAX_QWEN38_QSA_TP_SELECTOR=0
export RELAX_QWEN38_QSA_CPU_LAYOUT=0
export RELAX_QWEN38_QSA_LOCAL_QUERY=0
export RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=0
export RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=0
export RELAX_QWEN38_RETAIN_QSA_ATTENTION=0
export RELAX_QWEN38_PLE_STREAMED=0
export RELAX_QWEN38_PLE_FUSED_CONV=0
# Preserve the checkpoint's large PLE tables on host memory, frozen for SFT.
export QWEN38_PLE_CPU_OFFLOAD=1

: "${MODEL_DIR:?Set MODEL_DIR to the directory containing Qwen3.8-Flash-Next}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing train.jsonl}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/chunkflow}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-text-sft-32xgpu}"
now=$(date "+%Y%m%d-%H%M%S")

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3.8-Flash-Next"
    --megatron-to-hf-mode bridge
    --freeze-params-name-list '(^|\.)per_layer_embedding\.|self_attention\.indexer\.'
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-10}"
    --max-actor-ckpt-to-keep 2
)

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}/train.jsonl"
    --input-key messages
    --num-rollout "${NUM_STEPS:-100}"
    --global-batch-size 32
    --micro-batch-size 1
    --seq-length "${SEQ_LENGTH:-8192}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${SEQ_LENGTH:-8192}"
    --sft-oversize-strategy skip
)

CHUNKFLOW_ARGS=(
    --use-chunkflow
    --chunkflow-chunk-size "${CHUNK_SIZE:-2048}"
    --chunkflow-pack-policy staggered
    --chunkflow-keep-last-k "${KEEP_LAST_K:-1}"
    --custom-config-path examples/chunkflow/configs/qwen38.yaml
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --moe-token-dispatcher-type alltoall
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --qkv-format thd
    --bf16
    --use-distributed-optimizer
    --calculate-per-token-loss
    --no-rope-fusion
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
)

TRACKING_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

submit_training \
    --resource '{"sft": [1, 0], "actor": [1, 32]}' \
    --colocate \
    --num-data-storage-units 1 \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${CHUNKFLOW_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${TRACKING_ARGS[@]}"
