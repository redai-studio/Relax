#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xP800 colocate (sync) training script for DAPO math dataset.
#
# Usage:
#   bash scripts/training/sft/run-qwen35-9B-8xklx.sh

set -ex
set -o pipefail

gpus_num=8
export WORLD_SIZE=$(( gpus_num / 8 ))

now=$(date "+%Y-%m-%d-%H:%M:%S")

export WORKDIR="${WORKDIR:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
export EXP_DIR="${EXP_DIR:-/workspace}"
export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"
export WEB_PROXY="${WEB_PROXY:-}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XMLIR_USE_HYDRA_LINEAR=${XMLIR_USE_HYDRA_LINEAR:-1}
export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-1}
export XMLIR_MATMUL_FAST_MODE=${XMLIR_MATMUL_FAST_MODE:-1}
export XMLIR_MEMCPY_RETRY_SYNC=${XMLIR_MEMCPY_RETRY_SYNC:-true}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"eth0"}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"eth0"}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-"eth0"}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"eth1,eth2,eth3,eth4"}

exp_target="${EXP_TARGET:-}"

PROMPT_DATA="${PROMPT_DATA:=${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
PROJECT_NAME="${PROJECT_NAME:="XHS_Relax"}"
EXP_NAME="Qwen3.5-9B-MM-SFT-${gpus_num}xP800${exp_target}"

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen35-9B.sh"

if [ -z "${SYSTEM_PROMPT:-}" ] && [ -n "${SYSTEM_PROMPT_FILE:-}" ]; then
    SYSTEM_PROMPT="$(cat "${SYSTEM_PROMPT_FILE}")"
fi
SYSTEM_PROMPT="${SYSTEM_PROMPT:?SYSTEM_PROMPT is required — set it via env var or SYSTEM_PROMPT_FILE}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/Qwen3.5-9B
   --megatron-to-hf-mode bridge
   --load ${EXP_DIR}/sft/checkpoint/Qwen3-9B_mcore_${gpus_num}xklx/
   --save ${EXP_DIR}/sft/checkpoint/Qwen3-9B_mcore_${gpus_num}xklx/
   --save-interval 500
   --max-actor-ckpt-to-keep 1
   --num-epoch 1
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data ${PROMPT_DATA}
   --input-key ${INPUT_KEY:-problem}
   --label-key ${LABEL_KEY:-generated_solution}
   --global-batch-size 512
   --use-dynamic-batch-size
   # --max-tokens-per-gpu 20480
   --max-tokens-per-gpu 10240
   --balance-data
   --system-prompt "${SYSTEM_PROMPT}"
   --sft-prefetch-buffer-size 512
   --sft-prefetch-num-workers 8
   --use-distributed-optimizer
   --overlap-grad-reduce
   --overlap-param-gather
   --cross-entropy-fusion-impl te
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --calculate-per-token-loss

   # --recompute-granularity full
   # --recompute-method block
   # --recompute-num-layers 8

   --no-rope-fusion
   --colocate
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-05
   --lr-decay-style cosine
   --min-lr 1e-06
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer
)

EVAL_ARGS=(
   --eval-size 0.01
   --eval-interval 200
)

PREDICT_ARGS=()

WANDB_ARGS=(
   --tb-experiment-name ${EXP_NAME}-${now}
   --no-use-wandb
   --wandb-project ${PROJECT_NAME}
   --wandb-group ${EXP_NAME}-${now}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
   --no-use-metrics-service
   --no-use-tensorboard
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

export XTE_RECOMPUTE_LN_OUT_TOTAL=${XTE_RECOMPUTE_LN_OUT_TOTAL:-1}
export XMLIR_D_XPU_L3_SIZE=${XMLIR_D_XPU_L3_SIZE:-0}
export DEBUG_DUMP_TOKENS=${DEBUG_DUMP_TOKENS:-0}
EXTRA_ENV_VARS_JSON="\"XTE_RECOMPUTE_LN_OUT_TOTAL\": \"${XTE_RECOMPUTE_LN_OUT_TOTAL}\",
    \"XMLIR_D_XPU_L3_SIZE\": \"${XMLIR_D_XPU_L3_SIZE}\",
    \"DEBUG_DUMP_TOKENS\": \"${DEBUG_DUMP_TOKENS}\""
source "${SCRIPT_DIR}/../../entrypoint/runtime-env-klx.sh"

mkdir -p log
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, '"${gpus_num}"']}' \
   --colocate \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --selective-offload \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PREDICT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee log/qwen35-9B-mm-sft-${gpus_num}xklx-${now}.log
