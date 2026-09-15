#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B MTP SFT on OpenMathReasoning-mini, 8xklx, ray-submit launch.
#
# Usage:
#   bash scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-8xklx.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"
export WORKDIR="${WORKDIR:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
export PROJECT_NAME=qwen3.5-35B-sft
export PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
LOAD_DIR="${LOAD_DIR:-${MODEL_DIR}/Qwen3.5-35B-A3B/}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/qwen3.5-35B-A3B-mtp-sft}"
EXP_NAME="${EXP_NAME:-qwen3.5-35b-a3b-mtp-sft-8xklx}"

export MEGATRON=${WORKDIR}/Megatron-LM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XMLIR_USE_HYDRA_LINEAR=${XMLIR_USE_HYDRA_LINEAR:-1}
export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-1}
export XMLIR_MATMUL_FAST_MODE=${XMLIR_MATMUL_FAST_MODE:-1}
export XMLIR_MEMCPY_RETRY_SYNC=${XMLIR_MEMCPY_RETRY_SYNC:-true}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"eth0"}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"eth0"}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-"eth0"}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7"}

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen35-35B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --load ${LOAD_DIR}
   --save ${SAVE_DIR}/${EXP_NAME}
   --save-interval ${SAVE_INTERVAL:-100}
   --max-actor-ckpt-to-keep 1
   --num-epoch ${NUM_EPOCH:-1}
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key ${INPUT_KEY:-problem}
   --label-key ${LABEL_KEY:-generated_solution}
   --global-batch-size ${GLOBAL_BATCH_SIZE:-128}
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-18432}
   --balance-data
   # --cross-entropy-fusion-impl te
   # --cross-entropy-loss-fusion
)

MTP_ARGS=(
   --mtp-num-layers ${MTP_NUM_LAYERS:-1}
   --enable-mtp-training
   --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE:-4}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE:-1}
   --context-parallel-size ${CP_SIZE:-1}
   --expert-model-parallel-size ${EP_SIZE:-4}
   --expert-tensor-parallel-size ${ETP_SIZE:-1}

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --moe-grouped-gemm true
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-5}
   --lr-decay-style cosine
   --min-lr ${MIN_LR:-1e-6}
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --no-rope-fusion
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
)

WANDB_ARGS=(
   --tb-experiment-name qwen3.5-35B-klx-${now}
   --use-wandb
   --wandb-project ${PROJECT_NAME}
   --wandb-group qwen3.5-35B-SFT-8xklx
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
   --no-use-metrics-service
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
EXTRA_ENV_VARS_JSON="\"XTE_RECOMPUTE_LN_OUT_TOTAL\": \"${XTE_RECOMPUTE_LN_OUT_TOTAL}\",
    \"XMLIR_D_XPU_L3_SIZE\": \"${XMLIR_D_XPU_L3_SIZE}\""
source "${SCRIPT_DIR}/../../entrypoint/runtime-env-klx.sh"

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP:-127.0.0.1}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --selective-offload \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3.5-35b-a3b-mtp-sft-8xp800-${now}.log
