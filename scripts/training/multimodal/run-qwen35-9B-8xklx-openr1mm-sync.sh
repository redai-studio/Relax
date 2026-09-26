#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xklx sync training script (ray-job mode).
# The Ray cluster is managed externally — do NOT kill ray or start a new cluster.
#
# Usage:
#   bash scripts/training/multimodal/run-qwen35-9B-8xklx-openr1mm-sync.sh [sync]

set -ex
set -o pipefail


now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"
export WORKDIR="${WORKDIR:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"
export PROJECT_NAME="${PROJECT_NAME:=Qwen3.5-9B-multimodal-openr1mm}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RL_MODEL_TYPE="Qwen35_9B_multimodal"
export XMLIR_USE_HYDRA_LINEAR=${XMLIR_USE_HYDRA_LINEAR:-1}
export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-1}
export XMLIR_MATMUL_FAST_MODE=${XMLIR_MATMUL_FAST_MODE:-1}
export XMLIR_MEMCPY_RETRY_SYNC=${XMLIR_MEMCPY_RETRY_SYNC:-true}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"eth0"}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"eth0"}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-"eth0"}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"}

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

NUM_GPUS_TOTAL="${NUM_GPUS_TOTAL:-8}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/Qwen3.5-9B
   --load ${EXP_DIR}/Qwen3.5-9B_mcore_8xgpu/
   --save ${EXP_DIR}/Qwen3.5-9B_mcore_8xgpu/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)

PROMPT_SET=${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"


ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   # --rollout-shuffle
   --rm-type openr1mm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 64
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-max-prompt-len 2048
   --rollout-temperature 0.8
   --global-batch-size 512
   --multimodal-keys '{"image":"image"}'
   --system-prompt "${SYSTEM_PROMPT}"
   --use-streaming-dataset
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1

   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --calculate-per-token-loss
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192

   --no-rope-fusion
)

GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project ${PROJECT_NAME}
   --wandb-group qwen35-9b-GRPO-klx-image-${now}
   --disable-wandb-random-suffix
   --no-use-metrics-service
   --no-use-tensorboard
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.75
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-mm-enable-dp-encoder

   --sglang-disable-custom-all-reduce
   --sglang-page-size 64
   --sglang-attention-backend kunlun
   --sglang-mm-attention-backend fa3
   --sglang-disable-radix-cache
   --sglang-max-running-requests 256
   --sglang-reasoning-parser qwen3
   --sglang-tool-call-parser qwen3_coder
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

export XSGL_FUSE_SPLIT_NORM_ROPE_NEOX=${XSGL_FUSE_SPLIT_NORM_ROPE_NEOX:-0}
export XMLIR_D_XPU_L3_SIZE=${XMLIR_D_XPU_L3_SIZE:-0}
export DEBUG_DUMP_TOKENS=${DEBUG_DUMP_TOKENS:-0}
export XMLIR_ENABLE_H2D_SSE_COPY=${XMLIR_ENABLE_H2D_SSE_COPY:-1}
export XTE_RECOMPUTE_LN_OUT_TOTAL=${XTE_RECOMPUTE_LN_OUT_TOTAL:-1}
EXTRA_ENV_VARS_JSON="\"XSGL_FUSE_SPLIT_NORM_ROPE_NEOX\": \"${XSGL_FUSE_SPLIT_NORM_ROPE_NEOX}\",
    \"XMLIR_D_XPU_L3_SIZE\": \"${XMLIR_D_XPU_L3_SIZE}\",
    \"DEBUG_DUMP_TOKENS\": \"${DEBUG_DUMP_TOKENS}\",
    \"XMLIR_ENABLE_H2D_SSE_COPY\": \"${XMLIR_ENABLE_H2D_SSE_COPY}\",
    \"XTE_RECOMPUTE_LN_OUT_TOTAL\": \"${XTE_RECOMPUTE_LN_OUT_TOTAL}\""
source "${SCRIPT_DIR}/../../entrypoint/runtime-env-klx.sh"

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}'\
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   --balance-data \
   --selective-offload \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-9b-GRPO-gpu8-colocate-${now}.log
