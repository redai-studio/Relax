#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B 16xXPU (2-node) fully sync training script for DAPO math dataset.
#
# Usage:
# node1:
# export IP_ADDRESS1=<IP_ADDRESS1>
# export MASTER_ADDR=${IP_ADDRESS1}
# export POD_NAME=${IP_ADDRESS1}
# export HOST_IP=${IP_ADDRESS1}
# bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/text/run-qwen35-35B-A3B-16xklx.sh
# node2:
# export IP_ADDRESS2=<IP_ADDRESS2>
# export MASTER_ADDR=${IP_ADDRESS1}
# export POD_NAME=${IP_ADDRESS2}
# export HOST_IP=${IP_ADDRESS2}
# bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/text/run-qwen35-35B-A3B-16xklx.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

export WORKDIR="${WORKDIR:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
export PROJECT_NAME=Relax-Qwen3.5-35B-P800
export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RL_MODEL_TYPE="Qwen35_35B_text"
export XMLIR_USE_HYDRA_LINEAR=${XMLIR_USE_HYDRA_LINEAR:-1}
export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-1}
export XMLIR_MEMCPY_RETRY_SYNC=${XMLIR_MEMCPY_RETRY_SYNC:-true}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7}

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1000}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --megatron-to-hf-mode bridge

   # --load ${EXP_DIR}/Qwen3.5-35B-A3B_mcore_16xgpu/
   --save ${EXP_DIR}/Qwen3.5-35B-A3B_mcore_16xgpu/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128
   --use-fault-tolerance
   --balance-data
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --moe-token-dispatcher-type flex
   --moe-grouped-gemm true
   # --moe-permute-fusion true
   # --optimizer-offload-fraction 0.5
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   # NOTE(wuhuan): to avoid algorithm performance degradation
   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.7
   --sglang-disable-custom-all-reduce
   --sglang-page-size 64
   --sglang-attention-backend kunlun
   --sglang-disable-radix-cache
   --sglang-max-running-requests 256
   # --sglang-disable-cuda-graph
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --sglang-router-policy round_robin
)

WANDB_ARGS=(
   # --use-clearml
   # --use-metrics-service
   # --tb-project-name  ${PROJECT_NAME}
   # --tb-experiment-name relax-qwen35-35B-A3B-p800x8-sync-${now}
   --tb-experiment-name qwen3.5-35B-p800x16-${now}
   --use-wandb
   --wandb-project ${PROJECT_NAME}
   --wandb-group qwen3.5-35B-p800x16-${now}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
   --no-use-metrics-service
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

source "${SCRIPT_DIR}/../../entrypoint/runtime-env-klx.sh"

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 16], "rollout": [1, 16]}' \
   --colocate \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --use-health-check \
   --selective-offload \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-35B-A3B-GRPO-xpu16-sync-${now}.log
