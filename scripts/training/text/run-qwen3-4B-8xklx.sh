#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU colocate training script.
#
# Usage:
#   bash scripts/training/text/run-qwen3-4B-8xklx.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

export WORKDIR="${WORKDIR:-/workspace}"
export MODEL_DIR="${MODEL_DIR:-/workspace}"
export DATA_DIR="${DATA_DIR:-/workspace}"
export PROJECT_NAME=Relax-Qwen3-4B-P800
export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"
export MEGATRON=${WORKDIR}/Megatron-LM

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RL_MODEL_TYPE="Qwen3_4B_text"
export FORCE_NN_LINEAR=${FORCE_NN_LINEAR:-1}
export XMLIR_MATMUL_FAST_MODE=${XMLIR_MATMUL_FAST_MODE:-1}
export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-1}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"eth0"}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"}


unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen3-4B.sh"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"


CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B
   --ref-load ${MODEL_DIR}/Qwen3-4B
   --megatron-to-hf-mode bridge
   --load ${EXP_DIR}/Qwen3-4B_mcore_8xgpu/
   --save ${EXP_DIR}/Qwen3-4B_mcore_8xgpu/
   --save-interval 100
)

PROMPT_SET="${PROMPT_SET:-/workdir/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-/workdir/aime-2024/aime-2024.jsonl}"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
   --use-fault-tolerance
)

EVAL_ARGS=(
   --skip-eval-before-train
   --log-passrate
   --eval-interval 20
   --eval-prompt-data aime ${EVAL_DATA}
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss
   #--micro-batch-size 16 # avoid OOM
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
   --sglang-disable-custom-all-reduce
   --sglang-page-size 64
   --sglang-attention-backend kunlun
   --sglang-disable-radix-cache
   --sglang-max-running-requests 32
   --sglang-disable-cuda-graph
)

WANDB_ARGS=(
   --use-wandb
   --tb-experiment-name relax-qwen3-4B-p800-${now}
   --wandb-project ${PROJECT_NAME}
   --wandb-group relax-qwen3-4B-p800-${now}
   --wandb-key ${WANDB_API_KEY}
   # --wandb-mode ${WANDB_MODE:-offline}
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

export XSGL_INT8_LM_HEAD=${XSGL_INT8_LM_HEAD:-0}
EXTRA_ENV_VARS_JSON="\"XSGL_INT8_LM_HEAD\": \"${XSGL_INT8_LM_HEAD}\""
source "${SCRIPT_DIR}/../../entrypoint/runtime-env-klx.sh"

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}'\
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
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
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-4b-GRPO-gpu8-${now}.log
