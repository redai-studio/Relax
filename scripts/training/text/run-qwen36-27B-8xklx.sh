#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.6-27B 8xP800 colocate (sync) training script for DAPO math dataset.
#
# Usage:
#   bash scripts/training/text/run-qwen36-27B-8xklx.sh

set -ex
set -o pipefail

gpus_num=8
export WORLD_SIZE=$(( gpus_num / 8 ))

now=$(date "+%Y-%m-%d-%H:%M:%S")

WORKDIR="${WORKDIR:-/workspace}"
MODEL_DIR="${MODEL_DIR:-/workspace}"
DATA_DIR="${DATA_DIR:-/workspace}"
EXP_DIR="${EXP_DIR:-${MODEL_DIR}}"
WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"
WEB_PROXY="${WEB_PROXY:-}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RL_MODEL_TYPE="Qwen36_27B_text"
export XMLIR_USE_HYDRA_LINEAR=${XMLIR_USE_HYDRA_LINEAR:-1}
export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-1}
export XMLIR_MATMUL_FAST_MODE=${XMLIR_MATMUL_FAST_MODE:-1}
export XMLIR_MEMCPY_RETRY_SYNC=${XMLIR_MEMCPY_RETRY_SYNC:-true}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"eth0"}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"eth0"}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-"eth0"}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"eth1,eth2,eth3,eth4"}

PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-${DATA_DIR}/aime-2024/aime-2024.jsonl}"
PROJECT_NAME="${PROJECT_NAME:="XHS_Relax"}"
EXP_NAME="${EXP_NAME:="Qwen3.6-27B-Text-${gpus_num}xklx"}"

unset http_proxy
unset https_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen36-27B.sh"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.6-27B
   --ref-load ${MODEL_DIR}/Qwen3.6-27B
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --load ${EXP_DIR}/text/Qwen3.6-27B_mcore_${gpus_num}xklx/
   --save ${EXP_DIR}/text/Qwen3.6-27B_mcore_${gpus_num}xklx/
   --save-interval 50
   --max-actor-ckpt-to-keep 1
)

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key ${INPUT_KEY:-prompt}
   --label-key ${LABEL_KEY:-label}
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 10240
   --rollout-max-prompt-len 2048
   --rollout-max-context-len 12288
   --rollout-temperature 0.8
   --global-batch-size 256
   --balance-data
   --use-fault-tolerance
)

# EVAL_ARGS=(
#    --log-passrate
#    --skip-eval-before-train
#    --eval-interval 20
#    --eval-prompt-data aime ${EVAL_DATA}
#    --n-samples-per-eval-prompt 8
#    --eval-max-response-len 10240
#    --eval-top-p 0.7
# )

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2


   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # --recompute-granularity full
   # --recompute-method block
   # --recompute-num-layers 16

   --calculate-per-token-loss
   --use-dynamic-batch-size
   --max-tokens-per-gpu 12288
   --no-rope-fusion
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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)

   --sglang-router-policy round_robin

   --sglang-disable-custom-all-reduce
   --sglang-page-size 64
   --sglang-attention-backend kunlun
   --sglang-mm-attention-backend fa3
   --sglang-disable-radix-cache
   --sglang-max-running-requests 128
   # --sglang-disable-cuda-graph
   --sglang-reasoning-parser qwen3
   --sglang-tool-call-parser qwen3_coder
)

WANDB_ARGS=(
   --tb-experiment-name ${EXP_NAME}-${now}
   --use-wandb
   --wandb-project ${PROJECT_NAME}
   --wandb-group ${EXP_NAME}-${now}
   --wandb-key ${WANDB_API_KEY}
   --disable-wandb-random-suffix
   --no-use-metrics-service
   --no-use-tensorboard
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

export XTE_RECOMPUTE_LN_OUT_TOTAL=${XTE_RECOMPUTE_LN_OUT_TOTAL:-1}
export XMLIR_D_XPU_L3_SIZE=${XMLIR_D_XPU_L3_SIZE:-0}
export DEBUG_DUMP_TOKENS=${DEBUG_DUMP_TOKENS:-0}
export RELAX_LOG_SEQ_LENS=${RELAX_LOG_SEQ_LENS:-1}
EXTRA_ENV_VARS_JSON="\"XTE_RECOMPUTE_LN_OUT_TOTAL\": \"${XTE_RECOMPUTE_LN_OUT_TOTAL}\",
    \"XMLIR_D_XPU_L3_SIZE\": \"${XMLIR_D_XPU_L3_SIZE}\",
    \"DEBUG_DUMP_TOKENS\": \"${DEBUG_DUMP_TOKENS}\",
    \"RELAX_LOG_SEQ_LENS\": \"${RELAX_LOG_SEQ_LENS}\""
source "${SCRIPT_DIR}/../../entrypoint/runtime-env-klx.sh"

RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}"
mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, '"${gpus_num}"'], "rollout": [1, '"${gpus_num}"']}' \
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
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen36-27B-GRPO-${gpus_num}xklx-${now}.log
