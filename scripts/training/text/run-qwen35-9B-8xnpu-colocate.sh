#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen35-9B 8xNPU colocate training script.
#
# Usage:
#   bash scripts/training/text/run-qwen35-9B-8xnpu-colocate.sh

set -ex
set -o pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

ulimit -n 65535
export MINDSPEED_BRIDGE_GDN_USE_TORCH_CONV=1
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-enp23s0f3}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp23s0f3}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-enp23s0f3}"
export HCCL_CONNECT_TIMEOUT=1200
export RAY_DEDUP_LOGS=0
export PYTHONBUFFERED=1

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"
export ASCEND_COREDUMP_SIGNAL=none
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_HOST_SOCKET_PORT_RANGE=63000-63050
export HCCL_NPU_SOCKET_PORT_RANGE=64000-64050
export TMS_HOOK_MODE="preload"
export HYDRA_FULL_ERROR=1

# optimize
export CPU_AFFINITY_CONF=1
export TORCH_HCCL_ZERO_COPY=1
export MULTI_STREAM_MEMORY_REUSE=1
export HCCL_OP_EXPANSION_MODE="AIV"

export SGLANG_NPU_USE_MULTI_STREAM=1

export SGLANG_NPU_ASYNC_EXPONENTIAL=1
export SGLANG_GMM2_TRITON=0
export SGLANG_MOE_FRONT_FUSION=1
export SGLANG_NPU_FULL_ATTN_FUSION=1

export ASCEND_USE_FIA=1
export GDN_ATTN_BACKEND_TRITON=0
export STREAMS_PER_DEVICE=32
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES=30
export TASK_QUEUE_ENABLE=1

export SGLANG_NPU_GDN_UPDATE_FUSED=1

export SGLANG_NPU_TP_ASCENDC_FUSION=1
export SGLANG_NPU_GDN_RECURRENT_ASCENDC=1

export SGLANG_NPU_EXP_RACE_TRITON=1
export SGLANG_NPU_GDN_QKVZBA_PACK=1
export SGLANG_NPU_GDN_QKVZBA_PACK_MAX_M=256
export SGLANG_NPU_MOE_PREFETCH=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"
EXP_DIR="${EXP_DIR:-/root/exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
NUM_ROLLOUT="${NUM_ROLLOUT:=3000}"

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B/
    --ref-load ${MODEL_DIR}/Qwen3.5-9B/
    --megatron-to-hf-mode bridge
    --load ${EXP_DIR}/Qwen3.5-9B_mcore_8xnpu/
    --save ${EXP_DIR}/Qwen3.5-9B_mcore_8xnpu/
    --save-interval 50
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
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 8192
    --rollout-temperature 1
    --global-batch-size 256
    --use-fault-tolerance
)

EVAL_ARGS=(
    --log-passrate
    --eval-interval 20000
    --skip-eval-before-train
    --eval-prompt-data aime ${EXP_DIR}/aime-2024/aime-2024.jsonl
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 8192
    #--eval-top-p 0.7
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 2
    --use-dynamic-batch-size
    # Packing is not supported for GDN currently
    --qkv-format thd
    # --micro-batch-size 1
    --max-tokens-per-gpu 10240
    --log-probs-max-tokens-per-gpu 40960
    --no-rope-fusion
    --no-gradient-accumulation-fusion
    --balance-data
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
    # --optimizer-cpu-offload
    # --overlap-cpu-optimizer-d2h-h2d
    # --use-precision-aware-optimizer
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.85
   --sglang-max-running-requests 272
   --sglang-cuda-graph-bs 2 8 16 24 32 64 128 192 256
   --sglang-device npu
   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-chunked-prefill-size 16384
   --sglang-max-prefill-tokens 16384
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-attention-backend ascend
   --sglang-pp-size 1
   --sglang-dp-size 1
   --sglang-ep-size 1
   --sglang-max-mamba-cache-size 640
   --sglang-router-policy round_robin
   --sglang-mamba-ssm-dtype bfloat16
   --sglang-tokenizer-backend fastokens
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
    --use-flash-attn
)
# --debug-rollout-only \
mkdir -p log
    ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${MASTER_ADDR}:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 16], "rollout": [1, 16]}'\
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --nnodes 1 \
    --num-gpus-per-node 16 \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee log/qwen35-9B-GRPO-npu8-colocate-${now}.log
