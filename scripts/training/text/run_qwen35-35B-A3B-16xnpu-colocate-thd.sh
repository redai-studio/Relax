#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B 16xNPU colocate training script.
#
# Usage:
#   bash scripts/training/text/run_qwen35-35B-A3B-16xnpu-colocate-thd.sh

set -ex
set -o pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

ulimit -n 65535

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
export HCCL_HOST_SOCKET_PORT_RANGE=63000-63150
export HCCL_NPU_SOCKET_PORT_RANGE=64000-64150
export TMS_HOOK_MODE="preload"
export HYDRA_FULL_ERROR=1


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
NUM_ROLLOUT="${NUM_ROLLOUT:=3000}"

CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3.5-35B-A3B
   --ref-load ${EXP_DIR}/Qwen3.5-35B-A3B
   --megatron-to-hf-mode bridge
   # --load ${EXP_DIR}/Qwen3.5-35B-A3B-save-0821
   --save ${EXP_DIR}/Qwen3.5-35B-A3B-save-0821
   --save-interval 100
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${EXP_DIR}/dapo-math-17k/dapo-math-17k.jsonl

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
)

EVAL_ARGS=(
   --log-passrate
   --eval-interval 20
   --skip-eval-before-train
   --eval-prompt-data aime ${EXP_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   #--eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   # Packing is not supported for GDN currently
   --max-tokens-per-gpu 10240
   --no-rope-fusion
   --no-gradient-accumulation-fusion
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
   --use-distributed-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.6
   --sglang-max-running-requests 256
   --sglang-cuda-graph-bs 4 8 16 32 64 128 192 256
   --sglang-device npu
   --sglang-disable-radix-cache
   --sglang-chunked-prefill-size 8192
   --sglang-max-prefill-tokens 8192
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-attention-backend ascend
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

mkdir -p log
   ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${MASTER_ADDR}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 16], "rollout": [1, 16]}'\
   --max-staleness 0 \
   --colocate \
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
   "${MISC_ARGS[@]}" 2>&1 | tee log/qwen35-35B-MATH-npu16-colocate-${now}.log
