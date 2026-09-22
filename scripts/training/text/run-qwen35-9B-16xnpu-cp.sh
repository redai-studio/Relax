#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 16xNPU colocate (sync) training script for DAPO math dataset.
#

set -ex
set -o pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"
export ASCEND_COREDUMP_SIGNAL=none
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HCCL_HOST_SOCKET_PORT_RANGE=63000-63050
export HCCL_NPU_SOCKET_PORT_RANGE=64000-64050
export TMS_HOOK_MODE="preload"
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
export MINDSPEED_BRIDGE_GDN_USE_TORCH_CONV=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"

NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/models/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/models/Qwen3.5-9B
   --megatron-to-hf-mode bridge
#    --warm-hf-checkpoint-page-cache

#    --load ${EXP_DIR}/Qwen3-9B_mcore_8xgpu/
   --save ${EXP_DIR}/Qwen3-9B_mcore_8xgpu/
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
   --balance-data
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${EXP_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --calculate-per-token-loss
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity selective
   # --recompute-method uniform
   # --recompute-num-layers 1

   --use-distributed-optimizer
   --use-dynamic-batch-size

   --data-pad-size-multiplier 4096
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 8192
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
   # icepop: drop tokens with ratio outside [tis-clip-low, tis-clip] instead of clamping (vanilla TIS).
   --custom-tis-function-path relax.backends.megatron.loss.icepop_function
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

WANDB_ARGS=(
   --use-tensorboard
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen35-9B-8x-${now}
)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.8
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
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 16], "rollout": [1, 16]}' \
   --num-gpus-per-node 16 \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-9B-GRPO-npu16-${now}.log
