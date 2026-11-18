#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-4B fully async GRPO smoke test on 4 logical NPUs.
# Set MODEL_DIR and DATA_DIR, then run this script with bash.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${MODEL_DIR:?Set MODEL_DIR to the model root}/Qwen3.5-4B"
PROMPT_SET="${DATA_DIR:?Set DATA_DIR to the dataset root}/dapo-math-17k/dapo-math-17k.jsonl"
for input in "${CHECKPOINT}/config.json" "${PROMPT_SET}"; do
   if [[ ! -r "$input" ]]; then
      printf 'Missing or unreadable CI input: %s\n' "$input" >&2
      exit 1
   fi
done

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export HCCL_NPU_SOCKET_PORT_RANGE="62000-62050"
export HCCL_HOST_SOCKET_PORT_RANGE="62100-62200"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
   source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-4B.sh"

export RAY_NO_WAIT="${RAY_NO_WAIT-}"
PROJECT_NAME="${PROJECT_NAME:-Relax/dev/dapo-math}"
EXP_NAME="${EXP_NAME:-qwen35-4b-grpo-4xnpu-async}"
now=$(date "+%Y%m%d-%H%M%S")

CKPT_ARGS=(
   --hf-checkpoint "$CHECKPOINT"
   --ref-load "$CHECKPOINT"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_SET"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout "${NUM_ROLLOUT:-4}"
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 0.8
   --global-batch-size 8
   --use-fault-tolerance
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
   --sglang-device npu
   --sglang-attention-backend ascend
   --sglang-disable-radix-cache
   --sglang-chunked-prefill-size 8192
   --sglang-max-prefill-tokens 8192
   --sglang-cuda-graph-bs 4 8 16 32 64 96 128
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
)

TRAIN_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # Match the Qwen3.5 Ascend recipes: GDN uses unpacked batches.
   --qkv-format bshd
   --micro-batch-size 1
   --no-gradient-accumulation-fusion
   --use-flash-attn
   --distributed-timeout-minutes 60
   --max-tokens-per-gpu 9216
   --no-rope-fusion
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
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

METRICS_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "$PROJECT_NAME"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

# CI exercises rollout and weight sync without writing training checkpoints.
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 1], "rollout": [1, 1], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}' \
   --max-staleness 1 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 1 \
   --ref-actor-config '{"tensor_model_parallel_size": 1, "max_tokens_per_gpu": 9216, "sequence_parallel": false, "only_load_weight": true}' \
   --fully-async \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${TRAIN_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${METRICS_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
