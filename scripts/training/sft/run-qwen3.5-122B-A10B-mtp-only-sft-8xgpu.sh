#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-122B-A10B MTP-only SFT, 8xGPU single-node, ray-submit launch.
#
# Parallelism: TP=8, PP=1, CP=1, EP=8, ETP=1. Keeping PP at one lets the
# backbone and the single trainable MTP block shard their experts over all
# eight GPUs, while TP=8 also shards dense weights and leaves DP at one. This
# recipe targets 8x96 GB GPUs and starts conservatively at 12800 tokens/GPU
# with full activation recomputation. Allow roughly 300-400 GB of host RAM for
# the 244 GB HF checkpoint, Bridge loading peak, and CPU-offloaded MTP optimizer.
#
# Usage:
#   NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash scripts/training/sft/run-qwen3.5-122B-A10B-mtp-only-sft-8xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-122B-A10B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/mtp-only}"
EXP_NAME="${EXP_NAME:-qwen3.5-122B-A10B-mtp-only-sft-qwen-rollout-partial-219-gpu8}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
LOAD_DIR="${LOAD_DIR:-${MODEL_DIR}/Qwen3.5-122B-A10B/}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/qwen3.5-122B-A10B-mtp-only-sft}"
SAVE_HF_DIR="${SAVE_HF_DIR:-${EXP_DIR}/hf-checkpoints/${EXP_NAME}}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-qwen3.5-122B-A10B-partial-219.parquet}"
RAY_ADDRESS="${RAY_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-122B-A10B/
   --megatron-to-hf-mode bridge

   --load ${LOAD_DIR}
   --save ${SAVE_DIR}/${EXP_NAME}
   --save-interval ${SAVE_INTERVAL:-1000}
   --max-actor-ckpt-to-keep 1
   --num-epoch ${NUM_EPOCH:-10}
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key ${INPUT_KEY:-problem}
   --label-key ${LABEL_KEY:-generated_solution}
   --global-batch-size ${GLOBAL_BATCH_SIZE:-64}
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-12800}
   --sft-tq-timeout-minutes ${SFT_TQ_TIMEOUT_MINUTES:-60}
   --balance-data
   # --sft-oversize-strategy ${SFT_OVERSIZE_STRATEGY:-keep}
)

MTP_ARGS=(
   --mtp-only-training
   --mtp-num-layers 1
   --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --cross-entropy-loss-fusion
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-3e-5}
   --lr-decay-style cosine
   --min-lr ${MIN_LR:-1e-6}
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0

   # Only MTP parameters enter these CPU optimizer states.
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --no-rope-fusion
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

RUNTIME_ENV_JSON=$(python3 -c '
import json, os
d = json.loads(os.environ["RUNTIME_ENV_JSON"])
d.setdefault("env_vars", {}).update({
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
})
print(json.dumps(d))
')
export RUNTIME_ENV_JSON

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee log/qwen3.5-122B-A10B-mtp-only-sft-gpu8-${now}.log
