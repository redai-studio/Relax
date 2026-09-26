#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-397B-A17B MTP SFT (messages + tool_calls mode), 128xGPU (16-node), ray-submit launch.
#
# Parallelism: TP=2, PP=8 (decoder-first=5, decoder-last=1, middle 6 stages * 9 layers = 60),
# CP=8, EP=16, ETP=1. TP*PP*CP = 128 matches 128 GPUs.
#
# Context: 128K. CP=8 切分后单卡 16384 token, 与 --max-tokens-per-gpu 等量.
#
# Usage:
#   bash scripts/training/sft/run-qwen3.5-397B-A17B-mtp-sft-128k-128xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-397B-A17B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/mtp}"
EXP_NAME="${EXP_NAME:-qwen3.5-397B-A17B-mtp-sft-gpu128}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
LOAD_DIR="${LOAD_DIR:-${MODEL_DIR}/Qwen3.5-397B-A17B/}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/qwen3.5-397B-A17B-mtp-sft}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
RAY_ADDRESS="${RAY_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-397B-A17B/
   --ref-load ${MODEL_DIR}/Qwen3.5-397B-A17B/
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
   --input-key ${INPUT_KEY:-messages}
   # 不设 --label-key，使用完整 messages 模式
   --tool-key ${TOOL_KEY:-tools}
   --global-batch-size ${GLOBAL_BATCH_SIZE:-32}
   --use-dynamic-batch-size
   # 128K 上下文: CP=8 时单样本容量为 8*16384=131072 token
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
   --sft-tq-timeout-minutes ${SFT_TQ_TIMEOUT_MINUTES:-60}
   --balance-data
   --sft-oversize-strategy ${SFT_OVERSIZE_STRATEGY:-skip}
   --data-pad-size-multiplier 4096
)

# MTP (Multi-Token Prediction) 参数:
#   保留 MTP，对 tool calling 的格式学习有帮助
MTP_ARGS=(
   --mtp-num-layers ${MTP_NUM_LAYERS:-1}
   --enable-mtp-training
   --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}
)

# 性能/并行参数:
#   TP=2, PP=8, CP=8, EP=16, ETP=1, TP*PP*CP = 128
#   decoder-first=5, decoder-last=1, 中间 6 个 PP stage 各 9 层 (5+9*6+1=60)
PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE:-2}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE:-8}
   --context-parallel-size ${CP_SIZE:-8}
   --expert-model-parallel-size ${EP_SIZE:-16}
   --expert-tensor-parallel-size ${ETP_SIZE:-1}
   --decoder-first-pipeline-num-layers 5
   --decoder-last-pipeline-num-layers 1

   --log-probs-chunk-size 2048
   --recompute-loss-function

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --cross-entropy-loss-fusion
)

# 优化器参数:
#   lr=1e-5, cosine decay, min-lr=1e-6
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
   --resource '{"sft": [1, 0], "actor": [1, 128]}' \
   --max-staleness 0 \
   --num-data-storage-units 16 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3.5-397B-A17B-mtp-sft-gpu128-${now}.log
