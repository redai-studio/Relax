#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B MTP-only SFT on pokemon-gpt4o-captions, 8xGPU single-node, ray-submit launch.
# The backbone, vision tower, projection, and LM head are frozen; only parameters
# whose names belong to the MTP block are optimized.
#
# By default this calibrates MTP from the HF checkpoint produced by
# run-qwen3.5-35B-A3B-pokemon-8xgpu.sh. Override INIT_HF_DIR when using another
# SFT checkpoint.
#
# Usage:
#   INIT_HF_DIR=/path/to/pokemon-sft-hf \
#     bash scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-mtp-only-8xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/pokemon-mtp-only}"
EXP_NAME="${EXP_NAME:-qwen3.5-35B-A3B-pokemon-mtp-only-gpu8}"
EXP_DIR="${EXP_DIR:-${MODEL_DIR:-${SCRIPT_DIR}/../../../../exps}}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
SAVE_DIR="${SAVE_DIR:=${SCRIPT_DIR}/../../../checkpoints/qwen3.5-35B-A3B-pokemon-sft}"
INIT_HF_DIR="${INIT_HF_DIR:-${SAVE_DIR}/sft/qwen3.5-35B-A3B-sft-pokemon-gpu8-hf}"
RAY_ADDRESS="${RAY_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}"

TRAIN_FILES=(
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet'"
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

CKPT_ARGS=(
   --hf-checkpoint "${INIT_HF_DIR}"
   --ref-load "${INIT_HF_DIR}"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/sft/${EXP_NAME}"
   --save-hf "${SAVE_DIR}/sft/${EXP_NAME}-hf"
   --load "${SAVE_DIR}/sft/${EXP_NAME}"
   --save-interval "${SAVE_INTERVAL:-500}"
   --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
   --num-epoch "${NUM_EPOCH:-10}"
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key conversations
   --multimodal-keys '{"image":"images"}'
   --conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
   --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-20480}"
   --balance-data
   --per-rank-fetch
   --sft-prefetch-num-workers "${SFT_PREFETCH_NUM_WORKERS:-16}"
   --sft-prefetch-buffer-size "${SFT_PREFETCH_BUFFER_SIZE:-512}"
)

MTP_ARGS=(
   --mtp-only-training
   --mtp-num-layers 1
   --mtp-loss-scaling-factor "${MTP_LOSS_SCALING_FACTOR:-0.2}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0

   --no-rope-fusion
   --cross-entropy-loss-fusion
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-5}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR:-1e-6}"
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --use-tensorboard
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --sft-max-in-flight-steps 4 \
   --num-data-storage-units 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/${EXP_NAME}-${now}.log"
