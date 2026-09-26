#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B MTP SFT on OpenMathReasoning-mini, 16xGPU, ray-submit launch.
#
# Usage:
#   bash scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/mtp}"
EXP_NAME="${EXP_NAME:-qwen3.5-35b-a3b-mtp-sft-gpu16}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
LOAD_DIR="${LOAD_DIR:-${MODEL_DIR}/Qwen3.5-35B-A3B/}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/qwen3.5-35B-A3B-mtp-sft}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
RAY_ADDRESS="${RAY_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B/
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B/
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
   --input-key ${INPUT_KEY:-problem}
   --label-key ${LABEL_KEY:-generated_solution}
   --global-batch-size ${GLOBAL_BATCH_SIZE:-128}
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-8192}
   --balance-data
)

MTP_ARGS=(
   --mtp-num-layers ${MTP_NUM_LAYERS:-1}
   --enable-mtp-training
   --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE:-2}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE:-2}
   --context-parallel-size ${CP_SIZE:-1}
   --expert-model-parallel-size ${EP_SIZE:-4}
   --expert-tensor-parallel-size ${ETP_SIZE:-1}

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

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 16]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3.5-35b-a3b-mtp-sft-gpu16-${now}.log
