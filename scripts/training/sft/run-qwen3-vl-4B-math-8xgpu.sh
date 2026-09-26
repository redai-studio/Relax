#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-VL-4B SFT on prompt-data, 8xGPU single-node, ray-submit launch.
#
# Usage:
#   bash scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/math}"
EXP_NAME=qwen3-vl-4b-sft-math-gpu8
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
DATA_DIR="${DATA_DIR:=${SCRIPT_DIR}/data}"
PROMPT_DATA="${PROMPT_DATA:=${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
SAVE_DIR="${SAVE_DIR:=${SCRIPT_DIR}/../../../checkpoints/qwen3-vl-4B-math-sft}"

CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3-VL-4B-Instruct/
   --ref-load ${EXP_DIR}/Qwen3-VL-4B-Instruct/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}/sft/${EXP_NAME}
   --load ${SAVE_DIR}/sft/${EXP_NAME}
   --save-interval 1000
   --num-epoch 10
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key problem
   --label-key generated_solution
   --global-batch-size 256
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --balance-data
)

EVAL_ARGS=(
    --eval-size 0.01
    --eval-interval 100
)

PREDICT_ARGS=(
    --sft-predict-interval 100
    --eval-temperature 0.0
    --eval-max-response-len 10240
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --calculate-per-token-loss
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --no-rope-fusion

   --colocate
   --cross-entropy-loss-fusion
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
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

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PREDICT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-vl-4b-sft-math-gpu8-${now}.log
