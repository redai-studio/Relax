#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xGPU async training script (ray-job mode).
# The Ray cluster is managed externally — do NOT kill ray or start a new cluster.
#
# CISPO (Clipped Importance-ratio Soft Policy Optimization) variant of
# run-qwen35-9B-8xgpu-openr1mm-async.sh. Only the advantage estimator differs.
#
# Usage:
#   bash examples/algorithms/run-qwen35-9B-8xgpu-openr1mm-cispo-async.sh [async|sync]

set -ex
set -o pipefail

MODE=${1:-"async"}

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"
# source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/fully_async_openr1mm}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/Qwen3.5-9B
   # --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-4B-Instruct
   # --ref-load ${MODEL_DIR}/Qwen3-VL-4B-Instruct
   --load ${EXP_DIR}/Qwen3.5-9B_mcore_8xgpu/
   --save ${EXP_DIR}/Qwen3.5-9B_mcore_8xgpu/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)

PROMPT_SET=${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet

SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   # --rollout-shuffle
   --rm-type openr1mm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-max-prompt-len 2048
   --rollout-temperature 0.8
   --global-batch-size 256
   --multimodal-keys '{"image":"image"}'
   --system-prompt "${SYSTEM_PROMPT}"
   --use-streaming-dataset
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 3
   # --decoder-last-pipeline-num-layers
   --decoder-first-pipeline-num-layers 8

   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1
   --calculate-per-token-loss
   --micro-batch-size 2
   # --qkv-format bshd
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384

   --no-rope-fusion
)

CISPO_ARGS=(
   # --use-kl-loss
   --advantage-estimator cispo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 10
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
   --clip-grad 1.0
)

WANDB_ARGS=(
   --use-tensorboard
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen35-9b-CISPO-gpu8-${MODE}-${now}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
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


mkdir -p log
if [ ${MODE} = "async" ]; then
     ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 -m relax.entrypoints.train \
        --resource '{"actor": [1, 6], "rollout": [1, 2], "advantages": [1, 0]}'\
        --max-staleness 2 \
        --num-data-storage-units 1 \
        --num-iters-per-train-update 16 \
        --fully-async \
        --use-health-check \
        "${MODEL_ARGS[@]}" \
        "${CKPT_ARGS[@]}" \
        "${ROLLOUT_ARGS[@]}" \
        "${OPTIMIZER_ARGS[@]}" \
        "${CISPO_ARGS[@]}" \
        "${WANDB_ARGS[@]}" \
        "${PERF_ARGS[@]}" \
        "${SGLANG_ARGS[@]}" \
        "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-9b-CISPO-gpu8-async-${now}.log
else
    ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
         --runtime-env-json="${RUNTIME_ENV_JSON}" \
         -- python3 -m relax.entrypoints.train \
         --resource '{"actor": [1, 8], "rollout": [1, 8]}'\
         --max-staleness 0 \
         --num-data-storage-units 1 \
         --colocate \
         --use-health-check \
         --balance-data \
         "${MODEL_ARGS[@]}" \
         "${CKPT_ARGS[@]}" \
         "${ROLLOUT_ARGS[@]}" \
         "${OPTIMIZER_ARGS[@]}" \
         "${CISPO_ARGS[@]}" \
         "${WANDB_ARGS[@]}" \
         "${PERF_ARGS[@]}" \
         --tensor-model-parallel-size 4 \
         --pipeline-model-parallel-size 2 \
         "${SGLANG_ARGS[@]}" \
         "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-9b-CISPO-gpu8-colocate-${now}.log
fi
