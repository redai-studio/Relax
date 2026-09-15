#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-VL-4B 2xGPU colocate training script.
#
# Usage:
#   NUM_GPUS=2 bash scripts/training/multimodal/run-qwen3-vl-4B-2xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn | head -n 2 | cut -d, -f1 | paste -sd ',')

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/openr1mm}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-4B-Instruct/
   --ref-load ${MODEL_DIR}/Qwen3-VL-4B-Instruct/
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
   --use-streaming-dataset
   --streaming-buffer-size 1000
   # --rollout-shuffle
   --balance-data
   --rm-type openr1mm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-max-prompt-len 2048
   --rollout-temperature 0.8
    --global-batch-size 16
    --multimodal-keys '{"image":"image"}'
    --system-prompt "${SYSTEM_PROMPT}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --micro-batch-size 2 # avoid OOM
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --use-kl-loss
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
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
   --clip-grad 1.0
   --no-rope-fusion
)

WANDB_ARGS=(
   # --use-tensorboard
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name qwen3-vl-4b-GRPO-gpu2-${now}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
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

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 2], "rollout": [1, 2]}'\
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-vl-4b-GRPO-gpu2-${now}.log
