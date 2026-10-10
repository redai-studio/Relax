#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# DeepSeek-V4-Flash-0731 SFT on OpenMathReasoning-mini, 128xGPU.
#
# Usage:
#   RAY_ADDRESS=http://ray-head:8265 MODEL_DIR=/path/to/models \
#     DATA_DIR=/path/to/data SAVE_DIR=/path/to/checkpoints \
#     bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-deepseek-v4-flash-sft-0731-128xgpu.sh
#
# HF_CKPT and PROMPT_DATA can override the model and dataset paths directly.
# Optional CLEARML_CONFIG_FILE must point to a config visible on all worker nodes.
# Use an image with DeepSeek-V4 support in Megatron-Bridge and Megatron-LM.
# MTP remains disabled because the release uses a different MTP head layout.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/deepseek-v4-flash.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/deepseek-v4-flash-0731}"
EXP_NAME="${EXP_NAME:-deepseek-v4-flash-0731-sft-gpu128}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/DeepSeek-V4-Flash-0731}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/deepseek-v4-flash-0731-sft}"
LOG_DIR="${LOG_DIR:-log}"

# Load the original release; the bridge dequantizes MXFP4/FP8 weights to bf16.
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${HF_CKPT}"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/${EXP_NAME}-${now}"
   --save-interval 50
   # V4's config.json nests rope params and uses attribute_map aliases, which
   # trips the HF<->Megatron equality table in arguments.py:235-249.
   --skip-hf-validate
   --max-actor-ckpt-to-keep 1
   --num-epoch 1
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key problem
   --label-key generated_solution
   --global-batch-size 128
   --use-dynamic-batch-size
   # CP=16 gives a total packed-token budget of 16 * 16384 = 262144.
   --max-tokens-per-gpu 16384
   --sft-tq-timeout-minutes 60
   --balance-data
   --sft-oversize-strategy skip
   --data-pad-size-multiplier 4096
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
   # 128 GPUs: TP=1, PP=8, CP=16, DP=1; EP=16, ETP=1.
   # DSv4 hybrid attention requires TP=1, without sequence parallelism.
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 8
   # Keep the three hash-MoE layers with the embedding on the first stage.
   --pipeline-model-parallel-layout "Ettt|tttt|ttttt|tttttt|tttttt|ttttttt|ttttttt|tttttL"
   --context-parallel-size 16
   # DSv4 uses contiguous packed sequences under context parallelism.
   --sequence-packing-scheduler dp_balanced
   --cp-partition-mode contiguous
   --allgather-cp
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1
   --calculate-per-token-loss

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --recompute-loss-function

   --log-probs-chunk-size 4096

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type alltoall
   --moe-router-dtype fp32
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

   # Disable the bridge pretraining defaults for router load balancing.
   --no-rope-fusion
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
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

RUNTIME_ENV_JSON=$(python3 -c '
import json
import os

d = json.loads(os.environ["RUNTIME_ENV_JSON"])
env_vars = d.setdefault("env_vars", {})
env_vars.update({
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "NCCL_NVLS_ENABLE": "0",
    # Reduce allocator fragmentation for long-context SFT.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
if os.environ.get("CLEARML_CONFIG_FILE"):
    env_vars["CLEARML_CONFIG_FILE"] = os.environ["CLEARML_CONFIG_FILE"]
print(json.dumps(d))
')
export RUNTIME_ENV_JSON

mkdir -p "${LOG_DIR}"
stdbuf -oL -eL ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 128], "sft": [1, 0]}' \
   --max-staleness 0 \
   --num-data-storage-units 16 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PREDICT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_DIR}/${EXP_NAME}-${now}.log"
