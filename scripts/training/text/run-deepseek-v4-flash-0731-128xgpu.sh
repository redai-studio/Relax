#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# DeepSeek-V4-Flash-0731 colocate GRPO training on DAPO math, 128xGPU.
#
# Usage:
#   RAY_ADDRESS=http://ray-head:8265 MODEL_DIR=/path/to/models \
#     DATA_DIR=/path/to/data SAVE_DIR=/path/to/checkpoints \
#     bash scripts/entrypoint/ray-job.sh scripts/training/text/run-deepseek-v4-flash-0731-128xgpu.sh
#
# HF_CKPT, PROMPT_SET and EVAL_DATA can override the model and dataset paths.
# Optional CLEARML_CONFIG_FILE must point to a config visible on all worker nodes.
# Use an image with DeepSeek-V4 support in Megatron-Bridge, Megatron-LM and SGLang.
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

PROJECT_NAME="${PROJECT_NAME:-Relax/dev/deepseek-v4-flash-0731}"
EXP_NAME="${EXP_NAME:-deepseek-v4-flash-0731-text-gpu128}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/DeepSeek-V4-Flash-0731}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-${DATA_DIR}/aime-2024/aime-2024.jsonl}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/deepseek-v4-flash-0731-text}"
LOG_DIR="${LOG_DIR:-log}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

# Load the original release; the bridge dequantizes MXFP4/FP8 weights for training.
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${HF_CKPT}"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/${EXP_NAME}-${now}"
   --save-interval 50
   --max-actor-ckpt-to-keep 30
   # V4's config.json nests rope params and uses attribute_map aliases, which
   # trips the HF<->Megatron equality table in arguments.py:235-249.
   --skip-hf-validate
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   # Enable DeepSeek thinking mode consistently for rollout and evaluation.
   --apply-chat-template-kwargs '{"enable_thinking": true}'
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --use-fault-tolerance
   --rollout-health-check-timeout 120

   --rm-type deepscaler

   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1

   # 32 prompts * 8 samples = 256 trajectories per rollout.
   --global-batch-size 256
   --use-streaming-dataset
   --balance-data
)

FP8_ARGS=(
   --transformer-impl transformer_engine
   --bf16
   --fp8-format e4m3
   --fp8-recipe blockwise
)

EVAL_ARGS=(
   --log-passrate
   --eval-interval 10
   --eval-prompt-data aime "${EVAL_DATA}"
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 4096
   --eval-top-p 0.7
)

PERF_ARGS=(
   # 128 GPUs: TP=1, PP=8, CP=2, DP=8; EP=16, ETP=1.
   # DSv4 hybrid attention requires TP=1, without sequence parallelism.
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 8
   # Pass Megatron validation before the bridge installs the uneven DSv4 layout.
   --decoder-first-pipeline-num-layers 6

   # DSv4 uses contiguous packed sequences under context parallelism.
   --context-parallel-size 2
   --sequence-packing-scheduler dp_balanced
   --cp-partition-mode contiguous
   --allgather-cp

   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1
   --calculate-per-token-loss

   # Full recompute covers the hybrid attention and mHC layers.
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --recompute-loss-function

   --use-dynamic-batch-size
   # Per-rank budget after CP; the total packed-token budget is 2 * 6144.
   --max-tokens-per-gpu 6144
   --log-probs-chunk-size 4096

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type alltoall
   --moe-router-dtype fp32
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
   # Replay the rollout expert choices during actor training.
   --use-rollout-routing-replay
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

   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
   --no-rope-fusion
)

SGLANG_ARGS=(
   # 16 inference engines with 8 GPUs each, using DP attention and EP.
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.75
   --sglang-enable-dp-attention
   --sglang-dp-size 8
   --sglang-ep-size 8
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head

   --sglang-page-size 64
   --sglang-context-length 16384

   # Bound the KV pool and scheduler concurrency explicitly.
   --sglang-max-total-tokens 524288
   --sglang-max-running-requests 48

   --sglang-chunked-prefill-size 16384

   --sglang-watchdog-timeout 3600
   --sglang-enable-nan-detection

   # Colocate uses memory-saver mode, which does not support prefill CUDA graphs.
   --sglang-cuda-graph-backend-prefill disabled
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
   --use-health-check
   --trust-remote-code
   --update-weight-buffer-size $(( 1024 * 1024 * 1024 ))
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
    # Dequantize MXFP4 expert weights from the original release in SGLang.
    "SGLANG_DSV4_FP4_DEQUANT": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
    # Preconnect CP neighbors after colocated actor process-group reloads.
    "RELAX_DEBUG_DSV4_CP_P2P_WARMUP": "1",
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
   --resource '{"actor": [1, 128], "rollout": [1, 128]}' \
   --max-staleness 0 \
   --num-data-storage-units 16 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${FP8_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee "${LOG_DIR}/${EXP_NAME}-${now}.log"
