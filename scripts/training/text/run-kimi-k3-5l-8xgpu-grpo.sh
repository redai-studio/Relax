#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Kimi K3 5-layer smoke GRPO: 8xGPU colocate, text-only, native MXFP4 rollout.
#
# First RL path for Kimi K3 (reduced Kimi-K3-5L-128E-AttnRes4-MXFP4 release).
# The rollout engine loads the original MXFP4 release directly; on weight push
# the bridge converts trained BF16 routed experts back to the native packed
# pairs (w*.weight_packed + E8M0 w*.weight_scale) via Bridge's quantized
# mappings, so SGLang keeps serving the MXFP4 layout (see BridgeConverter.
# _convert_mxfp4_expert). Dense/vision weights are pushed as BF16, matching
# the release's ignore list.
#
# Layout: training TP2/EP4/ETP1 (+ sequence parallel, PP1/CP1), rollout one
# TP8 SGLang engine. Data: dapo-math-17k with the deepscaler (boxed) reward.
#
# Usage:
#   RAY_NO_WAIT=1 WORKING_DIR=./ RAY_ADDRESS=<ray:6379> \
#       bash scripts/entrypoint/ray-job.sh scripts/training/text/run-kimi-k3-5l-8xgpu-grpo.sh
# Or directly on a single node:
#   bash scripts/training/text/run-kimi-k3-5l-8xgpu-grpo.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-1}"
now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/kimi-k3-5layer.sh"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/kimi-k3-5l-grpo}"
EXP_NAME="${EXP_NAME:-kimi-k3-5l-8xgpu-grpo}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:?Set MODEL_DIR to the reduced HF model directory}}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR:?Set DATA_DIR to the prepared dataset directory}/dapo-math-17k.jsonl}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${HF_CHECKPOINT}"
   --megatron-to-hf-mode bridge
   --num-rollout "${NUM_ROLLOUT}"
)

# Optional full-state saving uses a stable directory across restarts.
if [ -n "${SAVE_DIR:-}" ]; then
    CKPT_ARGS+=(
        --load "${SAVE_DIR}/${EXP_NAME}"
        --save "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-50}"
        --max-actor-ckpt-to-keep 2
    )
fi

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --use-fault-tolerance
   --rollout-health-check-timeout 120

   --rm-type deepscaler

   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 2048
   --rollout-max-response-len 2048
   --rollout-temperature 1.0

   --global-batch-size 64
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --context-parallel-size 1
   --pipeline-model-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1
   --calculate-per-token-loss

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   # The reduced 5L checkpoint cannot solve dapo-math, so rewards are uniformly
   # zero and GRPO advantages vanish. The small entropy bonus keeps a real
   # gradient flowing so every step produces actual weight movement and the
   # MXFP4 push transfers changed data instead of a no-op.
   --entropy-coef 0.01
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

   --no-pin-cpu-grads
   --no-pin-cpu-params
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   # Reserve memory for colocated training and weight reload.
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-max-bs 8
   --sglang-server-concurrency 256
   --sglang-watchdog-timeout 3600
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --accumulate-allreduce-grads-in-fp32
   --trust-remote-code
   --update-weight-buffer-size $(( 4 * 512 * 1024 * 1024 ))
   # KDA training kernels: the tilelang FLA path segfaults on this stack; the
   # triton path is used for this recipe.
   # OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG puts the training forward on the same
   # MXFP4 grid the rollout engine serves (fake_qat_mxfp4.py), so the measured
   # train/rollout mismatch drops to engine numerics instead of the full
   # quantization error.
   --train-env-vars '{"FLA_TILELANG":"0","OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG":"1"}'
)

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
TRAIN_ARGS=(
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 8 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
   "${METRICS_ARGS[@]}"
   "$@"
)
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
RUNTIME_ENV_JSON=$(python3 -c '
import json, os
d = json.loads(os.environ["RUNTIME_ENV_JSON"])
d.setdefault("env_vars", {}).update({
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
})
print(json.dumps(d))
')
export RUNTIME_ENV_JSON

mkdir -p "${RELAX_ROOT}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RELAX_ROOT}/log/${EXP_NAME}-${now}.log"
