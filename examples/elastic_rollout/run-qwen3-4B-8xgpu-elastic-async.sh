#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B fully-async Rollout recipe with metric-driven autoscaling.
#
# Baseline layout: 4 Actor GPUs + 4 Rollout GPUs (four 1-GPU engines).
# Additional Rollout engines are requested at runtime and require an external
# Ray/Kubernetes or cloud resource provisioner to supply the GPUs.
#
set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "Current time: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/examples/qwen3-4b-elastic}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}"
AUTOSCALER_CONFIG="${AUTOSCALER_CONFIG:-examples/elastic_rollout/autoscaler.yaml}"

ELASTIC_ARGS=(
   # Required: deploy the Relax Autoscaler. The YAML must set enabled: true.
   --autoscaler-config "${AUTOSCALER_CONFIG}"
)

# Plain Ray clusters without node-group marker resources must opt out.
if [ "${ENABLE_AFFINITY:-1}" = "0" ]; then
   ELASTIC_ARGS+=(--no-enable-affinity)
fi

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-4B/"
   --ref-load "${MODEL_DIR}/Qwen3-4B/"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save "${EXP_DIR}/Qwen3-4B_mcore_8xgpu_elastic/"
   --save-interval 100
)

PROMPT_SET="${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --skip-eval-before-train
   --log-passrate
   --eval-interval 20
   --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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
)

SGLANG_ARGS=(
   # 4B topology: one elastic Rollout engine consumes one GPU.
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
)

METRICS_ARGS=(
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "qwen3-4b-GRPO-gpu8-elastic-${now}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

PIPELINE_ARGS=(
   # Required: elastic Rollout is supported in Fully Async mode.
   --fully-async
   # 4B baseline topology: 4 Actor GPUs and four protected 1-GPU Rollout engines.
   --resource '{"actor": [1, 4], "rollout": [1, 4], "advantages": [1, 0]}'
   --max-staleness 0
   --num-data-storage-units 1
   --num-iters-per-train-update 1
   --true-on-policy-mode
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   "${PIPELINE_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${METRICS_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${ELASTIC_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen3-4b-GRPO-gpu8-elastic-${now}.log"
