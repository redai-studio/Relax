#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xGPU colocate (sync) GRPO + MTP joint-training script.
#
# Phase-1 RL MTP: trains the native MTP head jointly with the policy via an
# auxiliary loss (slime-style). Rollout keeps `enable_draft_weights_cpu_backup=True`
# so SGLang inference uses the base model only — no speculative decoding here.
#
# Requires the HF checkpoint to contain MTP weights (`num_nextn_predict_layers>=1`).
#
# Usage:
#   bash scripts/training/text/run-qwen35-9B-mtp-8xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math-mtp}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/Qwen3.5-9B
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --load ${EXP_DIR}/Qwen3-9B_mtp_mcore_8xgpu/
   --save ${EXP_DIR}/Qwen3-9B_mtp_mcore_8xgpu/
   --save-interval 50
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   # --recompute-granularity full
   # --recompute-method uniform
   # --recompute-num-layers 1

   --use-distributed-optimizer
   --overlap-grad-reduce
   --overlap-param-gather

   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --log-probs-max-tokens-per-gpu 40960

   # --micro-batch-size 1 # avoid OOM
   --no-rope-fusion
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
   # icepop: drop tokens with ratio outside [tis-clip-low, tis-clip] instead of clamping (vanilla TIS).
   --custom-tis-function-path relax.backends.megatron.loss.icepop_function
)

# MTP joint-training: enables Megatron's MTP block + auxiliary loss.
# - mtp-num-layers must match the layer count in the HF checkpoint (num_nextn_predict_layers).
# - mtp-loss-scaling-factor defaults to 0.1 for RL (SFT uses 0.2) — RL gradients are noisier,
#   start conservative and tune up if MTP loss stalls without hurting policy loss.
MTP_ARGS=(
   --mtp-num-layers ${MTP_NUM_LAYERS:-1}
   --enable-mtp-training
   --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.1}
   --cross-entropy-loss-fusion
   --cross-entropy-fusion-impl native
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8

   # enable sglang MTP
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3
   --sglang-mamba-scheduler-strategy extra_buffer
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen35-9B-mtp-8x-${now}
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
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-9B-mtp-GRPO-gpu8-${now}.log
