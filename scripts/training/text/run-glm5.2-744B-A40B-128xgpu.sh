#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# GLM-5.2 / GLM-5.3 BF16 GRPO, 128 physical GPUs, colocated Actor/Rollout.
# Baseline: docker/Dockerfile (SGLang 0.5.17) or Dockerfile.cu13 (0.5.21),
# including their Megatron/SGLang patches. GPU validation is still required.
# MODEL_DIR, DATA_DIR and SAVE_DIR must be visible on all nodes.
# For GLM-5.3 set MODEL_NAME=GLM-5.3-BF16 (the default GLM-5.3 release is FP8).
# Submit through the established cluster entrypoint, for example:
#   bash scripts/entrypoint/ray-job.sh scripts/training/text/run-glm5.2-744B-A40B-128xgpu.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
MODEL_NAME="${MODEL_NAME:-GLM-5.2}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/dapo-math}"
EXP_NAME="${EXP_NAME:-${MODEL_NAME}-dapo-128xgpu-colocate}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
now=$(date "+%Y%m%d-%H%M%S")

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
if [ "${RELAX_ENTRYPOINT_MODE}" = local ]; then
    RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://${MASTER_ADDR}:8265}"
fi
: "${RAY_JOB_ADDRESS:?Set RAY_JOB_ADDRESS to the Ray Jobs HTTP URL}"
source "${SCRIPT_DIR}/../../models/glm5.3-744B-A40B.sh"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --ref-load "${MODEL_DIR}/${MODEL_NAME}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval 20
    --max-actor-ckpt-to-keep 2
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size 8
    --n-samples-per-prompt 8
    --rollout-max-response-len 8192
    --rollout-temperature 1
    --global-batch-size 64
    --balance-data
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 4
    # 22/20/20/16 layers: stages start at global layers 1/23/43/63.
    # Each starts on a full indexer; shared top-k never crosses PP boundaries.
    --decoder-first-pipeline-num-layers 22
    --decoder-last-pipeline-num-layers 16
    --context-parallel-size 2
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --data-pad-size-multiplier 4096
    --log-probs-chunk-size 1024
    --moe-flex-dispatcher-backend deepep
    --moe-token-dispatcher-type flex
    --calculate-per-token-loss
    # The discrete top-k path has no indexer auxiliary training loss.
    --freeze-params-name-list self_attention.wq_b self_attention.wk self_attention.k_norm self_attention.weights_proj
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.0
    --entropy-coef 0.0
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
    --rollout-num-gpus-per-engine 64
    --sglang-mem-fraction-static 0.7
    --sglang-enable-dp-attention
    --sglang-ep-size 64
    --sglang-dp-size 64
    --sglang-moe-dense-tp-size 1
    --sglang-enable-dp-lm-head
    --sglang-moe-a2a-backend deepep
    --sglang-deepep-mode auto
    --sglang-load-format dummy
    --sglang-page-size 64
    --sglang-dsa-decode-backend flashmla_sparse
    --sglang-dsa-prefill-backend flashmla_sparse
    --sglang-attention-backend dsa
    --sglang-cuda-graph-max-bs 8
    --sglang-max-running-requests 512
    --sglang-chunked-prefill-size 16384
    --sglang-watchdog-timeout 3600
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
    --skip-eval-before-train
    --update-weight-buffer-size 1073741824
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS}" \
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
    "${PERF_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
