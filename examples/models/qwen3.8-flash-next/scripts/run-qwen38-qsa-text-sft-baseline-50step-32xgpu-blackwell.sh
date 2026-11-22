#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.8 Flash Next text SFT on 4 x 8 Blackwell SM103, 128K context, global batch 512.
# TP2/PP4/CP1/EP8 with distributed Adam and HybridEP.
# Run with MODEL_DIR, DATA_DIR, SAVE_DIR, MEGATRON and RAY_JOBS_ADDRESS set.
# Use the shared patched Megatron checkout. DRY_RUN=1 prints the effective configuration.
# Runs 50 updates, reports to online ClearML and saves a checkpoint at update 50.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export RELAX="${RELAX_ROOT}"
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"
: "${MODEL_DIR:?Set MODEL_DIR to the HF model directory}"
: "${DATA_DIR:?Set DATA_DIR to the prompt-data directory or file}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${MEGATRON:?Set MEGATRON to the compatible Megatron-LM checkout}"
# Use the unified image's installed Bridge, Python dependencies and HybridEP.
# Do not prepend the pre-refactor /opt/qwen38-flash-next runtime.
: "${RELAX_HYBRID_EP_COMMIT:?Use the unified Blackwell image with HybridEP installed}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
export EXP_NAME="${EXP_NAME:-qwen38-qsa-text-baseline-50step-blackwell}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa}"
# Preserve this recipe's setting when ray-job.sh resets the generic variable.
export QWEN38_CUDA_DEVICE_MAX_CONNECTIONS="${QWEN38_CUDA_DEVICE_MAX_CONNECTIONS:-32}"
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"

# Use an exclusive Ray cluster; the operator must prevent concurrent training.
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    : "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the Ray Jobs HTTP endpoint}"
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi

source "${RELAX_ROOT}/scripts/training/sft/qwen38-blackwell-env.sh"
qwen38_blackwell_env pp4

PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --pipeline-model-parallel-layout "Et*12|t*14|t*14|t*8mL"
    --sequence-parallel
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend hybridep
    --moe-grouped-gemm
    --colocate
    --no-rope-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size 2048
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 2
    --no-offload-train
)

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --num-rollout "${NUM_STEPS:-50}"
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 2
)

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}"
    --eval-size 512
    --input-key messages
    --tool-key tools
    --apply-chat-template
    --apply-chat-template-kwargs '{"add_generation_prompt":false,"preserve_thinking":true}'
    --sft-invalid-multimodal-strategy skip
    --sft-oversize-strategy skip
    --calculate-per-token-loss
    --rollout-max-context-len 131072
    --global-batch-size 512
    --micro-batch-size 1
    --seq-length 131072
    --use-dynamic-batch-size
    --max-tokens-per-gpu 131072
    --data-pad-size-multiplier 16384
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-train-data-prefetch
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list '(^|\.)per_layer_embedding\.|self_attention\.indexer\.'
    --enable-mtp-training
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --mtp-detach-paths none
    --eval-interval "${EVAL_INTERVAL:-100}"
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 2e-5
    --lr-decay-style cosine
    --lr-decay-iters "${LR_DECAY_ITERS:-2023}"
    --min-lr 2e-6
    --lr-warmup-fraction 0.05
    --lr-warmup-init 0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --use-precision-aware-optimizer
    --no-store-param-remainders
    --moe-aux-loss-coeff 0
)

MISC_ARGS=(
    --seed 1234
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --train-env-vars "${TRAIN_ENV_JSON}"
    --moe-flex-dispatcher-num-sms 80
    # Keep the default balanced packing used by this baseline.
    --moe-hybridep-pad-variable-tokens
    --moe-shared-expert-overlap
    --overlap-grad-reduce
    --overlap-param-gather
)
if [[ "${QWEN38_SFT_HEALTH_CHECK:-1}" == 1 ]]; then
    MISC_ARGS+=(--use-health-check)
fi

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

TRAIN_ARGS=(
    --resource '{"sft":[1,0],"actor":[1,32]}'
    --actor-num-nodes 4
    --actor-num-gpus-per-node 8
    --sft-max-in-flight-steps 4
    --num-data-storage-units 2
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "$@"
)

qwen38_write_config

: "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the Ray Jobs HTTP endpoint}"
cd "${RELAX}"
mkdir -p "${SAVE_DIR}/logs"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOBS_ADDRESS}" \
    --submission-id "${SUBMISSION_ID}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SAVE_DIR}/logs/${EXP_NAME}-${now}.log"
