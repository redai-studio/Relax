#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# GLM-5.3-Flash full language-model SFT with frozen vision on 256 GPUs.
# MODEL_DIR/GLM-5.3-Flash is the original HF checkpoint.
# DATA_DIR/train.kimi_k3.jsonl contains messages, tools, and image paths.
# From an existing Ray cluster:
#   MODEL_DIR=... DATA_DIR=... SAVE_DIR=... RELAX_SKIP_CLUSTER_CLEANUP=1 RAY_JOB_ADDRESS=http://ray-head:8265 \
#     bash scripts/entrypoint/ray-job.sh \
#       scripts/training/sft/run-glm5.3-flash-320B-A18B-full-256xgpu.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${RELAX_ROOT}/../../ckpt}"
DATA_DIR="${DATA_DIR:-${RELAX_ROOT}/../../dataset}"
SAVE_DIR="${SAVE_DIR:-${RELAX_ROOT}/../../ckpt/relax-sft}"
MODEL_NAME="${MODEL_NAME:-GLM-5.3-Flash}"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
ACTOR_GPUS="${ACTOR_GPUS:-256}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-4}"
# TP4 / PP4 / CP4 gives DP4. EP32 with ETP1 gives expert DP2.
EP_SIZE="${EP_SIZE:-32}"
RESOURCE="{\"actor\": [1, ${ACTOR_GPUS}], \"sft\": [1, 0]}"
TRAIN_FILE="${TRAIN_FILE:-train.kimi_k3.jsonl}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/glm5.3-flash}"
EXP_NAME="${EXP_NAME:-glm5.3-flash-kimi-k3-think-full-256xgpu-cp4-pp4-ep32-deepep-u2-vfrozen-ds1017-pp14-24-24-28}"
SEQ_LENGTH="${SEQ_LENGTH:-131072}"
# Keep a 128K packed-token budget when changing the CP size.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((131072 / CP_SIZE))}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
now=$(date "+%Y%m%d-%H%M%S")

if [[ ! -f "${DATA_DIR}/${TRAIN_FILE}" ]]; then
    echo "Missing training data: ${DATA_DIR}/${TRAIN_FILE}" >&2
    exit 1
fi

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
if [[ "${RELAX_ENTRYPOINT_MODE}" == "local" ]]; then
    RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://${MASTER_ADDR}:8265}"
fi
: "${RAY_JOB_ADDRESS:?Set RAY_JOB_ADDRESS to the Ray Jobs HTTP URL}"
source "${SCRIPT_DIR}/../../models/glm5.3-flash-320B-A18B.sh"

PIPELINE_ARGS=()
if (( PP_SIZE > 1 )); then
    if (( PP_SIZE == 4 && ${#flash_hybrid_pattern} == 90 )); then
        # DSA-aligned stages of 14, 24, 24, and 28 physical layers.
        # The first stage also owns the vision tower and input embeddings.
        flash_hybrid_pattern="${flash_hybrid_pattern:0:14}|${flash_hybrid_pattern:14:24}|${flash_hybrid_pattern:38:24}|${flash_hybrid_pattern:62}"
    fi
    PIPELINE_ARGS+=(--hybrid-layer-pattern "${flash_hybrid_pattern}")
fi

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 3
)

TRAIN_LENGTH_ARGS=(--num-epoch "${NUM_EPOCH:-2}")
if [[ -n "${NUM_ROLLOUT:-}" ]]; then
    TRAIN_LENGTH_ARGS=(--num-rollout "${NUM_ROLLOUT}")
fi

SFT_ARGS=(
    --loss-type sft
    --freeze-vision-model
    --freeze-vision-projection
    --vision-dp-when-tp
    --sft-loss-last-turn-only
    --prompt-data "${DATA_DIR}/${TRAIN_FILE}"
    --input-key messages
    --tool-key tools
    --apply-chat-template-kwargs '{"reasoning_effort":"high"}'
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num 1024
    "${TRAIN_LENGTH_ARGS[@]}"
    --rollout-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-shuffle
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --seq-length "${SEQ_LENGTH}"
    --data-pad-size-multiplier 128
    --sft-oversize-strategy skip
    --sft-invalid-multimodal-strategy skip
    --sft-max-in-flight-steps 2
    --sft-prefetch-num-workers 16
    --sft-prefetch-buffer-size 64
    --sft-tq-timeout-minutes 90
    --per-rank-fetch
    --sft-chunked-logits
    --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-1024}"
    --cross-entropy-loss-fusion
    --balance-data
)
if [[ "${RECOMPUTE_LOSS_FUNCTION:-1}" == 1 ]]; then
    SFT_ARGS+=(--recompute-loss-function)
fi

CP_ARGS=(--context-parallel-size "${CP_SIZE}")
if (( CP_SIZE > 1 )); then
    # DSA gathers keys across CP while packed tokens keep per-document zigzag placement.
    CP_ARGS+=(--cp-comm-type allgather)
fi

MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-flex}"
MOE_ARGS=(--moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE}")
if [[ "${MOE_TOKEN_DISPATCHER_TYPE}" == flex ]]; then
    MOE_ARGS+=(
        --moe-flex-dispatcher-backend "${MOE_FLEX_DISPATCHER_BACKEND:-deepep}"
        --moe-flex-dispatcher-num-sms "${MOE_FLEX_DISPATCHER_NUM_SMS:-24}"
    )
    if [[ "${MOE_SHARED_EXPERT_OVERLAP:-1}" == 1 ]]; then
        MOE_ARGS+=(--moe-shared-expert-overlap)
    fi
fi

PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --sequence-parallel
    --pipeline-model-parallel-size "${PP_SIZE}"
    "${PIPELINE_ARGS[@]}"
    "${CP_ARGS[@]}"
    --linear-cp-mode "${LINEAR_CP_MODE:-headwise}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    "${MOE_ARGS[@]}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-2}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --disable-weights-backuper
    # GLM Flash DSA selection reuse currently requires LoRA SFT.
    # Disable the async Inductor process pool to avoid fork deadlocks during cold compilation.
    --train-env-vars "{\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"CUDA_MODULE_LOADING\":\"${CUDA_MODULE_LOADING:-EAGER}\",\"TORCHINDUCTOR_COMPILE_THREADS\":\"${TORCHINDUCTOR_COMPILE_THREADS:-1}\",\"RELAX_SFT_SELECTIVE_LOGITS\":\"${RELAX_SFT_SELECTIVE_LOGITS:-1}\",\"RELAX_GLM53_DSA_SELECTION_REUSE\":\"0\",\"RELAX_SFT_FUSED_LINEAR_CE\":\"${RELAX_SFT_FUSED_LINEAR_CE:-0}\",\"RELAX_QWEN38_FLA_CONV_CACHE_REUSE\":\"${RELAX_QWEN38_FLA_CONV_CACHE_REUSE:-1}\"}"
    --calculate-per-token-loss
    --empty-unused-memory-level "${EMPTY_UNUSED_MEMORY_LEVEL:-1}"
    --attention-backend flash
    --attention-softmax-in-fp32
    --dsa-kernel-backend "${DSA_KERNEL_BACKEND:-tilelang}"
    --attention-dropout 0
    --hidden-dropout 0
    --seed "${SEED:-1234}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LEARNING_RATE:-1e-5}"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.05
    --min-lr "${MIN_LEARNING_RATE:-1e-6}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --clip-grad 1.0
    # Full fine-tuning also keeps GPU gradient buffers for the trainable model.
    --optimizer-cpu-offload
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --accumulate-allreduce-grads-in-fp32
)

EVAL_ARGS=()
if [[ -n "${EVAL_FILE:-}" ]]; then
    EVAL_ARGS=(
        --eval-prompt-data heldout "${DATA_DIR}/${EVAL_FILE}"
        --eval-input-key messages
        --eval-tool-key tools
        --eval-interval "${EVAL_INTERVAL:-50}"
    )
fi

METRICS_ARGS=(
    --use-tensorboard
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource "${RESOURCE}" \
    --num-data-storage-units 16 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${METRICS_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
