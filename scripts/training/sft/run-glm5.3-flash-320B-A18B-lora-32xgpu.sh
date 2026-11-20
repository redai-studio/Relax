#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# GLM-5.3-Flash multimodal LoRA SFT on 32 GPUs.
# MODEL_DIR/GLM-5.3-Flash is the original HF checkpoint.
# DATA_DIR/train.jsonl contains messages, tools, and image paths.
# From an existing Ray cluster:
#   MODEL_DIR=... DATA_DIR=... SAVE_DIR=... RAY_JOB_ADDRESS=http://ray-head:8265 \
#     bash scripts/entrypoint/ray-job.sh \
#       scripts/training/sft/run-glm5.3-flash-320B-A18B-lora-32xgpu.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${RELAX_ROOT}/../../ckpt}"
DATA_DIR="${DATA_DIR:-${RELAX_ROOT}/../../dataset}"
SAVE_DIR="${SAVE_DIR:-${RELAX_ROOT}/../../ckpt/relax-sft}"
MODEL_NAME="${MODEL_NAME:-GLM-5.3-Flash}"
HF_CHECKPOINT="${MODEL_DIR}/${MODEL_NAME}"
ACTOR_GPUS="${ACTOR_GPUS:-32}"
if [[ ! "${ACTOR_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ACTOR_GPUS must be a positive integer" >&2
    exit 1
fi
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-2}"
# Shard the experts across the GPUs in each pipeline stage.
EP_SIZE="${EP_SIZE:-$((ACTOR_GPUS / PP_SIZE))}"
RESOURCE="{\"actor\": [1, ${ACTOR_GPUS}], \"sft\": [1, 0]}"
TRAIN_FILE="${TRAIN_FILE:-train.jsonl}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/glm5.3-flash}"
EXP_NAME="${EXP_NAME:-glm5.3-flash-multimodal-lora-32xgpu}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
# Keep a 128K packed-token budget when changing the CP size.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((131072 / CP_SIZE))}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
now=$(date "+%Y%m%d-%H%M%S")

if [[ ! -f "${DATA_DIR}/${TRAIN_FILE}" ]]; then
    echo "Missing training data: ${DATA_DIR}/${TRAIN_FILE}" >&2
    exit 1
fi

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    if [[ -n "${RAY_ADDRESS:-}" ]] && timeout 5 ray status >/dev/null 2>&1; then
        source "${SCRIPT_DIR}/../../entrypoint/ray-job.sh"
    else
        if ! visible_gpus=$(python3 -c 'import torch; print(torch.cuda.device_count())'); then
            echo "Cannot detect local CUDA GPUs. Check the training environment or use an existing Ray cluster." >&2
            exit 1
        fi
        NUM_GPUS="${NUM_GPUS:-${visible_gpus}}"
        if [[ ! "${NUM_GPUS}" =~ ^[0-9]+$ ]] ||
            (( ACTOR_GPUS > 10#${NUM_GPUS} || 10#${NUM_GPUS} > visible_gpus )); then
            echo "Local Flash SFT requests ${ACTOR_GPUS} GPUs (NUM_GPUS=${NUM_GPUS}, visible CUDA GPUs=${visible_gpus})." >&2
            echo "Configure enough local GPUs or use an existing Ray cluster via scripts/entrypoint/ray-job.sh." >&2
            exit 1
        fi
        source "${SCRIPT_DIR}/../../entrypoint/local.sh"
    fi
fi
if [[ "${RELAX_ENTRYPOINT_MODE}" == "local" ]]; then
    RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://${MASTER_ADDR}:8265}"
fi
: "${RAY_JOB_ADDRESS:?Set RAY_JOB_ADDRESS to the Ray Jobs HTTP URL}"
source "${SCRIPT_DIR}/../../models/glm5.3-flash-320B-A18B.sh"

PIPELINE_ARGS=()
if (( PP_SIZE > 1 )); then
    # The CLI needs the physical layer count for PP validation. Bridge chooses
    # the stage boundaries from the same pattern during provider finalization.
    PIPELINE_ARGS+=(--hybrid-layer-pattern "${flash_hybrid_pattern}")
fi

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 3
    --save-lora-only
)

LORA_ARGS=(
    --lora-rank "${LORA_RANK:-16}"
    --lora-alpha "${LORA_ALPHA:-32}"
    --lora-dropout 0
    --lora-scope all
    --lora-merge-mode
    --lora-target-modules in_proj beta_proj f_a_proj f_b_proj g_a_proj g_b_proj out_proj linear_q_down_proj linear_q_up_proj linear_kv_down_proj linear_kv_up_proj linear_proj linear_fc1 linear_fc2
    --freeze-vision-model
)
if [[ "${VISION_DP_WHEN_TP:-1}" == 1 ]]; then
    LORA_ARGS+=(--vision-dp-when-tp)
fi

TRAIN_LENGTH_ARGS=(--num-epoch "${NUM_EPOCH:-2}")
if [[ -n "${NUM_ROLLOUT:-}" ]]; then
    TRAIN_LENGTH_ARGS=(--num-rollout "${NUM_ROLLOUT}")
fi

SFT_ARGS=(
    --loss-type sft
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
    --data-pad-size-multiplier 4096
    --sft-oversize-strategy skip
    --sft-invalid-multimodal-strategy skip
    --sft-max-in-flight-steps 4
    --sft-prefetch-num-workers 16
    --sft-prefetch-buffer-size 256
    --sft-tq-timeout-minutes 60
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
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-6
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --disable-weights-backuper
    --train-env-vars "{\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\",\"CUDA_MODULE_LOADING\":\"${CUDA_MODULE_LOADING:-EAGER}\",\"RELAX_SFT_SELECTIVE_LOGITS\":\"${RELAX_SFT_SELECTIVE_LOGITS:-1}\",\"RELAX_QWEN38_FLA_CONV_CACHE_REUSE\":\"${RELAX_QWEN38_FLA_CONV_CACHE_REUSE:-1}\"}"
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
    --min-lr 1e-6
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --optimizer-cpu-offload
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
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
    --num-data-storage-units 4 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${METRICS_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
