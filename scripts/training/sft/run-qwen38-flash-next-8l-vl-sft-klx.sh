#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next 8L VL SFT on the lmms-lab multimodal-open-r1 parquet, 8xklx.
#
# Derived from
#   scripts/training/multimodal/run-qwen38-flash-next-8l-vl-8xgpu-grpo-klx.sh
# keeping the model / parallel / trainable / KLX-env config identical, and
# swapping the RL half for SFT:
#   dropped  : SGLANG_ARGS, GRPO_ARGS, rollout sampling, --colocate,
#              --max-staleness, --rm-type, --system-prompt
#   added    : --loss-type sft, epoch/eval control, chunked SFT logits
#   resource : {"sft":[1,0],"actor":[1,8]} -- no rollout engine at all
#
# Data: train-00000-of-00001_sft_localpath.parquet (7689 rows), columns
#   messages : list<struct<content,role>>  full OpenAI turns; the assistant
#              turn IS the supervision target, so --label-key stays unset
#   images   : list<string>                paths into the sibling images/ dir,
#              one per row; the <image> placeholder sits in the user turn
#
# This is NOT the RL parquet. scripts/tools/process_openr1.py emits the GRPO
# layout (prompt / image<binary> / label), which the SFT loader rejects both
# ways: with --label-key the prompt column must be a plain string
# (relax/engine/sft/dataset/streaming.py:279-283), and without it the messages
# list carries no assistant turn, so loss_mask comes out all zeros. Build this
# parquet with scripts/tools/process_openr1_sft.py instead.
#
# Usage:
#   bash scripts/training/sft/run-qwen38-flash-next-8l-vl-sft-klx.sh
#   NUM_ROLLOUT=2 bash scripts/training/sft/run-qwen38-flash-next-8l-vl-sft-klx.sh  # smoke
#
# NOTE(agent): batch sizes / LR are smoke-scale for the reduced 8L checkpoint on
# one 8-card node; they are not tuned for convergence.

set -ex
set -o pipefail

export WORKDIR="${WORKDIR:-/workspace}"
set +x
export WANDB_API_KEY="${WANDB_API_KEY:=YOUR-KEY}"
set -x
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XPU_VISIBLE_DEVICES="${XPU_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RELAX_PROPAGATE_ENV_VARS="CUDA_VISIBLE_DEVICES,XPU_VISIBLE_DEVICES,RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
export RL_MODEL_TYPE="Qwen38_flash_multimodal"
export XMLIR_USE_HYDRA_LINEAR=${XMLIR_USE_HYDRA_LINEAR:-1}
export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-1}
export XMLIR_MATMUL_FAST_MODE=${XMLIR_MATMUL_FAST_MODE:-1}
export XMLIR_MEMCPY_RETRY_SYNC=${XMLIR_MEMCPY_RETRY_SYNC:-false}

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-eth0}
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7}
unset http_proxy && unset https_proxy

now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-klx.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen38-flash-next-8l.sh"

#############################  paths & checks  ################################

MODEL_DIR="${MODEL_DIR:-/data}"
DATA_DIR="${DATA_DIR:-/data/lmms-lab/multimodal-open-r1-8k-verified/data}"
MODEL_NAME="${VL_MODEL_NAME:-Qwen3.8-Flash-Next-8L-VL}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/train-00000-of-00001_sft_localpath.parquet}"

[ -d "${MODEL_DIR}/${MODEL_NAME}" ] || { echo "missing model dir ${MODEL_DIR}/${MODEL_NAME}" >&2; exit 2; }
[ -f "${PROMPT_DATA}" ] || { echo "missing sft parquet ${PROMPT_DATA}" >&2; exit 2; }

: "${NO_SAVE:=1}"
if [[ "${NO_SAVE}" != 0 && "${NO_SAVE}" != 1 ]]; then echo "NO_SAVE must be 0 or 1" >&2; exit 2; fi
if [[ "${NO_SAVE}" == 0 ]]; then : "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"; fi

PROJECT_NAME="${PROJECT_NAME:-qwen38-flash-next-sft-vl}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-8l-vl-sft-openr1mm-8xklx}"

TP_SIZE="${TP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"

#################################  args  ######################################

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --ref-load "${MODEL_DIR}/${MODEL_NAME}"
    --megatron-to-hf-mode bridge
    --num-epoch ${NUM_EPOCH:-100}
)
if [[ "${NO_SAVE}" == 0 ]]; then
    CKPT_ARGS+=(
        --load "${SAVE_DIR}/${EXP_NAME}"
        --save "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-50}"
        --max-actor-ckpt-to-keep "${MAX_ACTOR_CKPT_TO_KEEP:-1}"
    )
fi


# Messages mode: `--input-key` holds a full OpenAI messages list whose assistant
# turn is the training target, so `--label-key` must stay UNSET. See
# relax/engine/sft/dataset/streaming.py:233-295 -- with --label-key the prompt
# column must be a plain string, which this dataset's chat turns are not.
SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key ${INPUT_KEY:-messages}
    --apply-chat-template
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num ${IMAGE_MAX_TOKEN_NUM:-1024}
    --sft-oversize-strategy skip
    --global-batch-size ${GLOBAL_BATCH_SIZE:-32}
    --use-dynamic-batch-size
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-4096}
    --balance-data
    --sft-chunked-logits
)
if [ -n "${NUM_ROLLOUT:-}" ]; then
    SFT_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
fi

PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size 1
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-grouped-gemm
    --micro-batch-size 1
    --no-rope-fusion

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --moe-flex-dispatcher-backend deepep
    --moe-token-dispatcher-type flex
    --moe-grouped-gemm true
)
if [ "${CP_SIZE}" -gt 1 ]; then
    PARALLEL_ARGS+=(--calculate-per-token-loss)
fi

# Same freezes as the GRPO script: QSA indexer and the PLE n-gram table.
TRAINABLE_ARGS=(
    --freeze-params-name-list
    'self_attention\.indexer\.'
    'per_layer_embedding\.ple_embedding\.ngram_embedding\.'
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --lr-decay-style cosine
    --min-lr "${MIN_LR:-1e-6}"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --cpu-optimizer-param-device-cast
)

TRACKING_ARGS=(
    --tb-experiment-name qwen3.8-flash-next-sft
    --use-wandb
    --wandb-project "${PROJECT_NAME//\//-}"
    --wandb-group qwen38-flash-next-sft-klx-${now}
    --disable-wandb-random-suffix
    --no-use-metrics-service
)

MISC_ARGS=(
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --use-health-check
)

################################  launch  #####################################
EXTRA_ENV_VARS_JSON="
    \"RELAX_PROPAGATE_ENV_VARS\": \"${RELAX_PROPAGATE_ENV_VARS}\""

source "${SCRIPT_DIR}/../../entrypoint/runtime-env-klx.sh"
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft": [1, 0], "actor": [1, 8]}' \
    --sft-max-in-flight-steps 2 \
    --num-data-storage-units 1 \
    --selective-offload \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${TRAINABLE_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
