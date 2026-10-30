#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Gemma-4 31B full-parameter 64K SFT with global FlexAttention.
# Defaults to one 8-GPU node (TP8/DP1). NUM_NODES=2 automatically uses both
# Ray GPU nodes (TP8/DP2, 16 GPUs total).
#
# Single node:
#   CKPT=<gemma-4-31B-checkpoint> PROMPT_DATA=<data.jsonl> \
#     bash scripts/training/sft/run-gemma4-31B-sft-64k-8xgpu.sh
#
# Two nodes (run once on the Ray head):
#   NUM_NODES=2 GPUS_PER_NODE=8 CKPT=<gemma-4-31B-checkpoint> \
#     PROMPT_DATA=<data.jsonl> bash scripts/entrypoint/ray-job.sh \
#       scripts/training/sft/run-gemma4-31B-sft-64k-8xgpu.sh

set -ex
set -o pipefail

unset NCCL_NVLS_ENABLE

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

TP_SIZE="${TP_SIZE:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NUM_GPUS:-8}}"
NUM_NODES="${NUM_NODES:-${WORLD_SIZE:-1}}"
ACTOR_GPUS="${ACTOR_GPUS:-$((GPUS_PER_NODE * NUM_NODES))}"
SEQ_LENGTH="${SEQ_LENGTH:-65536}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-65536}"
SFT_LOGITS_CHUNK_SIZE="${SFT_LOGITS_CHUNK_SIZE:-512}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"

for name in TP_SIZE GPUS_PER_NODE NUM_NODES ACTOR_GPUS; do
    if ! [[ "${!name}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer." >&2
        exit 1
    fi
done
DP_SIZE="$((ACTOR_GPUS / TP_SIZE))"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${DP_SIZE}}"

if [ "$((16 % TP_SIZE))" -ne 0 ]; then
    echo "ERROR: TP_SIZE=${TP_SIZE} does not divide the 16 KV heads." >&2
    exit 1
fi
if [ "$((ACTOR_GPUS % TP_SIZE))" -ne 0 ]; then
    echo "ERROR: ACTOR_GPUS=${ACTOR_GPUS} is not divisible by TP_SIZE=${TP_SIZE}." >&2
    exit 1
fi
if [ "${ACTOR_GPUS}" -gt "$((GPUS_PER_NODE * NUM_NODES))" ]; then
    echo "ERROR: ACTOR_GPUS=${ACTOR_GPUS} exceeds ${NUM_NODES} node(s) x ${GPUS_PER_NODE} GPU(s)." >&2
    exit 1
fi
echo "parallelism: TP=${TP_SIZE} DP=${DP_SIZE} PP=1 CP=1 on ${ACTOR_GPUS} GPU(s)"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" | jq -c \
    '.env_vars.GEMMA4_CONVERSION_MODE = "text"
     | .env_vars.RELAX_GEMMA4_FORCE_FLEX_GLOBAL = "1"
     | .env_vars.PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"')

GEMMA4_SFT_THINKING="${GEMMA4_SFT_THINKING:-0}"
GEMMA4_SFT_TOOL_TURN_BOUNDARY="${GEMMA4_SFT_TOOL_TURN_BOUNDARY:-1}"
GEMMA4_SFT_MS_SWIFT_COMPAT="${GEMMA4_SFT_MS_SWIFT_COMPAT:-0}"
RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" | jq -c \
    --arg thinking "${GEMMA4_SFT_THINKING}" \
    --arg boundary "${GEMMA4_SFT_TOOL_TURN_BOUNDARY}" \
    --arg compat "${GEMMA4_SFT_MS_SWIFT_COMPAT}" \
    '.env_vars.GEMMA4_SFT_THINKING = $thinking
     | .env_vars.GEMMA4_SFT_TOOL_TURN_BOUNDARY = $boundary
     | .env_vars.GEMMA4_SFT_MS_SWIFT_COMPAT = $compat')

if [ -n "${NVTE_DEBUG:-}" ]; then
    RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" | jq -c \
        --arg d "${NVTE_DEBUG}" --arg l "${NVTE_DEBUG_LEVEL:-2}" \
        '.env_vars.NVTE_DEBUG = $d | .env_vars.NVTE_DEBUG_LEVEL = $l')
fi
export RUNTIME_ENV_JSON

source "${MODEL_CONFIG_DIR}/gemma4-31B.sh"

CKPT="${CKPT:?set CKPT to the Gemma-4 31B checkpoint}"
PROMPT_DATA="${PROMPT_DATA:?set PROMPT_DATA to an SFT jsonl}"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/gemma4}"
EXP_NAME="${EXP_NAME:-gemma4-31b-sft-64k-flex}"
CKPT_ROOT="${CKPT_ROOT:-${PWD}/checkpoints}"
SAVE_DIR="${SAVE_DIR:-${CKPT_ROOT}/gemma4-31B-sft-64k}"
SAVE_HF_DIR="${SAVE_HF_DIR:-${SAVE_DIR}/hf_output/${EXP_NAME}}"
SAVE_HF_DTYPE="${SAVE_HF_DTYPE:-bf16}"
SAVE_HF_POST_HOOK_PATH="${SAVE_HF_POST_HOOK_PATH-scripts.tools.gemma4_chat_template_hook.patch_gemma4_chat_template}"
DISABLE_CHECKPOINT_SAVE="${DISABLE_CHECKPOINT_SAVE:-0}"
case "${DISABLE_CHECKPOINT_SAVE}" in
    0 | 1) ;;
    *) echo "ERROR: DISABLE_CHECKPOINT_SAVE must be 0 or 1." >&2; exit 1 ;;
esac

RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-${RAY_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}}"

INPUT_KEY="${INPUT_KEY:-messages}"
if [ -z "${LABEL_KEY+x}" ]; then
    LABEL_KEY=""
fi
TOOL_KEY="${TOOL_KEY:-tools}"
METADATA_KEY="${METADATA_KEY:-meta_info}"

CKPT_ARGS=(
   --hf-checkpoint "${CKPT}"
   --ref-load      "${CKPT}"
   --load          "${CKPT}"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --num-epoch "${NUM_EPOCH:-1}"
)
if [ "${DISABLE_CHECKPOINT_SAVE}" = "0" ]; then
    CKPT_ARGS+=(
       --save "${SAVE_DIR}"
       --save-interval "${SAVE_INTERVAL:-1000000}"
       --save-hf "${SAVE_HF_DIR}/iter_{rollout_id}"
       --save-hf-dtype "${SAVE_HF_DTYPE}"
    )
    if [ -n "${SAVE_HF_POST_HOOK_PATH}" ]; then
        CKPT_ARGS+=(--save-hf-post-hook-path "${SAVE_HF_POST_HOOK_PATH}")
    fi
fi

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key "${INPUT_KEY}"
   --tool-key "${TOOL_KEY}"
   --metadata-key "${METADATA_KEY}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --balance-data
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --seq-length "${SEQ_LENGTH}"
   --sft-oversize-strategy "${SFT_OVERSIZE_STRATEGY:-skip}"
   --sft-chunked-logits
   --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE}"
   ${NUM_ROLLOUT:+--num-rollout ${NUM_ROLLOUT}}
)

if [ -n "${LABEL_KEY}" ]; then
    SFT_ARGS+=(--label-key "${LABEL_KEY}")
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --max-seqlen-per-dp-cp-rank "${MAX_TOKENS_PER_GPU}"
   --pad-packed-seq-alignment 8
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --data-parallel-sharding-strategy optim
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-5e-6}"
   --min-lr "${MIN_LR:-5e-7}"
   --lr-decay-style "${LR_DECAY_STYLE:-cosine}"
   --lr-warmup-fraction "${LR_WARMUP_FRACTION:-0.1}"
   --weight-decay "${WEIGHT_DECAY:-0.1}"
   --adam-beta1 "${ADAM_BETA1:-0.9}"
   --adam-beta2 "${ADAM_BETA2:-0.95}"
   --adam-eps "${ADAM_EPS:-1e-8}"
   --clip-grad "${CLIP_GRAD:-1.0}"
)

if [ "${OPTIMIZER_CPU_OFFLOAD}" = "1" ]; then
    OPTIMIZER_ARGS+=(
       --optimizer-cpu-offload
       --overlap-cpu-optimizer-d2h-h2d
       --use-precision-aware-optimizer
    )
fi

PER_TOKEN_LOSS_FLAG=()
[ "${CALC_PER_TOKEN_LOSS:-1}" != "0" ] && PER_TOKEN_LOSS_FLAG=(--calculate-per-token-loss)

MISC_ARGS=(
   --bf16
   --attention-backend auto
   --cross-entropy-fusion-impl te
   --log-interval 1
   --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-30}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --recompute-loss-function
   --use-health-check
   --no-save-rng
   --seed "${SEED:-42}"
   "${PER_TOKEN_LOSS_FLAG[@]}"
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

mkdir -p log
if [ "${DISABLE_CHECKPOINT_SAVE}" = "0" ]; then
    mkdir -p "${SAVE_DIR}"
fi

# Optional extra train args injected by outer wrappers (env string -> array).
# e.g. EXTRA_TRAIN_ARGS="--max-actor-ckpt-to-keep 0 --use-wandb ..."
EXTRA_TRAIN_ARGS=(${EXTRA_TRAIN_ARGS:-})
if [ "${DISABLE_CHECKPOINT_SAVE}" = "1" ]; then
    for arg in "${EXTRA_TRAIN_ARGS[@]}"; do
        case "${arg}" in
            --save* | --rotate-ckpt* | --async-save*)
                echo "ERROR: ${arg} conflicts with DISABLE_CHECKPOINT_SAVE=1." >&2
                exit 1
                ;;
        esac
    done
fi

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"sft\": [1, 0], \"actor\": [1, ${ACTOR_GPUS}]}" \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${EXTRA_TRAIN_ARGS[@]}" 2>&1 | tee "log/${EXP_NAME}-${now}.log"
