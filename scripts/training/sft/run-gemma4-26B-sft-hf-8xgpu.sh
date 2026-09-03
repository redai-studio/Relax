#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# gemma-4-26B-A4B-it (MoE) full-parameter SFT with packing, 8xGPU, TP=8 EP=8 PP=1,
# ray-submit launch. Same as run-gemma4-26B-sft-8xgpu.sh, plus a HuggingFace-format
# export written alongside the native torch_dist checkpoint.
#
# Usage:
#   MEGATRON=<gemma-4 tree> MODEL_DIR=<dir with gemma-4-26B-A4B-it> \
#   PROMPT_DATA=<sft.jsonl> bash scripts/training/sft/run-gemma4-26B-sft-hf-8xgpu.sh
#
#   # fp8 export instead of bf16 (~half the size; needs a safetensors --hf-checkpoint):
#   SAVE_HF_DTYPE=fp8 MEGATRON=... bash scripts/training/sft/run-gemma4-26B-sft-hf-8xgpu.sh

set -ex
set -o pipefail

unset NCCL_NVLS_ENABLE

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${RELAX:-${SCRIPT_DIR}/../../..}" &>/dev/null && pwd)"
export RELAX="${RELAX_ROOT}"

export MEGATRON="${MEGATRON:-/root/Megatron-LM/}"

if ! PYTHONPATH="${RELAX_ROOT}:${MEGATRON}:${PYTHONPATH:-}" python3 -c \
    'import inspect; from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider, Gemma4ModelProvider; from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge; from relax.models.gemma4.gemma4_bridge import Gemma4DenseBridge; from relax.models.gemma4.gemma4_provider import RELAX_PROVIDERS, PackedSafeGemma4DenseProvider, RelaxGemma4MoEProvider; source = inspect.getsource(Gemma4VLBridge.provider_bridge); ok = source.count("_conversion_mode()") >= 2 and RELAX_PROVIDERS.get(Gemma4DenseProvider) is PackedSafeGemma4DenseProvider and RELAX_PROVIDERS.get(Gemma4ModelProvider) is RelaxGemma4MoEProvider
if not ok:
    raise RuntimeError("Gemma4 packed-safe provider mapping is unavailable")' \
    >/dev/null 2>&1; then
    echo "ERROR: Gemma4 packed-safe integration failed its startup probe." >&2
    echo "       ${MEGATRON} must provide the required providers and MoE text mode," >&2
    echo "       and Relax's Gemma4 bridge/provider replacements must import cleanly." >&2
    exit 1
fi

MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${SCRIPT_DIR}/../../models}"
source "${MODEL_CONFIG_DIR}/gemma4-26B.sh"

TP_SIZE="${TP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-8}"

if [ "$((ACTOR_GPUS % TP_SIZE))" -ne 0 ]; then
    echo "ERROR: ACTOR_GPUS=${ACTOR_GPUS} is not divisible by TP_SIZE=${TP_SIZE}." >&2
    exit 1
fi
if [ "$((NHEADS % TP_SIZE))" -ne 0 ] || [ "$((NUM_QUERY_GROUPS % TP_SIZE))" -ne 0 ]; then
    echo "ERROR: TP_SIZE=${TP_SIZE} must divide NHEADS=${NHEADS} and NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS}." >&2
    exit 1
fi
if [ "$((MOE_ROUTED_EXPERTS % EP_SIZE))" -ne 0 ]; then
    echo "ERROR: EP_SIZE=${EP_SIZE} must divide MOE_ROUTED_EXPERTS=${MOE_ROUTED_EXPERTS}." >&2
    exit 1
fi
if [ "${TP_SIZE}" -lt 1 ] || [ "${TP_SIZE}" -gt 16 ] || [ "$((TP_SIZE & (TP_SIZE - 1)))" -ne 0 ]; then
    echo "ERROR: TP_SIZE=${TP_SIZE} must be a power of two between 1 and 16." >&2
    exit 1
fi
if [ "${EP_SIZE}" -ne "${ACTOR_GPUS}" ]; then
    echo "ERROR: EP_SIZE must equal ACTOR_GPUS (= TP * CP * DP with PP=CP=1)." >&2
    echo "       Got EP_SIZE=${EP_SIZE}, ACTOR_GPUS=${ACTOR_GPUS}, TP_SIZE=${TP_SIZE}." >&2
    echo "       EP smaller than that replicates experts on every rank and will OOM;" >&2
    echo "       larger is rejected by megatron." >&2
    exit 1
fi
echo "parallelism: TP=${TP_SIZE} EP=${EP_SIZE} PP=1 CP=1 on ${ACTOR_GPUS} GPU(s); num_layers=${NUM_LAYERS:-30}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" \
    | jq -c '.env_vars.GEMMA4_CONVERSION_MODE = "text"')

GEMMA4_SFT_THINKING="${GEMMA4_SFT_THINKING:-1}"
if [ -n "${GEMMA4_SFT_THINKING}" ]; then
    RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" \
        | jq -c --arg v "${GEMMA4_SFT_THINKING}" '.env_vars.GEMMA4_SFT_THINKING = $v')
fi

if [ -n "${NVTE_DEBUG:-}" ]; then
    RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" | jq -c \
        --arg d "${NVTE_DEBUG}" --arg l "${NVTE_DEBUG_LEVEL:-2}" \
        '.env_vars.NVTE_DEBUG = $d | .env_vars.NVTE_DEBUG_LEVEL = $l')
fi
export RUNTIME_ENV_JSON

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the dir containing gemma-4-26B-A4B-it}"
CKPT="${CKPT:-${MODEL_DIR}/gemma-4-26B-A4B-it}"
PROMPT_DATA="${PROMPT_DATA:?set PROMPT_DATA to an SFT jsonl}"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/gemma4}"
EXP_NAME="${EXP_NAME:-gemma4-26b-moe-sft-gpu8}"

CKPT_ROOT="${CKPT_ROOT:-/data/temp}"
SAVE_DIR="${SAVE_DIR:-${CKPT_ROOT}/gemma4-26B-sft}"

SAVE_HF_DIR="${SAVE_HF_DIR:-${SAVE_DIR}/hf_output/${EXP_NAME}}"
SAVE_HF_DTYPE="${SAVE_HF_DTYPE:-bf16}"

RAY_ADDRESS="${RAY_ADDRESS:-http://${MASTER_ADDR:-127.0.0.1}:${RAY_DASHBOARD_PORT:-8265}}"

CKPT_ARGS=(
   --hf-checkpoint "${CKPT}"
   --ref-load      "${CKPT}"
   --load          "${CKPT}"
   --save          "${SAVE_DIR}"

   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --save-interval 1000000
   --num-epoch 1

   --save-hf       "${SAVE_HF_DIR}/iter_{rollout_id}"
   --save-hf-dtype "${SAVE_HF_DTYPE}"
)

if [ "${SAVE_HF_DTYPE}" = "fp8" ]; then
    CKPT_ARGS+=(
        --save-hf-fp8-quant-mode "${SAVE_HF_FP8_QUANT_MODE:-block}"
        --save-hf-fp8-block-size ${SAVE_HF_FP8_BLOCK_SIZE:-128 128}
    )
fi

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key   "${INPUT_KEY:-instruction}"
   --label-key   "${LABEL_KEY:-output}"

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-4096}
   --balance-data

   --global-batch-size ${GLOBAL_BATCH_SIZE:-512}
   --seq-length ${SEQ_LENGTH:-4096}
   --num-rollout ${NUM_ROLLOUT:-40}
)

PERF_ARGS=(
   --num-layers ${NUM_LAYERS:-30}

   --tensor-model-parallel-size ${TP_SIZE}
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size 1
   --sequence-parallel

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --data-parallel-sharding-strategy optim
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --min-lr 5e-7
   --lr-decay-style ${LR_DECAY_STYLE:-cosine}
   --lr-warmup-fraction 0.1

   --weight-decay ${WEIGHT_DECAY:-0.1}
   --adam-beta1 0.9
   --adam-beta2 ${ADAM_BETA2:-0.95}

   --adam-eps 1e-8
   --clip-grad 1.0
)

PER_TOKEN_LOSS_FLAG=()
[ "${CALC_PER_TOKEN_LOSS:-1}" != "0" ] && PER_TOKEN_LOSS_FLAG=(--calculate-per-token-loss)

MISC_ARGS=(
   --bf16
   --attention-backend ${ATTENTION_BACKEND:-auto}
   --cross-entropy-fusion-impl te
   --log-interval 1
   --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES:-30}
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --recompute-loss-function
   --use-health-check
   --no-save-rng

   --seed ${SEED:-42}

   "${PER_TOKEN_LOSS_FLAG[@]}"
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}-${now}
)

mkdir -p "${SAVE_DIR}" log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
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
   "${WANDB_ARGS[@]}" 2>&1 | tee log/${EXP_NAME}-${now}.log
