#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Two-node MiniMax-M3.1-Flash SFT on the four-layer BF16 checkpoint.
# Set SFT_MODE=full to train without LoRA, including the vision tower and projectors.
# Prepare it from a local copy of the official checkpoint:
# python scripts/tools/convert_minimax_m3_to_bf16.py --model-dir "$MODEL_DIR/MiniMax-M3.1-Flash" \
#   --layers 0,1,3,4 --save-dir "$MODEL_DIR/MiniMax-M3.1-Flash-4L-BF16" --device cpu
# Provide "$DATA_DIR/minimax-m3.1-flash-sft.jsonl" with messages.
# For image/video training, set MULTIMODAL_KEYS='{"image":"images","video":"videos"}'.
# Use <image>/<video> in messages and put the matching media paths in those fields.
# For Relax/Miles parity, set MINIMAX_NATIVE_PARITY=1 and provide the shared SFT_CHAT_TEMPLATE.
# Use MINIMAX_PARITY_LAYERS=4 for LoRA or full SFT, or 60 for full-depth LoRA.
# Launch through scripts/entrypoint/ray-job.sh on a running 2x8 GPU Ray cluster.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${SCRIPT_DIR}/../../entrypoint/ray-job.sh"
fi

: "${MODEL_DIR:?Set MODEL_DIR to the shared model root}"
: "${DATA_DIR:?Set DATA_DIR to the shared dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to the shared checkpoint root}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray Jobs dashboard HTTP address}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
MULTIMODAL_KEYS="${MULTIMODAL_KEYS:-}"
SFT_MODE="${SFT_MODE:-lora}"
native_parity="${MINIMAX_NATIVE_PARITY:-${MINIMAX_TEXT_PARITY:-0}}"
SAVE_INTERVAL=1
GLOBAL_BATCH_SIZE=16
MAX_TOKENS=4096
checkpoint_name=MiniMax-M3.1-Flash-4L-BF16
layer_count=4
moe_layer_freq='[0,0,1,1]'
case "$SFT_MODE" in
    lora) TP=2; EP=4; LR=1e-4; PARITY_NAME=minimax-m3.1-flash-4l-parity ;;
    full) TP=4; EP=8; LR=1e-5; PARITY_NAME=minimax-m3.1-flash-4l-full-parity ;;
    *) echo "Set SFT_MODE to lora or full" >&2; exit 1 ;;
esac
if [[ "$native_parity" == 1 ]]; then
    if [[ "${MINIMAX_MSA_IMPL:-relax}" != relax ]]; then
        echo "Native framework parity requires Relax MSA" >&2
        exit 1
    fi
    MINIMAX_PARITY=1
    layer_count="${MINIMAX_PARITY_LAYERS:-${MINIMAX_TEXT_PARITY_LAYERS:-4}}"
    case "$layer_count" in
        4) ;;
        60)
            if [[ "$SFT_MODE" == full ]]; then
                echo "The two-node native parity recipe uses four layers for full SFT" >&2
                exit 1
            fi
            checkpoint_name=MiniMax-M3.1-Flash-BF16
            moe_layer_freq="$(jq -cn '[range(60) | if . < 3 then 0 else 1 end]')"
            ;;
        *) echo "Set MINIMAX_PARITY_LAYERS to 4 or 60" >&2; exit 1 ;;
    esac
    TP=4
    if [[ "$SFT_MODE" == lora ]]; then
        EP=16
    fi
    GLOBAL_BATCH_SIZE=4
    MAX_TOKENS=9216
    if [[ -n "$MULTIMODAL_KEYS" ]]; then
        PARITY_NAME="minimax-m3.1-flash-${layer_count}l-${SFT_MODE}-vl-parity"
    elif [[ "$SFT_MODE" == lora ]]; then
        PARITY_NAME="minimax-m3.1-flash-${layer_count}l-text-parity"
    else
        PARITY_NAME="minimax-m3.1-flash-4l-full-text-parity"
    fi
fi
if [[ "${MINIMAX_PARITY:-0}" == 1 ]]; then
    : "${SFT_CHAT_TEMPLATE:?Prepare a verified parity template with scripts.tools.minimax_m31_sft_parity}"
    EXP_NAME="${EXP_NAME:-${PARITY_NAME}-${MINIMAX_MSA_IMPL:-relax}}"
    TRAIN_STEPS="${TRAIN_STEPS:-10}"
    if [[ "$SFT_MODE" == full || "$native_parity" == 1 ]]; then
        SAVE_INTERVAL="$TRAIN_STEPS"
    else
        KEEP_CHECKPOINTS="$TRAIN_STEPS"
    fi
    if [[ -e "${SAVE_DIR}/${EXP_NAME}/latest_checkpointed_iteration.txt" ||
          -d "${SAVE_DIR}/${EXP_NAME}/parity-state" ]]; then
        echo "Parity requires a fresh experiment name: ${EXP_NAME}" >&2
        exit 1
    fi
elif [[ -n "$MULTIMODAL_KEYS" ]]; then
    EXP_NAME="${EXP_NAME:-minimax-m3.1-flash-4l-vl-${SFT_MODE}-16xgpu}"
else
    EXP_NAME="${EXP_NAME:-minimax-m3.1-flash-4l-${SFT_MODE}-16xgpu}"
fi
TRAIN_STEPS="${TRAIN_STEPS:-2}"
now="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}}"
source "${SCRIPT_DIR}/../../models/minimax-m3.1-flash-4layer.sh" "$layer_count" "$moe_layer_freq"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/${checkpoint_name}"
   --ref-load "${MODEL_DIR}/${checkpoint_name}"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/${EXP_NAME}"
   --load "${SAVE_DIR}/${EXP_NAME}"
   --save-interval "$SAVE_INTERVAL"
   --max-actor-ckpt-to-keep "${KEEP_CHECKPOINTS:-2}"
)
LORA_ARGS=()
if [[ "$SFT_MODE" == lora ]]; then
    CKPT_ARGS+=(--save-lora-only)
    LORA_ARGS=(
        --lora-rank 16
        --lora-alpha 32
        --lora-target-modules linear_qkv linear_proj
        --lora-scope language
        --lora-dropout 0
        --lora-merge-mode
        --freeze-vision-model
        --freeze-vision-projection
    )
fi
SFT_ARGS=(
   --loss-type sft
   --prompt-data "${DATA_DIR}/minimax-m3.1-flash-sft.jsonl"
   --input-key messages
   --global-batch-size "$GLOBAL_BATCH_SIZE"
   --num-rollout "${TRAIN_STEPS}"
   --rollout-max-context-len "$MAX_TOKENS"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "$MAX_TOKENS"
   --calculate-per-token-loss
)
if [[ -n "$MULTIMODAL_KEYS" ]]; then
    SFT_ARGS+=(
        --multimodal-keys "$MULTIMODAL_KEYS"
    )
fi
if [[ -n "${SFT_CHAT_TEMPLATE:-}" ]]; then
    SFT_ARGS+=(
        --apply-chat-template-kwargs "$(jq -cn --rawfile chat_template "$SFT_CHAT_TEMPLATE" '{chat_template:$chat_template}')"
    )
fi
PARALLEL_ARGS=(
   --tensor-model-parallel-size "$TP"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size "$EP"
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
   --qkv-format thd
)
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "$LR"
   --lr-decay-style constant
   --use-checkpoint-opt-param-scheduler
   --weight-decay 0
   --clip-grad 1.0
)
MISC_ARGS=(
   --bf16
   --attention-backend flash
   --no-rope-fusion
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --use-health-check
)
if [[ "$native_parity" == 1 ]]; then
    SFT_ARGS+=(
        --tool-key tools
        --seq-length "$MAX_TOKENS"
    )
    if [[ -n "$MULTIMODAL_KEYS" ]]; then
        SFT_ARGS+=(--image-resize-scale-factor 0)
    fi
    MISC_ARGS+=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        --log-probs-chunk-size 256
    )
else
    MISC_ARGS+=(
        --sft-chunked-logits
        --sft-logits-chunk-size 256
    )
fi
if [[ "${MINIMAX_PARITY:-0}" == 1 ]]; then
    OPTIMIZER_ARGS+=(
        --seed "${SEED:-1234}"
        --adam-beta1 0.9
        --adam-beta2 0.999
        --moe-router-bias-update-rate 0
    )
    MISC_ARGS+=(
        --deterministic-mode
        --custom-megatron-init-path scripts.tools.minimax_m31_sft_parity.configure_worker
    )
    if [[ "$SFT_MODE" == full && "$native_parity" != 1 ]]; then
        MISC_ARGS+=(
            --custom-megatron-before-train-step-hook-path scripts.tools.minimax_m31_sft_parity_state.record_full_sft_step
        )
    fi
    RUNTIME_ENV_JSON=$(jq -c \
        --arg implementation "${MINIMAX_MSA_IMPL:-relax}" \
        --arg seed "${SEED:-1234}" \
        --arg text_parity "$native_parity" \
        '.env_vars += {
            "NCCL_ALGO": "Ring",
            "NCCL_PROTO": "Simple",
            "NCCL_NVLS_ENABLE": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "PYTHONHASHSEED": "0",
            "RELAX_PROCESSOR_POOL_SIZE": "4",
            "MINIMAX_MSA_IMPL": $implementation,
            "MINIMAX_PARITY_SEED": $seed,
            "MINIMAX_TEXT_PARITY": $text_parity
        }' <<< "$RUNTIME_ENV_JSON")
elif [[ "$SFT_MODE" == full && -n "$MULTIMODAL_KEYS" ]]; then
    RUNTIME_ENV_JSON=$(jq -c \
        --arg pool_size "${RELAX_PROCESSOR_POOL_SIZE:-4}" \
        '.env_vars.RELAX_PROCESSOR_POOL_SIZE = $pool_size' <<< "$RUNTIME_ENV_JSON")
fi
WORKING_DIR="${WORKING_DIR:-${RELAX_ROOT}}"
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --address="${RAY_ADDRESS}" \
   --submission-id="${SUBMISSION_ID}" \
   --entrypoint-num-cpus=1 \
   --working-dir="${WORKING_DIR}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft":[1,0],"actor":[1,16]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${PARALLEL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${SUBMISSION_ID}.log"
