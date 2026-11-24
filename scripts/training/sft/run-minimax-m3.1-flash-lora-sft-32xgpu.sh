#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Full 60-layer MiniMax-M3.1-Flash LoRA SFT on a running 32-GPU Ray cluster.
# Convert a complete official checkpoint to BF16:
# python scripts/tools/convert_minimax_m3_to_bf16.py --model-dir "$MODEL_DIR/MiniMax-M3.1-Flash" \
#   --save-dir "$MODEL_DIR/MiniMax-M3.1-Flash-BF16" --device cpu
# Provide "$DATA_DIR/minimax-m3.1-flash-sft.jsonl" with messages and optional tools.
# For image training, use <image> in messages, include corresponding paths in
# images, and set MULTIMODAL_KEYS='{"image":"images"}'.
# Use a fresh EXP_NAME for a new run.
# Launch through scripts/entrypoint/ray-job.sh.
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
CONTEXT_LENGTH="${SFT_CONTEXT_LENGTH:-65536}"
EXP_NAME="${EXP_NAME:-minimax-m3.1-flash-lora-sft-32xgpu}"
now="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}}"

source "${SCRIPT_DIR}/../../models/minimax-m3.1-flash.sh"
CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/MiniMax-M3.1-Flash-BF16"
   --ref-load "${MODEL_DIR}/MiniMax-M3.1-Flash-BF16"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/${EXP_NAME}"
   --load "${SAVE_DIR}/${EXP_NAME}"
   --save-interval 100
   --max-actor-ckpt-to-keep 10
   --save-lora-only
)
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
SFT_ARGS=(
   --loss-type sft
   --prompt-data "${DATA_DIR}/minimax-m3.1-flash-sft.jsonl"
   --input-key messages
   --tool-key tools
   --seq-length "$CONTEXT_LENGTH"
   --sft-oversize-strategy skip
   --sft-prefetch-buffer-size 16
   --sft-prefetch-chunk-size 8
   --sft-prefetch-num-workers 4
   --global-batch-size 8
   --num-epoch 1
   --rollout-max-context-len "$CONTEXT_LENGTH"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "$CONTEXT_LENGTH"
   --calculate-per-token-loss
)
if [[ -n "${MULTIMODAL_KEYS:-}" ]]; then
    SFT_ARGS+=(
        --multimodal-keys "$MULTIMODAL_KEYS"
        --image-resize-scale-factor 0
    )
fi
PARALLEL_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
   --qkv-format thd
)
OPTIMIZER_ARGS=(
   --optimizer adam
   --seed "${SEED:-1234}"
   --lr 1e-4
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
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --sft-chunked-logits
   --recompute-loss-function
   --empty-unused-memory-level 1
   --sft-logits-chunk-size 256
)

RUNTIME_ENV_JSON=$(jq -c '.env_vars.PYTORCH_ALLOC_CONF = "expandable_segments:True"' <<< "$RUNTIME_ENV_JSON")
WORKING_DIR="${WORKING_DIR:-${RELAX_ROOT}}"
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --address="${RAY_ADDRESS}" \
   --submission-id="${SUBMISSION_ID}" \
   --entrypoint-num-cpus=1 \
   --working-dir="${WORKING_DIR}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft":[1,0],"actor":[1,32]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${PARALLEL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${SUBMISSION_ID}.log"
