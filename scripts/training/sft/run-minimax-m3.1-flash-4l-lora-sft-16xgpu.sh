#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Two-node MiniMax-M3.1-Flash LoRA SFT on the four-layer BF16 checkpoint.
# Prepare it from a local copy of the official checkpoint:
# python scripts/tools/convert_minimax_m3_to_bf16.py --model-dir "$MODEL_DIR/MiniMax-M3.1-Flash" \
#   --layers 0,1,3,4 --save-dir "$MODEL_DIR/MiniMax-M3.1-Flash-4L-BF16" --device cpu
# Provide "$DATA_DIR/minimax-m3.1-flash-sft.jsonl" with text-only messages.
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
EXP_NAME="${EXP_NAME:-minimax-m3.1-flash-4l-lora-16xgpu}"
TRAIN_STEPS="${TRAIN_STEPS:-2}"
now="$(date +%Y%m%d-%H%M%S)"
SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}}"
source "${SCRIPT_DIR}/../../models/minimax-m3.1-flash-4layer.sh"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/MiniMax-M3.1-Flash-4L-BF16"
   --ref-load "${MODEL_DIR}/MiniMax-M3.1-Flash-4L-BF16"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/${EXP_NAME}"
   --load "${SAVE_DIR}/${EXP_NAME}"
   --save-lora-only
   --save-interval 1
   --max-actor-ckpt-to-keep 2
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
   --global-batch-size 16
   --num-rollout "${TRAIN_STEPS}"
   --rollout-max-context-len 4096
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --calculate-per-token-loss
)
PARALLEL_ARGS=(
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --moe-token-dispatcher-type alltoall
   --moe-grouped-gemm
   --qkv-format thd
)
OPTIMIZER_ARGS=(
   --optimizer adam
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
   --sft-chunked-logits
   --sft-logits-chunk-size 256
   --use-health-check
)
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
