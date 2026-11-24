#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# OpenSearch-VL full SFT on 8 x 8 GPUs. Supply a converted 48-layer BF16 checkpoint
# and a messages JSONL with images and optional tools.
# Usage: bash run-mimo-v2.6-flash-opensearch-vl-sft-48layer-64xgpu.sh BF16_CHECKPOINT MESSAGES_JSONL SAVE_DIR [train args...]
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint=${1:?provide the full BF16 checkpoint directory}
data=${2:?provide the OpenSearch-VL JSONL}
output=${3:?provide the checkpoint output directory}
shift 3

export EXPECTED_LAYERS=48 SFT_INPUT=image
export NODES=${NODES:-8} GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export TP=${TP:-4} PP=${PP:-8} EP=${EP:-8}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128} STEPS=${STEPS:-50}
export SEQ_LENGTH=${SEQ_LENGTH:-40960} SFT_MAX_IN_FLIGHT_STEPS=2
export SAVE_INTERVAL=${SAVE_INTERVAL:-$STEPS}

exec bash "${SCRIPT_DIR}/run-mimo-v2.6-flash-vision-sft.sh" "$checkpoint" "$data" "$output/native" \
    --save-hf "$output/final-hf-{rollout_id}" --seed 1234 \
    --apply-chat-template-kwargs '{"thinking_mode":"thinking"}' --sft-tq-timeout-minutes 90 "$@"
