#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# 48-layer Kimi SFT on 8 x 8 L20Y GPUs. Supply a converted BF16 checkpoint and a Kimi messages JSONL.
# From an existing Ray head, set RAY_ADDRESS to its GCS address and RAY_DASHBOARD_ADDRESS to its dashboard URL.
# Usage: bash run-mimo-v2.6-flash-kimi-sft-48layer-64xgpu.sh BF16_CHECKPOINT MESSAGES_JSONL SAVE_DIR [train args...]
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint=${1:?provide the full BF16 checkpoint directory}
data=${2:?provide the Kimi JSONL}
output=${3:?provide the checkpoint output directory}
shift 3

export EXPECTED_LAYERS=48 SFT_INPUT=image
export NODES=8 GPUS_PER_NODE=8 TP=2 PP=8 CP=2 EP=8
export SEQ_LENGTH=131072 MAX_TOKENS_PER_GPU=65536 GLOBAL_BATCH_SIZE=64
export STEPS=${STEPS:-1731} SAVE_INTERVAL=${SAVE_INTERVAL:-100} SFT_MAX_IN_FLIGHT_STEPS=2
export DATA_STORAGE_UNITS=8 LR_DECAY_STYLE=cosine WEIGHT_DECAY=0.1 ADAM_BETA2=0.95

# PP0 owns the vision tower. Move one decoder layer from PP0 to PP5.
exec bash "${SCRIPT_DIR}/run-mimo-v2.6-flash-vision-sft.sh" "$checkpoint" "$data" "$output/native" \
    --pipeline-model-parallel-layout 'Et*5|t*6|t*6|t*6|t*6|t*7|t*6|t*6L' --cp-comm-type a2a \
    --freeze-vision-model --freeze-vision-projection --calculate-per-token-loss --sft-loss-last-turn-only \
    --seed 1234 --rollout-shuffle --apply-chat-template-kwargs '{"thinking_mode":"thinking","preserve_thinking":true}' \
    --sft-invalid-multimodal-strategy skip --sft-prefetch-num-workers 16 --sft-prefetch-buffer-size 64 \
    --sft-tq-timeout-minutes 90 --min-lr 1e-6 --lr-warmup-fraction 0.05 --adam-eps 1e-8 --clip-grad 1.0 \
    --sft-logits-chunk-size 1024 --log-probs-chunk-size 4096 --moe-permute-fusion \
    --overlap-cpu-optimizer-d2h-h2d --empty-unused-memory-level 1 \
    --train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True,garbage_collection_threshold:0.8"}' \
    --max-actor-ckpt-to-keep 2 --save-hf "$output/final-hf-{rollout_id}" --save-hf-final-only "$@"
