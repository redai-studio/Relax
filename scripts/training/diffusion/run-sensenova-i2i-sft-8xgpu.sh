#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Invoke through scripts/entrypoint/ray-job.sh on an existing Ray cluster.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the worker-visible SenseNova model directory}"
: "${DATA_ROOT:?Set DATA_ROOT to the image root on training workers}"
: "${METADATA_PATH:?Set METADATA_PATH to paired image JSON metadata}"
: "${DATASET_SIZE:?Set DATASET_SIZE to the exact number of metadata rows}"
MODE="${MODE:-lora}"
GPUS="${GPUS:-8}"
STEPS="${STEPS:-5}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
CONDITIONING_IMAGES="${CONDITIONING_IMAGES:-1}"
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-flash_attention_2}"
DETERMINISTIC_ATTENTION="${DETERMINISTIC_ATTENTION:-0}"
PREFIX_GRADIENT_CHECKPOINTING="${PREFIX_GRADIENT_CHECKPOINTING:-1}"
# Opt in after validating host-memory headroom; zero preserves synchronous decoding.
IMAGE_PREFETCH_DEPTH="${IMAGE_PREFETCH_DEPTH:-0}"
IMAGE_PREFETCH_WORKERS="${IMAGE_PREFETCH_WORKERS:-2}"
export IMAGE_PREFETCH_DEPTH IMAGE_PREFETCH_WORKERS
if [[ "$MODE" == full ]]; then
    LORA_RANK=0
    LR="${LR:-1e-6}"
else
    LORA_RANK="${LORA_RANK:-32}"
    LR="${LR:-1e-4}"
fi
I2I_CONFIG=$(python3 - "$DATA_ROOT" "$DATASET_SIZE" "$MAX_PIXELS" "$CONDITIONING_IMAGES" \
    "$ATTENTION_IMPLEMENTATION" "$DETERMINISTIC_ATTENTION" "$PREFIX_GRADIENT_CHECKPOINTING" <<'PY'
import json
import os
import sys
print(json.dumps({
    "data_root": sys.argv[1],
    "dataset_size": int(sys.argv[2]),
    "max_pixels": int(sys.argv[3]),
    "conditioning_images": int(sys.argv[4]),
    "attention_implementation": sys.argv[5],
    "deterministic_attention": sys.argv[6] == "1",
    "prefix_gradient_checkpointing": sys.argv[7] == "1",
    "image_prefetch_depth": int(os.environ["IMAGE_PREFETCH_DEPTH"]),
    "image_prefetch_workers": int(os.environ["IMAGE_PREFETCH_WORKERS"]),
}))
PY
)
EXTRA_ARGS=()
if [[ -n "${LOAD_DIR:-}" ]]; then EXTRA_ARGS+=(--load "$LOAD_DIR"); fi
if [[ -n "${SAVE_DIR:-}" ]]; then
    EXTRA_ARGS+=(--save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-5}")
fi
SUBMIT_ARGS=(--no-wait --address "${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}")
if [[ -n "${WORKING_DIR:-}" ]]; then SUBMIT_ARGS+=(--working-dir "$WORKING_DIR"); fi
ray job submit "${SUBMIT_ARGS[@]}" --runtime-env-json "${RUNTIME_ENV_JSON:-"{}"}" -- \
    python3 -m relax.entrypoints.train \
    --train-backend fsdp --loss-type sft --generation-task i2i \
    --model-path "$MODEL_PATH" --prompt-data "$METADATA_PATH" --i2i-sft-config "$I2I_CONFIG" \
    --resource "{\"actor\": [1, $GPUS], \"sft\": [1, 0]}" \
    --fsdp-trainable-mode "$MODE" --fsdp-master-dtype fp32 \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_RANK" \
    --micro-batch-size 1 --global-batch-size "$GPUS" --rollout-batch-size "$GPUS" \
    --n-samples-per-prompt 1 --num-rollout "$STEPS" --seed 42 \
    --lr "$LR" --lr-warmup-iters 200 --weight-decay 0.01 \
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1.0 \
    "${EXTRA_ARGS[@]}" "$@"
