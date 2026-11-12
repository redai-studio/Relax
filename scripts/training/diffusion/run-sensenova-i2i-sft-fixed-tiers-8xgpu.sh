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
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$GPUS}"
if (( GPUS <= 0 || GPUS_PER_NODE <= 0 || GLOBAL_BATCH_SIZE <= 0 || GPUS % GPUS_PER_NODE != 0 || GLOBAL_BATCH_SIZE % GPUS != 0 )); then
    echo "GPUS must be a positive multiple of GPUS_PER_NODE; GLOBAL_BATCH_SIZE must be divisible by GPUS" >&2
    exit 1
fi
ACTOR_NODES=$((GPUS / GPUS_PER_NODE))
STEPS="${STEPS:-5}"
SEED="${SEED:-42}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"
# FIXED_PIXELS_TIERS: 逗号分隔的固定像素档位列表 (如 "2073600,4194304" = 1440²/2048² 均匀随机采样;
# 三张图同档, 逐样本确定性抽样, resume 可复现). 留空走 max_pixels 只缩不放模式.
FIXED_PIXELS_TIERS="${FIXED_PIXELS_TIERS:-}"
CONDITIONING_IMAGES="${CONDITIONING_IMAGES:-1}"
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-flash_attention_2}"
DETERMINISTIC_ATTENTION="${DETERMINISTIC_ATTENTION:-0}"
PREFIX_GRADIENT_CHECKPOINTING="${PREFIX_GRADIENT_CHECKPOINTING:-1}"
# Opt in after validating host-memory headroom; zero preserves synchronous decoding.
IMAGE_PREFETCH_DEPTH="${IMAGE_PREFETCH_DEPTH:-0}"
IMAGE_PREFETCH_WORKERS="${IMAGE_PREFETCH_WORKERS:-2}"
IMAGE_DP_TIER_MICRO_BALANCE="${IMAGE_DP_TIER_MICRO_BALANCE:-0}"
case "$IMAGE_DP_TIER_MICRO_BALANCE" in
    0|1) ;;
    *) echo "IMAGE_DP_TIER_MICRO_BALANCE must be 0 or 1" >&2; exit 1 ;;
esac
IMAGE_DP_TIER_RANK_ROTATION="${IMAGE_DP_TIER_RANK_ROTATION:-0}"
case "$IMAGE_DP_TIER_RANK_ROTATION" in
    0|1) ;;
    *) echo "IMAGE_DP_TIER_RANK_ROTATION must be 0 or 1" >&2; exit 1 ;;
esac
if [[ "$IMAGE_DP_TIER_RANK_ROTATION" == 1 && "$IMAGE_DP_TIER_MICRO_BALANCE" != 1 ]]; then
    echo "IMAGE_DP_TIER_RANK_ROTATION requires IMAGE_DP_TIER_MICRO_BALANCE=1" >&2
    exit 1
fi
export IMAGE_PREFETCH_DEPTH IMAGE_PREFETCH_WORKERS IMAGE_DP_TIER_MICRO_BALANCE IMAGE_DP_TIER_RANK_ROTATION
if [[ "$MODE" == full ]]; then
    LORA_RANK=0
    LR="${LR:-1e-6}"
else
    LORA_RANK="${LORA_RANK:-32}"
    LR="${LR:-1e-4}"
fi
I2I_CONFIG=$(python3 - "$DATA_ROOT" "$DATASET_SIZE" "$MAX_PIXELS" "$CONDITIONING_IMAGES" \
    "$ATTENTION_IMPLEMENTATION" "$DETERMINISTIC_ATTENTION" "$PREFIX_GRADIENT_CHECKPOINTING" \
    "$FIXED_PIXELS_TIERS" "$SEED" <<'PY'
import json
import os
import sys
tiers = [int(t.strip()) for t in sys.argv[8].split(",") if t.strip()] if len(sys.argv) > 8 and sys.argv[8].strip() else None
config = {
    "data_root": sys.argv[1],
    "dataset_size": int(sys.argv[2]),
    "max_pixels": int(sys.argv[3]),
    "conditioning_images": int(sys.argv[4]),
    "attention_implementation": sys.argv[5],
    "deterministic_attention": sys.argv[6] == "1",
    "prefix_gradient_checkpointing": sys.argv[7] == "1",
    "image_prefetch_depth": int(os.environ["IMAGE_PREFETCH_DEPTH"]),
    "image_prefetch_workers": int(os.environ["IMAGE_PREFETCH_WORKERS"]),
    "fixed_pixels_tiers": tiers,
    "seed": int(sys.argv[9]),
}
if os.environ["IMAGE_DP_TIER_MICRO_BALANCE"] == "1":
    config["dp_tier_micro_balance"] = True
if os.environ["IMAGE_DP_TIER_RANK_ROTATION"] == "1":
    config["dp_tier_rank_rotation"] = True
print(json.dumps(config))
PY
)
EXTRA_ARGS=()
if [[ -n "${LOAD_DIR:-}" ]]; then EXTRA_ARGS+=(--load "$LOAD_DIR"); fi
if [[ -n "${SAVE_DIR:-}" ]]; then
    EXTRA_ARGS+=(--save "$SAVE_DIR" --save-interval "${SAVE_INTERVAL:-5}")
fi
SUBMIT_ARGS=(--no-wait --address "${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}")
if [[ -n "${JOB_ID:-}" ]]; then SUBMIT_ARGS+=(--submission-id "$JOB_ID"); fi
if [[ -n "${WORKING_DIR:-}" ]]; then SUBMIT_ARGS+=(--working-dir "$WORKING_DIR"); fi
DRIVER_ENV=()
# Set external import paths on the driver, not in submission runtime_env: train
# merges PYTHONPATH into its own actor runtime_env and Ray rejects conflicting keys.
if [[ -n "${DRIVER_PYTHONPATH:-}" ]]; then DRIVER_ENV=(env "PYTHONPATH=$DRIVER_PYTHONPATH"); fi
ray job submit "${SUBMIT_ARGS[@]}" --runtime-env-json "${RUNTIME_ENV_JSON:-"{}"}" -- \
    "${DRIVER_ENV[@]}" python3 -m relax.entrypoints.train \
    --train-backend fsdp --loss-type sft --generation-task i2i \
    --model-path "$MODEL_PATH" --prompt-data "$METADATA_PATH" --i2i-sft-config "$I2I_CONFIG" \
    --resource "{\"actor\": [1, $GPUS], \"sft\": [1, 0]}" \
    --actor-num-nodes "$ACTOR_NODES" --actor-num-gpus-per-node "$GPUS_PER_NODE" \
    --num-gpus-per-node "$GPUS_PER_NODE" \
    --fsdp-trainable-mode "$MODE" --fsdp-master-dtype fp32 \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_RANK" \
    --micro-batch-size 1 --global-batch-size "$GLOBAL_BATCH_SIZE" --rollout-batch-size "$GLOBAL_BATCH_SIZE" \
    --n-samples-per-prompt 1 --num-rollout "$STEPS" --seed "$SEED" \
    --lr "$LR" --lr-warmup-iters 200 --weight-decay 0.01 \
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1.0 \
    "${EXTRA_ARGS[@]}" "$@"
