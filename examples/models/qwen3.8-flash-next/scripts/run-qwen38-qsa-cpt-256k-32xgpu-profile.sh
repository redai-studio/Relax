#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Two warmup updates, followed by one profiled update, with the original LR schedule.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${PROFILE_OUTPUT_DIR:?Set PROFILE_OUTPUT_DIR to a shared trace directory}"
export GBS="${GBS:-256}"
export SAVE_CHECKPOINTS=0
export NUM_EPOCH="${NUM_EPOCH:-1}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-262144}"
export SEQ_LENGTH="${SEQ_LENGTH:-262144}"

exec bash "${SCRIPT_DIR}/run-qwen38-qsa-cpt-256k-32xgpu-blackwell.sh" \
    --num-rollout 3 \
    --lr-decay-iters 10 \
    --empty-unused-memory-level 1 \
    --use-pytorch-profiler \
    --profile-target train_overall \
    --profile-step-start 2 \
    --profile-step-end 2 \
    --profile-ranks 0 8 16 24 \
    --profile-output-dir "${PROFILE_OUTPUT_DIR}" \
    "$@"
