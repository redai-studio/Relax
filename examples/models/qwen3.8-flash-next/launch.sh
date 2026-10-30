#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Edit the defaults here or override them through the environment.
# Also sourced by the training script so both entrypoints share these defaults.
set -eo pipefail

# Paths. Model, data, and output paths must be supplied by the user.
QWEN38_LAUNCH_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_DIR="$(cd -- "${RELAX_DIR:-${QWEN38_LAUNCH_DIR}/../../..}" && pwd)"
export RELAX_DIR
export MODEL_DIR="${MODEL_DIR:-}"
export DATA_DIR="${DATA_DIR:-}"
export SAVE_DIR="${SAVE_DIR:-}"

# Experiment naming.
export EXP_NAME="${EXP_NAME:-qwen38-qsa-40xh20}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa}"

# Training. Parallel layout, freezing, and offload settings stay in scripts/.
export NUM_STEPS="${NUM_STEPS:-10}"
export GBS="${GBS:-1024}"
export SEQ_LENGTH="${SEQ_LENGTH:-65536}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-65536}"
export LR="${LR:-2e-5}"
export MIN_LR="${MIN_LR:-2e-6}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-2023}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"
export SEED="${SEED:-1234}"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exec bash "${RELAX_DIR}/examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-40xh20.sh" "$@"
fi
