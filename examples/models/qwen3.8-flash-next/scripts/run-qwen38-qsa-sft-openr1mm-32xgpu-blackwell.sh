#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# OpenR1MM SFT with MoK on 4 nodes x 8 Blackwell GPUs; TP2/PP4/CP1/EP8/ETP1.
# DATA_DIR is prepared SFT JSONL with messages/images, not the raw GRPO parquet.
# Set MODEL_DIR, DATA_DIR, SAVE_DIR and MEGATRON to shared paths on all workers.
# MOE_BACKEND=hybridep selects the reference backend with the same workload.
# DRY_RUN=1 prints the effective configuration without starting Ray or using GPUs.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MOE_BACKEND="${MOE_BACKEND:-mok}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-${SEQ_LENGTH}}"
export GBS="${GBS:-128}"
export EXP_NAME="${EXP_NAME:-qwen38-openr1mm-sft-${MOE_BACKEND}}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-openr1mm}"

exec bash "${SCRIPT_DIR}/run-qwen38-qsa-sft-256k-32xgpu-blackwell.sh" "$@"
