#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5-35B-A3B OpenSearch SFT on 8 GPUs; TP1/EP8 defaults for Blackwell GPUs.
# MODEL_DIR=/models DATA_DIR=/data SAVE_DIR=/checkpoints bash "$0" relax
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_CONFIG=qwen35-35B-A3B
export EXP_NAME="${EXP_NAME:-qwen3.5-35b-a3b-opensearch-vl-32k-8xgpu}"
export TP_SIZE="${TP_SIZE:-1}"
export PP_SIZE="${PP_SIZE:-1}"
export EP_SIZE="${EP_SIZE:-8}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:?Set MODEL_DIR}/Qwen3.5-35B-A3B}"
exec bash "${SCRIPT_DIR}/run-qwen3.5-4B-8xgpu.sh" "$@"
