#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3-VL-8B-Instruct SFT and V*Bench entry point on 8 GPUs.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_CONFIG=qwen3-vl-8B
export EXP_NAME="${EXP_NAME:-qwen3-vl-8b-opensearch-vl-32k-8xgpu}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-1}"
export RECOMPUTE="${RECOMPUTE:-full}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:?Set MODEL_DIR}/Qwen3-VL-8B-Instruct}"
exec bash "${SCRIPT_DIR}/run-qwen3.5-4B-8xgpu.sh" "$@"
