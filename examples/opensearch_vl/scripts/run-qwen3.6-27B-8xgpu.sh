#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full-parameter Qwen3.6-27B SFT on 8 GPUs; shared data, metrics and final export.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_CONFIG=qwen36-27B
export EXP_NAME="${EXP_NAME:-qwen3.6-27b-opensearch-vl-32k-8xgpu}"
export TP_SIZE="${TP_SIZE:-4}"
export PP_SIZE="${PP_SIZE:-2}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:?Set MODEL_DIR}/Qwen3.6-27B}"
exec bash "${SCRIPT_DIR}/run-qwen3.5-4B-8xgpu.sh" "$@"
