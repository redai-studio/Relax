#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Multimodal Flash/PR1765 SFT: use the existing 48-layer TP2/PP4 recipe.
# Requires the pinned cuDNN QSA package; see docs/en/guide/qwen38-qsa.md.
# MODEL_DIR=... DATA_DIR=... SAVE_DIR=... MEGATRON=... bash <this-script>
# DRY_RUN=1 prints effective configuration without Ray or GPU imports.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXP_NAME="${EXP_NAME:-qwen38-qsa-nv-32xgpu}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa-nv}"
export RELAX_QWEN38_QSA_MIXED_LONG_BACKEND="${RELAX_QWEN38_QSA_MIXED_LONG_BACKEND:-cudnn_pr1765}"
export RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=2051
export RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS=0
export NVTE_USE_CUTLASS_GROUPED_GEMM="${NVTE_USE_CUTLASS_GROUPED_GEMM:-0}"
export QWEN38_CUDA_DEVICE_MAX_CONNECTIONS=1
export RELAX_QWEN38_RECORD_FUNCTION_TAGS=0
export RELAX_AUTOTUNE_LOG_DIR=
export RELAX_SKIP_NONFINITE_GRAD="${RELAX_SKIP_NONFINITE_GRAD:-0}"
export QWEN38_SFT_HEALTH_CHECK=0

exec bash "${SCRIPT_DIR}/run-qwen38-qsa-sft-32xgpu-blackwell.sh" \
    --max-global-restart 0 --sft-tq-timeout-minutes 120 "$@"
