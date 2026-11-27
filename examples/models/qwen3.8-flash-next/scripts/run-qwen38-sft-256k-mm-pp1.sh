#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 32 Blackwell GPUs, TP1/PP1/CP8/DP4/EP8, MoK MXFP8 + Muon + DoSA.
# Set MODEL_DIR, DATA_DIR (10_multimodal_256k.jsonl), SAVE_DIR and RAY_JOBS_ADDRESS.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MEGATRON="${MEGATRON:-/root/Megatron-LM}"
export EXP_NAME="${EXP_NAME:-qwen38-sft-256k-mm-pp1}"
export QWEN38_SFT_SEQ_LENGTH=262144 QWEN38_SFT_MODALITY=multimodal
export NUM_STEPS="${NUM_STEPS:-50}" GBS="${GBS:-32}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-200}" SEED="${SEED:-9714}"
exec bash "${SCRIPT_DIR}/run-qwen38-sft-32xgpu-muon.sh" "$@"
