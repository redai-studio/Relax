#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 32 Blackwell GPUs, TP1/PP1/CP4/DP8/EP8, MoK MXFP8 + Muon + DoSA.
# Set MODEL_DIR, DATA_DIR (full128k-data/data.jsonl (author-masked messages)), SAVE_DIR and RAY_JOBS_ADDRESS.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MEGATRON="${MEGATRON:-/root/Megatron-LM}"
export EXP_NAME="${EXP_NAME:-qwen38-sft-128k-text-pp1}"
export QWEN38_SFT_SEQ_LENGTH=131072 QWEN38_SFT_MODALITY=text
export NUM_STEPS="${NUM_STEPS:-8}" GBS="${GBS:-64}"
export LR_DECAY_ITERS="${LR_DECAY_ITERS:-2023}" SEED="${SEED:-1234}"
exec bash "${SCRIPT_DIR}/run-qwen38-sft-32xgpu-muon.sh" "$@"
