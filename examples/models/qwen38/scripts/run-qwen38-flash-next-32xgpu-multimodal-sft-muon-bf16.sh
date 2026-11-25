#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# BF16 AdamW reference for the Qwen3.8-Flash-Next MXFP8 Muon multimodal SFT recipe.
# Use the same BF16 HF checkpoint, data and CLI overrides for both precision runs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PRECISION=bf16
exec bash "${SCRIPT_DIR}/run-qwen38-flash-next-32xgpu-multimodal-sft-muon-fp8.sh" "$@"
