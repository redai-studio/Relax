#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# BF16 AdamW reference for the Qwen3.8-Flash-Next MXFP8 Muon multimodal SFT recipe.
# Uses grouped QSA, the torch selector, unfused TE grouped GEMMs and the image's default cuBLAS.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PRECISION=bf16
exec bash "${SCRIPT_DIR}/run-qwen38-flash-next-32xgpu-multimodal-sft-muon-fp8.sh" "$@"
