#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Raw-text CPT with the shared 32-GPU Qwen3.8 recipe and complete 256k documents.
# MODEL_DIR, DATA_DIR and SAVE_DIR are required, as in the SFT recipe.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SFT_TRAINING_MODE=cpt
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-262144}"
export SFT_TQ_TIMEOUT_MINUTES="${SFT_TQ_TIMEOUT_MINUTES:-360}"
DEFAULT_EXP_NAME=qwen38-qsa-cpt-256k-32xb300 # gitleaks:allow -- preserve existing experiment/checkpoint identity
if [[ "${MOE_BACKEND:-hybridep}" == mok ]]; then DEFAULT_EXP_NAME+=-mok; fi
export EXP_NAME="${EXP_NAME:-${DEFAULT_EXP_NAME}}"
exec bash "${SCRIPT_DIR}/run-qwen38-qsa-sft-256k-32xgpu-blackwell.sh" "$@"
