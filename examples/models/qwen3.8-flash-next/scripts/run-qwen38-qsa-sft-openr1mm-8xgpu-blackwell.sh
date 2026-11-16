#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Prepare with scripts/tools/process_openr1.py --mode sft; DATA_DIR is its output directory.
# MODEL_DIR, SAVE_DIR and MEGATRON have the same meaning as the base 8-GPU recipe.
# MOE_BACKEND=hybridep or mok; use identical batch/token settings for comparisons.
# Process more short image/question/solution samples per step; allow explicit overrides.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_DIR:?Set DATA_DIR to the prepared OpenR1MM SFT directory}"
export DATA_DIR="${DATA_DIR}/sft.jsonl"
if [[ "${DRY_RUN:-0}" != 1 && ! -f "${DATA_DIR}" ]]; then
    echo "Prepared OpenR1MM sft.jsonl does not exist: ${DATA_DIR}" >&2
    exit 2
fi
export EXP_NAME="${EXP_NAME:-qwen38-qsa-openr1mm-8xb300-${MOE_BACKEND:-hybridep}}" # gitleaks:allow -- preserve existing experiment/checkpoint identity
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-openr1mm-8xgpu}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
exec bash "${SCRIPT_DIR}/run-qwen38-qsa-sft-8xgpu-blackwell.sh" "$@"
