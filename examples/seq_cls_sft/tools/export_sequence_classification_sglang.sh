#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

EXP_DIR="${EXP_DIR:-${RELAX_ROOT}/../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
MODEL_VARIANT="${MODEL_VARIANT:-9B}"

case "${MODEL_VARIANT}" in
    9B)
        CLASSIFICATION_DATASET="${CLASSIFICATION_DATASET:-sst2}"
        EXP_NAME="${EXP_NAME:-qwen3.5-9b-${CLASSIFICATION_DATASET}-seq-cls-gpu8}"
        ORIGIN_MODEL_DIR="${ORIGIN_MODEL_DIR:-${MODEL_DIR}/Qwen3.5-9B}"
        SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/qwen3.5-9B-classification-sft}"
        ;;
    35B-A3B)
        CLASSIFICATION_DATASET="${CLASSIFICATION_DATASET:-go_emotions}"
        EXP_NAME="${EXP_NAME:-qwen3.5-35b-a3b-${CLASSIFICATION_DATASET}-seq-cls-lora-gpu8}"
        ORIGIN_MODEL_DIR="${ORIGIN_MODEL_DIR:-${MODEL_DIR}/Qwen3.5-35B-A3B}"
        SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/qwen3.5-35B-A3B-classification-lora-sft}"
        ;;
    *)
        echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}; expected 9B or 35B-A3B" >&2
        exit 2
        ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SAVE_DIR}/${EXP_NAME}}"
CLASSIFICATION_MODEL_DIR="${CLASSIFICATION_MODEL_DIR:-${MODEL_DIR}/${EXP_NAME}-sglang}"

exec python "${SCRIPT_DIR}/convert_sequence_classification_checkpoint.py" \
    --input-dir "${CHECKPOINT_DIR}" \
    --origin-hf-dir "${ORIGIN_MODEL_DIR}" \
    --output-dir "${CLASSIFICATION_MODEL_DIR}" \
    "$@"
