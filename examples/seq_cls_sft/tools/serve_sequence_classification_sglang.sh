#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

EXP_DIR="${EXP_DIR:-${RELAX_ROOT}/../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
MODEL_VARIANT="${MODEL_VARIANT:-9B}"
if [ "${MODEL_VARIANT}" = "9B" ]; then
    CLASSIFICATION_DATASET="${CLASSIFICATION_DATASET:-sst2}"
    EXP_NAME="${EXP_NAME:-qwen3.5-9b-${CLASSIFICATION_DATASET}-seq-cls-gpu8}"
elif [ "${MODEL_VARIANT}" = "35B-A3B" ]; then
    CLASSIFICATION_DATASET="${CLASSIFICATION_DATASET:-go_emotions}"
    EXP_NAME="${EXP_NAME:-qwen3.5-35b-a3b-${CLASSIFICATION_DATASET}-seq-cls-lora-gpu8}"
else
    echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}; expected 9B or 35B-A3B" >&2
    exit 2
fi
CLASSIFICATION_MODEL_DIR="${CLASSIFICATION_MODEL_DIR:-${MODEL_DIR}/${EXP_NAME}-sglang}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-seq-cls}"
TP_SIZE="${TP_SIZE:-4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"

export PYTHONPATH="${RELAX_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_EXTERNAL_MODEL_PACKAGE="examples.seq_cls_sft.models.sglang"
export SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE="examples.seq_cls_sft.models.sglang"

exec python "${SCRIPT_DIR}/launch_sglang_text_classification.py" \
    --model-path "${CLASSIFICATION_MODEL_DIR}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tp "${TP_SIZE}" \
    --is-embedding \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
