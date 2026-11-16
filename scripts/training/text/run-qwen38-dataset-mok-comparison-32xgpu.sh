#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Per-dataset settings are supplied by the recorded comparison manifest.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${MOE_BACKEND:?Set hybridep or mok}"
: "${SFT_TRAINING_MODE:?Set sft or cpt for the dataset}"
: "${SEQ_LENGTH:?Set the full valid-sample limit}"
: "${MAX_TOKENS_PER_GPU:?Set the recorded dataset packing budget}"
: "${DATASET_MULTIMODAL:?Set 0 for text or 1 for real multimodal inputs}"
export IMAGE_MAX_TOKENS=16384
case "${DATASET_MULTIMODAL}" in
    0) DATASET_MEDIA_KEYS='{}' ;;
    1) DATASET_MEDIA_KEYS='{"image":"images"}' ;;
    *) echo "DATASET_MULTIMODAL must be 0 or 1" >&2; exit 2 ;;
esac

exec bash "${REPO_ROOT}/examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-sft-256k-32xgpu-blackwell.sh" \
    --multimodal-keys "${DATASET_MEDIA_KEYS}" "$@"
