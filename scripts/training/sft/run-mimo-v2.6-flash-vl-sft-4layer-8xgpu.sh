#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Four-layer image/text SFT. JSONL rows contain messages, optional tools and images.
# Pass the BF16 checkpoint, JSONL and output directory as the first three arguments.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SEQ_LENGTH=${SEQ_LENGTH:-40960}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-$SEQ_LENGTH}
exec bash "${SCRIPT_DIR}/run-mimo-v2.6-flash-sft-4layer-8xgpu.sh" "${1:?provide the BF16 checkpoint}" \
    "${2:?provide the image messages dataset}" "${3:?provide the output directory}" \
    --multimodal-keys '{"image":"images"}' --tool-key tools \
    --image-max-token-num 1024 --sft-oversize-strategy skip \
    --sft-chunked-logits --recompute-loss-function \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "${@:4}"
