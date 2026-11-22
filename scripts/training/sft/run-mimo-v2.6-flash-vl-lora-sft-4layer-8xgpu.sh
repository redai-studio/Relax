#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Four-layer, 8-GPU image/text LoRA SFT. Arguments: BF16_CHECKPOINT MESSAGES_JSONL SAVE_DIR [train args...]
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-$SEQ_LENGTH}

# Use the regular LM-head forward so both the vocabulary projection and its adapter contribute to the loss.
exec bash "${SCRIPT_DIR}/run-mimo-v2.6-flash-sft-4layer-8xgpu.sh" "${1:?provide the BF16 checkpoint}" \
    "${2:?provide the image messages dataset}" "${3:?provide the output directory}" \
    --multimodal-keys '{"image":"images"}' --tool-key tools \
    --image-max-token-num 1024 --sft-oversize-strategy skip \
    --freeze-vision-model --freeze-vision-projection \
    --lora-scope language --lora-rank "${LORA_RANK:-32}" --lora-alpha "${LORA_ALPHA:-64}" --lora-dropout 0 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 output_layer \
    --train-env-vars '{"RELAX_LORA_SHARE_EXPERT_ADAPTERS":"false"}' \
    --lora-merge-mode --no-gradient-accumulation-fusion \
    --save-lora-only --save-hf "${3}/hf-{rollout_id}" "${@:4}"
