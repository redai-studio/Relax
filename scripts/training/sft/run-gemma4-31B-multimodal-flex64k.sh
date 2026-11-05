#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Packed image SFT on the shared TP8/PP1/CP1 64K profile.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export GEMMA4_SFT_CONVERSION_MODE=vl
export GEMMA4_SFT_THINKING="${GEMMA4_SFT_THINKING:-0}"
export INPUT_KEY="${INPUT_KEY:-messages}"
export MULTIMODAL_KEYS='{"image":"images"}'
export SAVE_HF_POST_HOOK_PATH="${SAVE_HF_POST_HOOK_PATH-scripts.tools.gemma4_chat_template_hook.patch_gemma4_chat_template}"
export EXP_NAME="${EXP_NAME:-gemma4-31b-multimodal-flex64k}"
exec bash "${SCRIPT_DIR}/run-gemma4-31B-sft-64k-8xgpu.sh" "$@"
