#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Multiturn profile of the existing Gemma4 64K toolcall delivery launcher.
# Keep its native Flex attention, topology options, and checkpoint/HF export.
# Launch through scripts/entrypoint/ray-job.sh with RAY_NO_WAIT=1.
# Required: CKPT and PROMPT_DATA. SAVE_DIR/SAVE_HF_DIR follow the base launcher.
# Current gemma-4-31B-it template was validated with THINKING=0 only.
# Validate 64K stability and numerical parity on the target runtime before production use.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export RAY_NO_WAIT=1
export GEMMA4_SFT_THINKING="${GEMMA4_SFT_THINKING:-0}"
export INPUT_KEY=messages LABEL_KEY=""
export EXP_NAME="${EXP_NAME:-gemma4-31b-multiturn-flex64k}"
exec bash "${SCRIPT_DIR}/run-gemma4-31B-sft-64k-8xgpu.sh"
