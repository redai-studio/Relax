#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Kimi K3.1 LoRA SFT; shares the configurable full-SFT submission path.
set -eo pipefail
export KIMI_LORA=1
export RELAX_LORA_SHARE_EXPERT_ADAPTERS=false
export NUM_GPUS="${NUM_GPUS:-64}"
export CP_SIZE="${CP_SIZE:-4}"
export EP_SIZE="${EP_SIZE:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export NUM_ROLLOUT="${NUM_ROLLOUT-${NUM_STEPS-200}}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export OPTIMIZER_OFFLOAD_FRACTION="${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"
export EXP_NAME="${EXP_NAME:-kimi-k3.1-pro-lora}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-fused}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
exec bash "${SCRIPT_DIR}/run-kimi-k3.1-pro-sft.sh" "$@"
