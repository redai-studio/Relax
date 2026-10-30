#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Pure-text Flash/PR1765 SFT with the measured balanced pipeline partition.
# MODEL_DIR=... DATA_DIR=... SAVE_DIR=... MEGATRON=... bash <this-script>
# Data must contain text messages/tool calls, without structured image inputs.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QWEN38_SFT_MODALITY=text
# Three backbone QSA layers per stage; PP3 also owns MTP and the output head.
export QWEN38_PP_LAYOUT='Et*14|t*12|t*12|t*10mL'
export LR_DECAY_ITERS=996
export EVAL_INTERVAL=9223372036854775807
export EXP_NAME="${EXP_NAME:-qwen38-qsa-nv-text-32xgpu}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen38-qsa-nv-text}"

exec bash "${SCRIPT_DIR}/run-qwen38-qsa-nv-sft-32xgpu-blackwell.sh" "$@"
