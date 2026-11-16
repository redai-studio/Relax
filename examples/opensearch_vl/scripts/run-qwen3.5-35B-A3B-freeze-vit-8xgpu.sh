#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5-35B-A3B OpenSearch SFT: freeze the vision trunk; train merger and language.
# MODEL_DIR=/models DATA_DIR=/data SAVE_DIR=/checkpoints bash "$0" relax
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FREEZE_VISION_MODEL=1
export EXP_NAME="${EXP_NAME:-qwen3.5-35b-a3b-opensearch-vl-32k-8xgpu-freeze-vit}"
exec bash "${SCRIPT_DIR}/run-qwen3.5-35B-A3B-8xgpu.sh" "$@"
