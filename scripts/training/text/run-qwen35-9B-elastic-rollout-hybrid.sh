#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eo pipefail

# Hybrid runs actor/reference forward on eight GPUs (TP4, DP2).
# Rollout starts with two engines on a second eight-GPU worker and can grow to eight.
export RELAX_ELASTIC_TRAIN_MODE=hybrid
export PROJECT_NAME="${PROJECT_NAME:-Relax/CE/run-qwen35-9B-elastic-rollout-hybrid}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/run-qwen35-9B-elastic-rollout-async.sh"
