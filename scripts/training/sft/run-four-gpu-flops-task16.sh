#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail
export TASK16_PROFILE=four
TASK16_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${TASK16_SCRIPT_DIR}/run-qwen35-flops-task16.sh"
