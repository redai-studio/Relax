#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONTEXT_LENGTH=512k
exec bash "${SCRIPT_DIR}/run-qwen38-long-context.sh" "$@"
