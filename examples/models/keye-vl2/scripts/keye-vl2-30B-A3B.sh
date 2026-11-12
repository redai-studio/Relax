# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Keye's language backbone retains Qwen3-MoE dimensions, with a different RoPE base.
source "$(dirname "${BASH_SOURCE[0]}")/../../../../scripts/models/qwen3-30B-A3B.sh"
MODEL_ARGS+=(--rotary-base 10000000)
