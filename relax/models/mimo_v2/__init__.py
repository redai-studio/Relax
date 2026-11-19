# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Adapted from Miles (RadixArk/miles, fa13e2ec), licensed under Apache-2.0.
# Copyright 2025-2026 RadixArk, Inc. Miles includes software developed from Slime,
# Copyright 2025 Zhipu AI, licensed under Apache-2.0.

"""Megatron Bridge support for the MiMo-V2 text decoder."""

from .attention import MiMoV2SelfAttention, MiMoV2TEDotProductAttention
from .bridge import MiMoV2Bridge
from .provider import MiMoV2ModelProvider, mimo_v2_layer_spec


__all__ = [
    "MiMoV2Bridge",
    "MiMoV2ModelProvider",
    "MiMoV2SelfAttention",
    "MiMoV2TEDotProductAttention",
    "mimo_v2_layer_spec",
]
