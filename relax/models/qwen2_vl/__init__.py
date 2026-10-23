# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Qwen2-VL registration for Relax's existing Megatron training backend."""

from relax.models.qwen2_vl.bridge import Qwen2VLBridge
from relax.models.qwen2_vl.model import Qwen2VLModel
from relax.models.qwen2_vl.provider import Qwen2VLModelProvider


__all__ = ["Qwen2VLBridge", "Qwen2VLModel", "Qwen2VLModelProvider"]
