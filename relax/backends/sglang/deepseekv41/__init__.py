# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""V4.1 inference hooks without importing the Megatron model registry.

Keep this package outside relax.models: its eager bridge registrations load the
training FlashMLA extension before SGLang's, changing native dispatch.
"""

import os

from .patch import apply_dp_prefill_padding_patch, apply_engram_idle_patch, apply_vision_route_capture_patch


apply_dp_prefill_padding_patch()
apply_engram_idle_patch()
if os.environ.get("RELAX_DSV41_CAPTURE_VISION_ROUTES") == "1":
    apply_vision_route_capture_patch()
