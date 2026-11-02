# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Mapping
from typing import Any


def validate_expert_weight_sync_layout(layout: Mapping[str, Any], config: Any) -> None:
    if layout.get("tp_size", 1) > 1 and "KeyeVL2MoeForConditionalGeneration" in (
        getattr(config, "architectures", None) or []
    ):
        raise ValueError("Keye TP row projections require HF weight loading, not native expert shards")
