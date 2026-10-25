# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Compatibility adapter for OSWorld's rebuilt observation prompts."""

from typing import Any

from relax.utils.visual_context import VisualContext as SharedVisualContext
from relax.utils.visual_context import image_count  # noqa: F401 - compatibility re-export


class VisualContext(SharedVisualContext):
    def __init__(self, keep_images: int = 2, prune_images: int = 2, **kwargs: Any) -> None:
        kwargs.setdefault("history_mode", "latest_observation")
        kwargs.setdefault("metric_prefix", "osworld")
        super().__init__(keep_images, prune_images, **kwargs)
