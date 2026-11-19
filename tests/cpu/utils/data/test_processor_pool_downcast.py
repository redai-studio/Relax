# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Guard the bf16-downcast allowlist in processor_pool.

The shared-memory IPC path downcasts large fp32 encoder inputs to bf16, but
must NOT touch small fp32 metadata tensors consumed at full precision (e.g.
Qwen3-Omni's ``video_second_per_grid``, which drives mrope temporal position
IDs). This test pins the allowlist so a future "downcast every fp32 tensor"
regression is caught.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image


try:
    from relax.utils.data.processor_pool import _BF16_DOWNCAST_KEYS, _resize_images_for_processor
    from relax.utils.multimodal.config import MultimodalConfig
except Exception as exc:  # pragma: no cover
    pytest.skip(f"processor_pool unavailable: {exc}", allow_module_level=True)


def test_downcast_allowlist_contains_only_large_encoder_inputs() -> None:
    # Exactly the tensors that are re-cast to their encoder weight dtype downstream.
    assert _BF16_DOWNCAST_KEYS == frozenset({"pixel_values", "pixel_values_videos", "input_features"})


def test_video_second_per_grid_is_not_downcast() -> None:
    # Regression: this fp32 time-grid tensor is consumed at full precision
    # (second_per_grids[...].cpu().float()); it must stay out of the allowlist.
    assert "video_second_per_grid" not in _BF16_DOWNCAST_KEYS


def test_int_metadata_keys_are_not_in_allowlist() -> None:
    # Grids / masks are int or precision-sensitive; never in the downcast set.
    for key in ("image_grid_thw", "video_grid_thw", "feature_attention_mask"):
        assert key not in _BF16_DOWNCAST_KEYS


def test_downcast_predicate_matches_intended_behavior() -> None:
    # Mirror the in-loop guard `k in _BF16_DOWNCAST_KEYS and v.dtype == fp32`.
    def would_downcast(key: str, dtype: torch.dtype) -> bool:
        return key in _BF16_DOWNCAST_KEYS and dtype == torch.float32

    assert would_downcast("pixel_values", torch.float32) is True
    assert would_downcast("input_features", torch.float32) is True
    # fp32 metadata: left untouched.
    assert would_downcast("video_second_per_grid", torch.float32) is False
    # allowlisted key but already low precision: no-op.
    assert would_downcast("pixel_values", torch.bfloat16) is False


def test_processor_pool_applies_image_token_limit_with_processor_patch_size() -> None:
    processor = type("Processor", (), {"image_processor": type("ImageProcessor", (), {"patch_size": 16})()})()
    image = Image.new("RGB", (1080, 1440))

    resized = _resize_images_for_processor(
        processor,
        [image],
        MultimodalConfig(image_max_token_num=1024),
    )[0]

    assert resized.size == (864, 1152)
    assert resized.width * resized.height <= 1024 * (16 * 2) ** 2
