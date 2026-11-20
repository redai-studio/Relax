# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check the dtype contract for encoder inputs and metadata in transport."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image


try:
    from relax.utils.data.minimax import IMAGE_TOKEN
    from relax.utils.data.processing_utils import cast_encoder_input_for_transport
    from relax.utils.data.processor_pool import _resize_images_for_processor
    from relax.utils.multimodal.config import MultimodalConfig
except Exception as exc:  # pragma: no cover
    pytest.skip(f"processor_pool unavailable: {exc}", allow_module_level=True)


@pytest.mark.parametrize("key", ["pixel_values", "pixel_values_videos", "input_features"])
def test_transport_downcasts_large_encoder_inputs(key: str) -> None:
    pixels = torch.tensor([0.123456], dtype=torch.float32)
    torch.testing.assert_close(cast_encoder_input_for_transport(object(), key, pixels), pixels.bfloat16())


def test_transport_preserves_temporal_metadata() -> None:
    timestamps = torch.tensor([0.123456], dtype=torch.float32)
    assert cast_encoder_input_for_transport(object(), "video_second_per_grid", timestamps) is timestamps


def test_transport_preserves_minimax_pixels() -> None:
    pixels = torch.tensor([0.123456], dtype=torch.float32)
    processor = SimpleNamespace(image_token=IMAGE_TOKEN)
    assert cast_encoder_input_for_transport(processor, "pixel_values", pixels) is pixels


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
