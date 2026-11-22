# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Flow-Factory's fixed-pixel, aspect-preserving SenseNova reference
geometry."""

from __future__ import annotations


def fixed_pixel_size(
    width: int, height: int, pixels: int, max_side_length: int = 0, factor: int = 32
) -> tuple[int, int]:
    if any(type(v) is not int or v <= 0 for v in (width, height, pixels, factor)):
        raise ValueError("Image dimensions, fixed pixels and alignment factor must be positive integers")
    if type(max_side_length) is not int or max_side_length < 0:
        raise ValueError("max_side_length must be a nonnegative integer")
    scale = (width * height / pixels) ** 0.5
    height, width = int(round(height / scale)), int(round(width / scale))
    if max_side_length and max(height, width) > max_side_length:
        ratio = max_side_length / max(height, width)
        height, width = int(height * ratio), int(width * ratio)
    height, width = height // factor * factor, width // factor * factor
    if min(height, width) <= 0:
        raise ValueError("Fixed-pixel resize produced an empty image")
    return width, height
