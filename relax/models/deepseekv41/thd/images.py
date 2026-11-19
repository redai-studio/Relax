# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Sequence

import torch


def pack_image_inputs(
    samples: Sequence[dict[str, torch.Tensor] | None],
    lengths: Sequence[int],
    *,
    include_pixels: bool,
) -> dict:
    """Rebase the processor's sample-local image spans onto the contiguous packed stream."""
    grids, spans, pixels = [], [], []
    offset = 0
    for sample, length in zip(samples, lengths, strict=True):
        if sample:
            grid = sample["image_grid_hws"]
            positions = sample["image_spans"]
            if grid.device.type != "cpu" or positions.device.type != "cpu":
                raise ValueError("DeepSeek V4.1 image geometry must remain on the host until packing")
            if grid.ndim != 2 or grid.shape[1] != 2 or positions.shape != grid.shape:
                raise ValueError("DeepSeek V4.1 expects one (grid, span) pair per image")
            end = num_patches = 0
            for (h, w), (start, size) in zip(grid.tolist(), positions.tolist(), strict=True):
                expanded = ((h + 2) // 3) * ((w + 2) // 3 + 1) + 2
                if min(h, w) < 1 or start < end or size != expanded or start + size > length:
                    raise ValueError("DeepSeek V4.1 image span disagrees with its grid or sample boundaries")
                grids.append((h, w))
                spans.append((offset + start, size))
                end = start + size
                num_patches += h * w
            if include_pixels:
                value = sample["pixel_values"]
                if value.shape[0] != num_patches:
                    raise ValueError("DeepSeek V4.1 patch count disagrees with the per-image grids")
                if num_patches:
                    pixels.append(value)
        offset += length
    if not spans:
        return {}
    result = {"image_grid_hws": tuple(grids), "image_spans": tuple(spans)}
    if pixels:
        result["pixel_values"] = pixels[0] if len(pixels) == 1 else torch.cat(pixels)
    return result
