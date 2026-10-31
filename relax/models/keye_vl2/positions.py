# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Image-only Keye mRoPE, independent of attention's causal token offsets."""

import torch


def image_positions(input_ids: torch.Tensor, grid: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Return [3, B, T] positions without copying token metadata to the host.

    Rows are individual documents; right padding is excluded by the caller.
    """
    batch, width = input_ids.shape
    linear = torch.arange(width, device=input_ids.device).expand(batch, -1)
    image = input_ids == image_token_id
    starts = image & ~torch.nn.functional.pad(image[:, :-1], (1, 0), value=False)
    if grid is None or grid.shape[0] == 0:
        torch._assert_async(~image.any(), "Keye image placeholders require image grids")
        return linear.unsqueeze(0).expand(3, -1, -1)
    grids = grid.to(device=input_ids.device, dtype=torch.long)
    image_index = (starts.reshape(-1).long().cumsum(0) - 1).clamp_min(0).view(batch, width)
    torch._assert_async(starts.sum() == grid.shape[0], "Keye image runs and grids must match")
    t, h, w = grids[image_index].unbind(-1)
    h, w = h // 2, w // 2
    begin = torch.where(starts, linear, 0).cummax(-1).values
    offset = linear - begin
    coords = torch.stack((offset // (h * w), offset // w % h, offset % w))
    ends = image & ~torch.nn.functional.pad(image[:, 1:], (0, 1), value=False)
    reduction = torch.where(ends, t * h * w - torch.maximum(t, torch.maximum(h, w)), 0)
    shift = reduction.cumsum(-1) - reduction
    text = linear - shift
    base = torch.where(starts, text, 0).cummax(-1).values
    return torch.where(image.unsqueeze(0), coords + base.unsqueeze(0), text.unsqueeze(0))
