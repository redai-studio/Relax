# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Checked BF16 rectangle metadata; no per-rectangle Tensor view creation."""

from collections import defaultdict
from itertools import zip_longest

import torch


_INT64_MAX = (1 << 63) - 1


def rectangle_metadata(
    sources: dict, targets: dict, writes: list[dict], *, owner_rotation: int = 0
) -> dict[tuple[int, int], list[list[int]]]:
    """Group equal-size copies so the GPU grid has no padded rectangles."""
    metadata = {}
    for mapping, side in ((sources, "src"), (targets, "dst")):
        for key, tensor in mapping.items():
            if tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                raise ValueError("Batched IPC requires contiguous BF16 storage")
            metadata[side, key] = (tensor.data_ptr(), tensor.numel() * 2, tensor.device)
            pointer, size, _ = metadata[side, key]
            if not 0 <= pointer <= pointer + size <= _INT64_MAX:
                raise ValueError("IPC allocation exceeds int64 address bounds")
    if len({item[2] for item in metadata.values()}) != 1:
        raise ValueError("All batched IPC allocations must use the same CUDA context")
    groups = defaultdict(list)
    for write in writes:
        source = metadata["src", (write["source_rank"], write["source"])]
        target = metadata["dst", write["param"]]
        rows, width = write["rows"], write["row_bytes"]
        if type(rows) is not int or type(width) is not int or min(rows, width) <= 0 or width % 2:
            raise ValueError("Invalid BF16 rectangle extent")
        if source[2] != target[2]:
            raise ValueError("IPC sources must be mapped into the target CUDA context")
        for tensor, side in ((source, "src"), (target, "dst")):
            offset, stride = write[f"{side}_offset"], write[f"{side}_stride"]
            if (
                type(offset) is not int
                or type(stride) is not int
                or offset < 0
                or stride < width
                or offset % 2
                or stride % 2
                or max(offset, stride, rows, width) > _INT64_MAX
                or offset + (rows - 1) * stride + width > tensor[1]
            ):
                raise ValueError("Batched IPC rectangle exceeds aligned storage bounds")
        groups[rows * width // 2, width // 2].append(
            [
                source[0],
                target[0],
                write["src_offset"] // 2,
                write["dst_offset"] // 2,
                width // 2,
                write["src_stride"] // 2,
                write["dst_stride"] // 2,
            ]
        )
    # Interleave owners instead of draining one owner's entire model first.
    # All destinations are disjoint, as checked by validate_expert_targets.
    output = {}
    for elements, descriptors in groups.items():
        by_source = defaultdict(list)
        for descriptor in descriptors:
            by_source[descriptor[0]].append(descriptor)
        owners = list(by_source.values())
        start = owner_rotation % len(owners)
        owners = owners[start:] + owners[:start]
        output[elements] = [item for row in zip_longest(*owners) for item in row if item is not None]
    return output


def launch_rectangles(
    groups: dict[tuple[int, int], list[list[int]] | torch.Tensor],
    device: torch.device,
    live_metadata: list,
    *,
    native_checks: list | None = None,
) -> None:
    """Caller owns the final fence and retains metadata until it completes."""
    from .rectangle_kernel import BLOCK_SIZE, launch

    if device != torch.device("cuda", torch.cuda.current_device()):
        raise ValueError("Batched IPC must launch in the receiving CUDA context")
    for (elements, columns), descriptors in groups.items():
        if isinstance(descriptors, torch.Tensor):
            if descriptors.device.type != "cpu" or descriptors.dtype != torch.int64:
                raise ValueError("Compiled rectangle descriptors must be CPU int64 tensors")
            metadata = descriptors.to(device=device)
        else:
            metadata = torch.tensor(descriptors, dtype=torch.int64, device=device)
        live_metadata.append(metadata)
        checks = None
        if native_checks is not None:
            checks = torch.ones(
                (len(descriptors), (elements + BLOCK_SIZE - 1) // BLOCK_SIZE), dtype=torch.int32, device=device
            )
            live_metadata.append(checks)
            native_checks.append(checks)
        for start in range(0, len(descriptors), 65535):
            if checks is None:
                launch(metadata[start : start + 65535], elements, columns)
            else:
                launch(metadata[start : start + 65535], elements, columns, checks[start : start + 65535])
