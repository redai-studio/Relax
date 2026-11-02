# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""In-place native expert slices for SGLang's custom weight loader API."""

import json
import re
from collections.abc import Sequence

import torch


EXPERT_SLICE_LOADER = "relax.backends.sglang.weight_loader.load_expert_slices"
_SLICE_PREFIX = "relax.expert_slice.v1:"
_PARAMETER = re.compile(r"model\.layers\.\d+\.mlp\.experts\.(w13_weight|w2_weight)")


def expert_slice_name(name: str, start: int, shape: Sequence[int]) -> str:
    """Encode the receiver-local offset and expected full parameter shape."""
    return _SLICE_PREFIX + json.dumps([name, start, list(shape)], separators=(",", ":"))


@torch.no_grad()
def load_expert_slices(model: torch.nn.Module, weights: Sequence[tuple[str, torch.Tensor]]) -> None:
    """Validate a complete request, then copy into existing parameter storage.

    Resolve parameters on every call: rollout memory may have been released and
    restored since the previous update. No model handles or GPU buffers
    persist.
    """
    copies, ranges = [], {}
    for descriptor, tensor in weights:
        if not descriptor.startswith(_SLICE_PREFIX):
            raise ValueError("Unsupported expert slice descriptor")
        name, start, shape = json.loads(descriptor[len(_SLICE_PREFIX) :])
        if (
            not isinstance(name, str)
            or _PARAMETER.fullmatch(name) is None
            or type(start) is not int
            or not isinstance(shape, list)
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise ValueError("Invalid expert slice metadata")
        parameter = model.get_parameter(name)
        if (
            tuple(parameter.shape) != tuple(shape)
            or parameter.dtype != torch.bfloat16
            or tensor.dtype != parameter.dtype
            or tensor.device != parameter.device
            or tensor.ndim != 3
            or tuple(tensor.shape[1:]) != tuple(shape[1:])
            or not 0 <= start < start + tensor.shape[0] <= shape[0]
        ):
            raise ValueError(f"Expert slice does not match receiver parameter: {name}")
        end = start + tensor.shape[0]
        previous = ranges.setdefault(name, [])
        if any(start < other_end and other_start < end for other_start, other_end in previous):
            raise ValueError(f"Overlapping expert slices: {name}")
        previous.append((start, end))
        copies.append((parameter.narrow(0, start, tensor.shape[0]), tensor))
    for destination, tensor in copies:
        destination.copy_(tensor)
