# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Sequence
from dataclasses import dataclass

import torch


SAMPLING_MASK_FIELDS = ("rollout_sampling_mask_ids", "rollout_sampling_mask_offsets")


def _owned_integer_tensor(values: Sequence[int] | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(values, device="cpu")
    if tensor.ndim != 1:
        raise ValueError("sampling mask arrays must be one-dimensional")
    if tensor.numel():
        if tensor.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("sampling mask arrays must contain integers")
        tensor = tensor.to(torch.int64)
        if (tensor < 0).any() or (tensor > torch.iinfo(dtype).max).any():
            raise ValueError(f"sampling mask values must fit nonnegative {dtype}")
    return tensor.to(dtype=dtype).clone()


@dataclass(frozen=True, eq=False)
class RolloutSamplingMask:
    """Owned CPU CSR support, with one nonempty row per response token.

    ``ids[offsets[t]:offsets[t + 1]]`` is the original rollout support at t.
    Construction and all operations own their storage; tensors are read-only by
    convention, like other tensor-valued sample fields.
    """

    ids: torch.Tensor
    offsets: torch.Tensor

    def __post_init__(self) -> None:
        ids = _owned_integer_tensor(self.ids, torch.int32)
        offsets = _owned_integer_tensor(self.offsets, torch.int64)
        if not offsets.numel() or offsets[0] != 0 or offsets[-1] != ids.numel():
            raise ValueError("sampling mask offsets must start at zero and end at the ids count")
        if (offsets[1:] <= offsets[:-1]).any():
            raise ValueError("every response token must have nonempty sampling support")
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "offsets", offsets)

    def __len__(self) -> int:
        return self.offsets.numel() - 1

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[int]]) -> "RolloutSamplingMask":
        ids, offsets = [], [0]
        for row in rows:
            ids.extend(row)
            offsets.append(len(ids))
        return cls(ids, offsets)

    @classmethod
    def concatenate(cls, masks: Sequence["RolloutSamplingMask"]) -> "RolloutSamplingMask":
        if not masks:
            return cls([], [0])
        offsets = [torch.zeros(1, dtype=torch.int64)]
        size = 0
        for mask in masks:
            offsets.append(mask.offsets[1:] + size)
            size += mask.ids.numel()
        return cls(torch.cat([mask.ids for mask in masks]), torch.cat(offsets))

    def prefix(self, response_length: int) -> "RolloutSamplingMask":
        if not 0 <= response_length <= len(self):
            raise ValueError(f"sampling mask prefix length must be in [0, {len(self)}]")
        return type(self)(self.ids[: self.offsets[response_length]], self.offsets[: response_length + 1])
