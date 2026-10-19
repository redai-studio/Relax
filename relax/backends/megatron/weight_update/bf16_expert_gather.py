# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Owner-balanced BF16 expert export, following Miles' packed gather design.

Layout descriptors survive updates; GPU buffers and current process groups do
not. The caller supplies the live EP group after each offload/reload cycle.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist


if TYPE_CHECKING:
    from relax.utils.types import ParamInfo


NamedTensor = tuple[str, torch.Tensor]
TensorLayout = tuple[str, tuple[int, ...], int]


def bucket_experts_by_owner(
    infos: Sequence["ParamInfo"], buffer_size: int, ranks: Sequence[int]
) -> list[list["ParamInfo"]]:
    """Pack each owner's share so one gathered round stays within the
    budget."""
    budget = buffer_size // len(ranks)
    owners: dict[int, list] = {rank: [] for rank in ranks}
    for info in infos:
        if info.src_rank not in owners or not 0 < info.size <= budget:
            raise ValueError(f"Unsupported expert owner or byte size: {info.name}")
        owners[info.src_rank].append(info)
    owner_batches = []
    for rank in ranks:
        batches, pending, size = [], [], 0
        for info in owners[rank]:
            if pending and size + info.size > budget:
                batches.append(pending)
                pending, size = [], 0
            pending.append(info)
            size += info.size
        if pending:
            batches.append(pending)
        owner_batches.append(batches)
    return [
        [info for batches in owner_batches if index < len(batches) for info in batches[index]]
        for index in range(max(map(len, owner_batches), default=0))
    ]


def _layout(tensors: Sequence[NamedTensor]) -> tuple[TensorLayout, ...]:
    if any(tensor.dtype != torch.bfloat16 for _, tensor in tensors):
        raise ValueError("Packed BF16 expert gather requires BF16 tensors")
    return tuple((name, tuple(tensor.shape), tensor.numel()) for name, tensor in tensors)


class Bf16ExpertGather:
    """Cache CPU layouts; allocate independent receive storage on every
    call."""

    def __init__(self) -> None:
        self._ranks: tuple[int, ...] | None = None
        self._layouts: tuple[tuple[TensorLayout, ...], ...] | None = None
        self._sizes: tuple[int, ...] = ()

    def __call__(
        self, tensors: Sequence[NamedTensor], *, group: dist.ProcessGroup, device: torch.device
    ) -> list[NamedTensor]:
        ranks = tuple(dist.get_process_group_ranks(group))
        local_index = dist.get_rank(group=group)
        layout = _layout(tensors)
        if self._layouts is None:
            layouts = [None] * len(ranks)
            dist.all_gather_object(layouts, layout, group=group)
            self._layouts = tuple(layouts)
            self._ranks = ranks
            self._sizes = tuple(sum(entry[2] for entry in peer) for peer in self._layouts)
        elif ranks != self._ranks or layout != self._layouts[local_index]:
            raise ValueError("Expert ownership or tensor layout changed; rebuild the gather plan")

        storage = torch.empty(sum(self._sizes), dtype=torch.bfloat16, device=device)
        payloads = list(storage.split(self._sizes))
        local = payloads[local_index]
        offset = 0
        for (_, tensor), (_, _, size) in zip(tensors, layout, strict=True):
            local.narrow(0, offset, size).copy_(tensor.reshape(-1))
            offset += size

        sources = [index for index, size in enumerate(self._sizes) if size]
        handle = None
        if len(sources) == 1:
            src = sources[0]
            handle = dist.broadcast(payloads[src], src=ranks[src], group=group, async_op=True)
        elif sources and len(set(self._sizes)) == 1:
            handle = dist.all_gather_into_tensor(storage, local, group=group, async_op=True)
        elif sources:
            handle = dist.all_gather(payloads, local, group=group, async_op=True)

        result = []
        for peer, payload in zip(self._layouts, payloads, strict=True):
            offset = 0
            for name, shape, size in peer:
                result.append((name, payload.narrow(0, offset, size).view(shape)))
                offset += size
        if handle is not None:
            handle.wait()
        return result
