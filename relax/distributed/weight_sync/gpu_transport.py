# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-source NCCL delivery to GPU owners; no model or Ray ownership."""

from __future__ import annotations

from datetime import timedelta
from time import monotonic
from typing import Any


def remaining(deadline: float) -> float:
    value = deadline - monotonic()
    if value <= 0:
        raise TimeoutError("GPU weight synchronization deadline exceeded")
    return value


class GpuSyncGroup:
    """A dedicated group, initialized by all sources and proxies
    concurrently."""

    def __init__(self, connection: dict, *, rank: int, device: Any, timeout_s: float) -> None:
        import torch
        import torch.distributed as dist

        from relax.utils.distributed_utils import init_process_group

        self.device = torch.device(device)
        self.rank = rank
        self.group = init_process_group(
            backend="nccl",
            init_method=connection["init_method"],
            world_size=connection["world_size"],
            rank=rank,
            group_name=connection["group_name"],
            timeout=timedelta(seconds=timeout_s),
        )
        # Admit every fixed communicator member before the first payload.
        warmup = torch.zeros(1, dtype=torch.int32, device=self.device)
        dist.all_reduce(warmup, group=self.group)
        torch.cuda.synchronize(self.device)

    def receive(self, batch: dict, slab: Any = None) -> tuple[Any, list]:
        import torch
        import torch.distributed as dist

        if slab is None:
            slab = torch.empty(batch["bytes"], dtype=torch.uint8, device=self.device)
        if (
            slab.device != self.device
            or slab.dtype != torch.uint8
            or slab.ndim != 1
            or not slab.is_contiguous()
            or slab.numel() != batch["bytes"]
        ):
            raise ValueError("Receive storage must match the admitted byte slab")
        works = [dist.broadcast(slab, src=0, group=self.group, async_op=True)] if slab.numel() else []
        return slab, works

    def send(self, raw: Any, *, deadline: float) -> None:
        import torch.distributed as dist

        if self.rank != 0 or raw.device != self.device or raw.ndim != 1:
            raise ValueError("GPU frame broadcast requires the elected source and its contiguous payload")
        works = [dist.broadcast(raw, src=0, group=self.group, async_op=True)] if raw.numel() else []
        self.fence(works, deadline=deadline)

    def fence(self, works: list, *, deadline: float) -> None:
        import torch

        for work in works:
            work.wait(timeout=timedelta(seconds=remaining(deadline)))
        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        import torch.distributed as dist

        dist.destroy_process_group(self.group)
