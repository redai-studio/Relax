# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""The profiler's collectives over its dedicated Gloo group: the window row,
the rank metadata, the samples to the primary rank and the primary rank's id.

Every rank issues each collective whatever fails locally (a rank that cannot
build or pickle its part sends a placeholder and raises after the collective),
so a local failure never leaves the ranks on different collectives. The group's
ranks are the global ranks. Torch only.
"""

from __future__ import annotations

import pickle
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


_GATHER_GROUP: Any = None


def init_straggler_gather_group(timeout_s: float) -> None:
    """Create the profiler's dedicated Gloo group, with a timeout far shorter
    than the training one.

    The generic Gloo group is shared with weight sync and other paths, so its
    timeout cannot be lowered for this; a straggler gather is ``world × 312 B``
    of CPU copies (milliseconds), and a rank that vanished must not park the
    rest of the job for ``--distributed-timeout-minutes``. Every member of the
    default group must call this in the same order, so it runs on all ranks of
    the actor group before any gather.
    """
    global _GATHER_GROUP
    if _GATHER_GROUP is None:
        _GATHER_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=timeout_s))


def get_straggler_gather_group() -> Any:
    if _GATHER_GROUP is None:
        raise RuntimeError("the straggler Gloo group has not been initialized; call install_straggler_profiler first")
    return _GATHER_GROUP


def gloo_all_gather(values: list[float]) -> list[list[float]]:
    """All-gather one CPU vector over the straggler Gloo group.

    A rank that cannot build its tensor sends NaN in every slot and raises
    after the gather, and the collector skips a window with such a row on every
    rank.
    """
    group = get_straggler_gather_group()
    error = None
    try:
        local = torch.tensor(values, dtype=torch.float64)
    except Exception as exc:
        error, local = exc, torch.full((len(values),), float("nan"), dtype=torch.float64)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    if error is not None:
        raise error
    return [row.tolist() for row in gathered]


def gloo_primary_rank(is_primary: bool) -> int | None:
    """The primary rank's global rank, the same on every rank: one
    ``all_reduce`` over the straggler Gloo group; ``None`` without a
    primary."""
    value = torch.tensor([dist.get_rank() if is_primary else -1], dtype=torch.int64)
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=get_straggler_gather_group())
    rank = int(value.item())
    return rank if rank >= 0 else None


def _pickled(obj: Any) -> tuple[bytes, Exception | None]:
    """``obj`` pickled, or ``None`` pickled and the error.

    The object collectives pickle their argument before their first collective,
    so a rank whose pickling failed there would skip the collectives its peers
    issue; pickling first keeps every rank on them.
    """
    try:
        return pickle.dumps(obj), None
    except Exception as exc:
        return pickle.dumps(None), exc


def gloo_all_gather_objects(obj: Any) -> list[Any]:
    """All-gather one picklable object per rank over the straggler Gloo group;
    every rank raises when one of them could not send its object."""
    group = get_straggler_gather_group()
    data, error = _pickled(obj)
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, data, group=group)
    if error is not None:
        raise error
    objects = [pickle.loads(item) for item in gathered]
    missing = [rank for rank, item in enumerate(objects) if item is None]
    if missing:
        raise RuntimeError(f"ranks {missing} sent no object")
    return objects


def gloo_gather_objects_to(obj: Any, dst: int) -> list[Any] | None:
    """Gather one picklable object per rank on rank ``dst`` (``None``
    elsewhere) over the straggler Gloo group; a rank that could not send its
    object raises after the gather."""
    group = get_straggler_gather_group()
    data, error = _pickled(obj)
    gathered = [None] * dist.get_world_size(group) if dist.get_rank() == dst else None
    dist.gather_object(data, gathered, dst=dst, group=group)
    if error is not None:
        raise error
    return [pickle.loads(item) for item in gathered] if gathered is not None else None
