# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Freeze canonical Bridge outputs into bounded contiguous owner ranges."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

import torch

from relax.distributed.weight_sync.streaming_delta import SnapshotBudget

from .expert_layout import DirectExpertLayout


_CANONICAL = re.compile(r"^(.*\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def iter_owner_expert_blocks(
    named: Iterable[tuple[int, int, str, torch.Tensor]],
    layout: DirectExpertLayout,
    owner: int,
    owners: int,
    budget: SnapshotBudget,
    *,
    bucket_bytes: int | None = None,
    source_rank: int | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Copy each output before the converter advances; retain one projection.

    Input must be grouped by layer/fc1-or-fc2 and bounded expert range. The
    Bridge grouped iterator provides numeric expert order. The iterator
    validates exact owner coverage rather than trusting the input order. The
    lease includes the canonical slab and one active TP shard; encoder
    clones/pending/diff scratch remain charged by the existing encoder.
    """
    if owners <= 0 or layout.num_experts % owners or not 0 <= owner < owners:
        raise ValueError("Owner blocks require evenly partitioned experts")
    if layout.ep_size != 1 or layout.moe_dp_size != 1:
        raise ValueError("Owner blocks require receiver EP1 and MoE DP1")
    capacity = budget.limit_bytes if bucket_bytes is None else bucket_bytes
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("Owner packing capacity must be positive")
    source_rank = owner if source_rank is None else source_rank
    count = layout.num_experts // owners
    start, end = owner * count, (owner + 1) * count
    key, slab, lease = None, None, 0
    block_start, block_end = start, end
    seen, completed = set(), set()
    required = set()

    def check_complete():
        projections = {"gate_proj", "up_proj"} if key[1] == "gate_up_proj" else {"down_proj"}
        if seen != {(expert, projection) for expert in range(block_start, block_end) for projection in projections}:
            raise ValueError("Canonical owner block has missing or duplicate expert projections")

    try:
        for source_owner, expert, name, tensor in named:
            match = _CANONICAL.fullmatch(name)
            if match is None or source_owner != source_rank or int(match[2]) != expert or not start <= expert < end:
                raise ValueError("Canonical expert does not belong to its owner block")
            prefix, _, projection = match.groups()
            h, i = layout.hidden_size, layout.intermediate_size
            elements = h * i * (1 if projection == "down_proj" else 2)
            if tensor.dtype != torch.bfloat16:
                raise ValueError("Canonical owner projection shape, dtype or coverage mismatch")
            per_expert = elements * tensor.element_size()
            # Account for canonical storage plus one live receiver TP slice.
            # Capacity is fixed configuration, never the current free budget.
            packing_per_expert = per_expert + per_expert // layout.tp_size
            block_count = min(count, capacity // packing_per_expert, (2**31 - 1) // elements)
            if block_count < 1 or packing_per_expert > budget.limit_bytes:
                raise ValueError("A single expert exceeds the owner packing capacity or shared budget")
            incoming_start = start + (expert - start) // block_count * block_count
            incoming = (prefix, "down_proj" if projection == "down_proj" else "gate_up_proj", incoming_start)
            required.update((prefix, incoming[1], offset) for offset in range(start, end, block_count))
            if key is not None and incoming != key:
                check_complete()
                yield f"{key[0]}.{key[1]}[{block_start}:{block_end}]", slab
                completed.add(key)
                slab = None
                budget.release(lease)
                lease = 0
                seen.clear()
            if slab is None:
                if incoming in completed:
                    raise ValueError("Canonical owner projections must be grouped")
                key = incoming
                block_start, block_end = incoming_start, min(incoming_start + block_count, end)
                shape = (
                    (block_end - block_start, h, i)
                    if projection == "down_proj"
                    else (block_end - block_start, 2 * i, h)
                )
                slab_bytes = (block_end - block_start) * per_expert
                temporary = slab_bytes + slab_bytes // layout.tp_size
                budget.reserve(temporary)
                lease = temporary
                slab = torch.empty(shape, dtype=torch.bfloat16, device=tensor.device)
            target = slab[expert - block_start]
            if projection == "gate_proj":
                target = target[: layout.intermediate_size]
            elif projection == "up_proj":
                target = target[layout.intermediate_size :]
            if (expert, projection) in seen or tensor.dtype != torch.bfloat16 or tensor.shape != target.shape:
                raise ValueError("Canonical owner projection shape, dtype or coverage mismatch")
            target.copy_(tensor)
            del target
            seen.add((expert, projection))
        if slab is None:
            raise ValueError("No canonical owner expert blocks")
        check_complete()
        completed.add(key)
        if completed != required or any(
            {item[1] for item in completed if item[0] == prefix} != {"gate_up_proj", "down_proj"}
            for prefix in {item[0] for item in completed}
        ):
            raise ValueError("Canonical owner block has missing or duplicate expert projections")
        yield f"{key[0]}.{key[1]}[{block_start}:{block_end}]", slab
    finally:
        if lease:
            budget.release(lease)
