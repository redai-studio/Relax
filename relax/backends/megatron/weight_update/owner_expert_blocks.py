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
) -> Iterator[tuple[str, torch.Tensor]]:
    """Copy each output before the converter advances; retain one projection.

    Input must be grouped by layer/fc1-or-fc2. The iterator validates exact
    owner coverage rather than relying on the original lexicographic order. The
    lease includes the canonical slab and one active TP shard; encoder
    clones/pending/diff scratch remain charged by the existing encoder.
    """
    if owners <= 0 or layout.num_experts % owners or not 0 <= owner < owners or layout.tp_size != 2:
        raise ValueError("Owner blocks require evenly partitioned experts and TP2")
    count = layout.num_experts // owners
    start, end = owner * count, (owner + 1) * count
    key, slab, lease = None, None, 0
    seen, completed = set(), set()

    def check_complete():
        projections = {"gate_proj", "up_proj"} if key[1] == "gate_up_proj" else {"down_proj"}
        if seen != {(expert, projection) for expert in range(start, end) for projection in projections}:
            raise ValueError("Canonical owner block has missing or duplicate expert projections")

    try:
        for source_owner, expert, name, tensor in named:
            match = _CANONICAL.fullmatch(name)
            if match is None or source_owner != owner or int(match[2]) != expert or not start <= expert < end:
                raise ValueError("Canonical expert does not belong to its owner block")
            prefix, _, projection = match.groups()
            incoming = (prefix, "down_proj" if projection == "down_proj" else "gate_up_proj")
            if key is not None and incoming != key:
                check_complete()
                yield f"{key[0]}.{key[1]}[{start}:{end}]", slab
                completed.add(key)
                slab = None
                budget.release(lease)
                lease = 0
                seen.clear()
            if slab is None:
                if incoming in completed:
                    raise ValueError("Canonical owner projections must be grouped")
                key = incoming
                h, i = layout.hidden_size, layout.intermediate_size
                shape = (count, h, i) if projection == "down_proj" else (count, 2 * i, h)
                slab_bytes = 2 * shape[0] * shape[1] * shape[2]
                # At most 192MiB for the fixed 35B owner/TP2 geometry.
                temporary = slab_bytes + slab_bytes // layout.tp_size
                if temporary > 192 << 20:
                    raise ValueError("Canonical owner packing exceeds the bounded conversion scratch")
                budget.reserve(temporary)
                lease = temporary
                slab = torch.empty(shape, dtype=torch.bfloat16, device=tensor.device)
            target = slab[expert - start]
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
        yield f"{key[0]}.{key[1]}[{start}:{end}]", slab
    finally:
        if lease:
            budget.release(lease)
