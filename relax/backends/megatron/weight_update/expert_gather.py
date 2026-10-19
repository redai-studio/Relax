# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""BF16 expert export using owner-balanced or projection-packed collectives.

Layout descriptors survive updates; GPU buffers and current process groups do
not. The caller supplies the live EP group after each offload/reload cycle.
"""

import re
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


_SOURCE = re.compile(r"^(.*\.layers\.(\d+)\.mlp\.experts)\.linear_fc([12])\.weight(\d+)$")
_TARGET = re.compile(r"^(.*\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight$")


def bucket_expert_projections(
    infos: Sequence["ParamInfo"], num_experts: int, ranks: Sequence[int]
) -> list[list["ParamInfo"]]:
    """Validate complete, contiguous EP ownership and group by
    layer/projection."""
    if not ranks or num_experts <= 0 or num_experts % len(ranks):
        raise ValueError("Packed experts require evenly partitioned EP ownership")
    local_count = num_experts // len(ranks)
    groups = {}
    for info in infos:
        match = _SOURCE.fullmatch(info.name)
        if match is None:
            raise ValueError(f"Unsupported packed expert source: {info.name}")
        prefix, layer, projection, expert = match.groups()
        expert = int(expert)
        if (
            not 0 <= expert < num_experts
            or info.src_rank != ranks[expert // local_count]
            or info.dtype != torch.bfloat16
            or len(info.shape) != 2
            or min(info.shape) <= 0
        ):
            raise ValueError(f"Invalid packed expert ownership, dtype or shape: {info.name}")
        group = groups.setdefault((prefix, int(layer), int(projection)), {})
        if expert in group:
            raise ValueError(f"Duplicate packed expert: {info.name}")
        group[expert] = info
    buckets = []
    for key in sorted(groups, key=lambda key: (key[1], key[0], key[2])):
        group = groups[key]
        if set(group) != set(range(num_experts)) or len({tuple(info.shape) for info in group.values()}) != 1:
            raise ValueError(f"Incomplete or inconsistent packed experts: {key[0]}")
        other = groups.get((*key[:2], 3 - key[2]))
        if other is None or 0 not in other:
            raise ValueError(f"Missing packed expert projection: {key[0]}")
        fc1 = group[0] if key[2] == 1 else other[0]
        fc2 = other[0] if key[2] == 1 else group[0]
        if tuple(fc1.shape) != (2 * fc2.shape[1], fc2.shape[0]):
            raise ValueError(f"Inconsistent gate/up and down shapes: {key[0]}")
        buckets.append([group[expert] for expert in range(num_experts)])
    return buckets


class PackedExpertGather:
    """Gather one native projection directly into global expert order."""

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self._layouts = None
        self._ranks = None
        self._orders = ()
        self._name = None
        self._shape = ()

    def _prepare(self, layouts: Sequence[tuple[TensorLayout, ...]], ranks: tuple[int, ...]) -> None:
        local_count = self.num_experts // len(ranks)
        orders, expected_prefix, expected_projections, expected_shape = [], None, None, None
        for index, layout in enumerate(layouts):
            entries = {}
            for position, (name, shape, _size) in enumerate(layout):
                match = _TARGET.fullmatch(name)
                if match is None or len(shape) != 2:
                    raise ValueError(f"Unsupported packed expert output: {name}")
                prefix, expert, projection = match.groups()
                if expected_prefix is None:
                    expected_prefix, expected_shape = prefix, shape
                key = (int(expert), projection)
                if prefix != expected_prefix or shape != expected_shape or key in entries:
                    raise ValueError(f"Mixed or duplicate packed expert output: {name}")
                entries[key] = position
            projections = {projection for _, projection in entries}
            if projections not in ({"gate", "up"}, {"down"}):
                raise ValueError("Packed experts require complete gate/up or down projections")
            if expected_projections is None:
                expected_projections = projections
            if projections != expected_projections:
                raise ValueError("Packed expert projection differs between owners")
            order = ("gate", "up") if "gate" in projections else ("down",)
            keys = [
                (expert, projection)
                for expert in range(index * local_count, (index + 1) * local_count)
                for projection in order
            ]
            if set(keys) != set(entries):
                raise ValueError("Packed expert IDs must follow contiguous process-group rank order")
            orders.append(tuple(entries[key] for key in keys))
        rows, cols = expected_shape
        self._name = expected_prefix + (".gate_up_proj" if "gate" in expected_projections else ".down_proj")
        self._shape = (self.num_experts, rows * len(expected_projections), cols)
        self._orders = tuple(orders)
        self._layouts, self._ranks = tuple(layouts), ranks

    def __call__(
        self, tensors: Sequence[NamedTensor], *, group: dist.ProcessGroup, device: torch.device
    ) -> list[NamedTensor]:
        ranks = tuple(dist.get_process_group_ranks(group))
        local_index = dist.get_rank(group=group)
        layout = _layout(tensors)
        if not ranks or self.num_experts <= 0 or self.num_experts % len(ranks):
            raise ValueError("Packed experts require evenly partitioned EP ownership")
        if self._layouts is None:
            layouts = [None] * len(ranks)
            dist.all_gather_object(layouts, layout, group=group)
            self._prepare(layouts, ranks)
        elif ranks != self._ranks or layout != self._layouts[local_index]:
            raise ValueError("Expert ownership or tensor layout changed; rebuild the gather plan")
        output = torch.empty(self._shape, dtype=torch.bfloat16, device=device)
        local = output.chunk(len(ranks), dim=0)[local_index].view(-1)
        torch.cat([tensors[index][1].reshape(-1) for index in self._orders[local_index]], out=local)
        handle = dist.all_gather_into_tensor(output.view(-1), local, group=group, async_op=True)
        handle.wait()
        return [(self._name, output)]
