# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in Kimi expert routing to colocated rollout GPUs.

Only static EP with one complete expert per destination shard is supported. HF
names retain global expert IDs; the SGLang loader performs local indexing. All
ranks call exchange once per common source bucket, including empty ranks.
"""

import dataclasses
import json
import math
import re
from collections.abc import Sequence

import torch
import torch.distributed as dist


_EXPERT = re.compile(
    r"^(?:language_model\.)?model\.layers\.\d+\.block_sparse_moe\.experts\.(\d+)\.w[123]\.weight_(?:packed|scale)$"
)


@dataclasses.dataclass(frozen=True)
class ExpertLayout:
    num_experts: int
    ep_size: int
    engine_offsets: tuple[int, ...]
    world_size: int

    def __post_init__(self) -> None:
        if self.ep_size <= 0 or self.num_experts <= 0 or self.num_experts % self.ep_size:
            raise ValueError("Expert routing requires num_experts divisible by positive rollout EP size")
        ranks = [rank for offset in self.engine_offsets for rank in range(offset, offset + self.ep_size)]
        if sorted(ranks) != list(range(self.world_size)) or not ranks:
            raise ValueError("Expert routing requires non-overlapping engine GPU ranges covering all training ranks")

    def destinations(self, name: str) -> tuple[int, ...]:
        match = _EXPERT.fullmatch(name)
        if match is None or int(match[1]) >= self.num_experts:
            raise ValueError(f"Unsupported Kimi routed-expert tensor: {name}")
        shard = int(match[1]) // (self.num_experts // self.ep_size)
        return tuple(offset + shard for offset in self.engine_offsets)


def validate_server_layout(config: dict, gpu_count: int, ep_size: int, num_experts: int | None = None) -> None:
    """Validate actual resolved SGLang settings, including engine overrides."""
    required = {"tp_size": gpu_count, "ep_size": ep_size, "pp_size": 1, "moe_dp_size": 1}
    if gpu_count != ep_size or any(config.get(key) != value for key, value in required.items()):
        raise ValueError(f"Expert routing requires rollout TP=EP, PP=1 and MoE DP=1; expected {required}")
    if num_experts is not None and config.get("num_experts") != num_experts:
        raise ValueError("Expert routing requires identical training and rollout expert counts")
    if json.loads(config.get("json_model_override_args") or "{}"):
        raise ValueError("Expert routing does not support rollout model config overrides")
    if (
        config.get("enable_eplb", False)
        or config.get("ep_num_redundant_experts", 0)
        or config.get("init_expert_location", "trivial") != "trivial"
        or config.get("ep_join_mode")
        or config.get("speculative_algorithm")
        or config.get("moe_runner_backend") != "flashinfer_mxfp4"
    ):
        raise ValueError("Expert routing requires static flashinfer_mxfp4 experts without EPLB, elastic EP or MTP")


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize

    @property
    def padded_bytes(self) -> int:
        # Align mixed-dtype views, including source boundaries in all-to-all.
        return (self.nbytes + 7) // 8 * 8


@dataclasses.dataclass
class TransferPlan:
    send_indices: list[list[int]]
    receive_specs: list[list[TensorSpec]]
    send_splits: list[int]
    receive_splits: list[int]
    max_receive_bytes: int
    destination_bytes: tuple[int, ...]
    total_payload_bytes: int
    replicated_payload_bytes: int


def build_transfer_plan(schemas: Sequence[Sequence[TensorSpec]], layout: ExpertLayout, rank: int) -> TransferPlan:
    if len(schemas) != layout.world_size or not 0 <= rank < layout.world_size:
        raise ValueError("Expert routing schema/rank does not match process group")
    send_indices = [[] for _ in schemas]
    receive_specs = [[] for _ in schemas]
    seen = set()
    total_bytes = 0
    destination_bytes = [0] * layout.world_size
    for source, schema in enumerate(schemas):
        for index, spec in enumerate(schema):
            if spec.name in seen:
                raise ValueError(f"Expert routing requires a unique owner: {spec.name}")
            seen.add(spec.name)
            destinations = layout.destinations(spec.name)
            total_bytes += spec.nbytes
            for destination in destinations:
                destination_bytes[destination] += spec.padded_bytes
            if source == rank:
                for destination in destinations:
                    send_indices[destination].append(index)
            if rank in destinations:
                receive_specs[source].append(spec)
        # Packed weights and scales must be complete within the source bucket.
        weights = {spec.name.removesuffix("_packed") for spec in schema if spec.name.endswith("_packed")}
        scales = {spec.name.removesuffix("_scale") for spec in schema if spec.name.endswith("_scale")}
        if weights != scales:
            raise ValueError("Expert routing requires packed/scale pairs in the same source bucket")
    return TransferPlan(
        send_indices=send_indices,
        receive_specs=receive_specs,
        send_splits=[sum(schemas[rank][index].padded_bytes for index in indices) for indices in send_indices],
        receive_splits=[sum(spec.padded_bytes for spec in specs) for specs in receive_specs],
        max_receive_bytes=max(destination_bytes),
        destination_bytes=tuple(destination_bytes),
        total_payload_bytes=total_bytes * len(layout.engine_offsets),
        replicated_payload_bytes=total_bytes * layout.world_size,
    )


class ExpertRouter:
    def __init__(
        self, layout: ExpertLayout, *, payload_group: dist.ProcessGroup, metadata_group: dist.ProcessGroup
    ) -> None:
        self.layout = layout
        self.payload_group = payload_group
        self.metadata_group = metadata_group
        expected = list(range(layout.world_size))
        if any(dist.get_process_group_ranks(group) != expected for group in (payload_group, metadata_group)):
            raise ValueError("Expert routing requires world-rank-ordered payload and CPU metadata groups")
        self.rank = dist.get_rank(payload_group)
        self._plans: dict[int, tuple[tuple[TensorSpec, ...], TransferPlan]] = {}
        self.max_receive_bytes = 0
        self.destination_bytes: tuple[int, ...] = (0,) * layout.world_size
        self.payload_bytes = 0
        self.replicated_bytes = 0

    def reset_metrics(self) -> None:
        self.payload_bytes = self.replicated_bytes = 0

    def exchange(
        self,
        bucket: int,
        named_tensors: list[tuple[str, torch.Tensor]],
        device: torch.device,
        *,
        error: str | None = None,
        receive_budget: int | None = None,
    ) -> list[tuple[str, torch.Tensor]]:
        schema = tuple(TensorSpec(name, tuple(tensor.shape), tensor.dtype) for name, tensor in named_tensors)
        cached = self._plans.get(bucket)
        # CPU-only agreement: no accelerator-to-host synchronization. A change
        # on one owner refreshes all ranks before any variable-sized collective.
        refresh = torch.tensor(
            [int(cached is None or cached[0] != schema), int(error is not None)], dtype=torch.int32, device="cpu"
        )
        dist.all_reduce(refresh, op=dist.ReduceOp.MAX, group=self.metadata_group)
        if refresh[1].item():
            errors = [None] * self.layout.world_size
            dist.all_gather_object(errors, error, group=self.metadata_group)
            raise RuntimeError(f"Expert conversion failed before routing: {errors}")
        if refresh[0].item():
            schemas: list = [None] * self.layout.world_size
            dist.all_gather_object(schemas, schema, group=self.metadata_group)
            plan = build_transfer_plan(schemas, self.layout, self.rank)
            self._plans[bucket] = (schema, plan)
        else:
            plan = cached[1]
        # The maximum includes alignment padding and is identical on all ranks.
        # Reject before allocating packing/receive buffers or entering NCCL.
        if receive_budget is not None and plan.max_receive_bytes > receive_budget:
            raise ValueError("Routed expert receive exceeds the IPC budget")
        self.max_receive_bytes = plan.max_receive_bytes
        self.destination_bytes = plan.destination_bytes
        self.payload_bytes += plan.total_payload_bytes
        self.replicated_bytes += plan.replicated_payload_bytes
        if not plan.total_payload_bytes:
            return []
        parts = []
        for indices in plan.send_indices:
            for index in indices:
                tensor = named_tensors[index][1].contiguous().flatten().view(torch.uint8)
                parts.append(tensor)
                padding = schema[index].padded_bytes - schema[index].nbytes
                if padding:
                    parts.append(torch.zeros(padding, dtype=torch.uint8, device=device))
        send = torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8, device=device)
        receive = torch.empty(sum(plan.receive_splits), dtype=torch.uint8, device=device)
        dist.all_to_all_single(
            receive,
            send,
            output_split_sizes=plan.receive_splits,
            input_split_sizes=plan.send_splits,
            group=self.payload_group,
        )
        result = []
        offset = 0
        for specs in plan.receive_specs:
            for spec in specs:
                tensor = receive[offset : offset + spec.nbytes].view(spec.dtype).reshape(spec.shape)
                result.append((spec.name, tensor))
                offset += spec.padded_bytes
        return result
