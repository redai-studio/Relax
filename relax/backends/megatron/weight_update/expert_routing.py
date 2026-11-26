# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Route complete expert weights to their colocated rollout EP owners.

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
_BF16_EXPERT = re.compile(
    r"^(?:(?:language_model\.)?model\.(?:language_model\.)?)?layers\.\d+\."
    r"(?:mlp|block_sparse_moe|ffn)\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
    r"|^(?:(?:language_model\.)?model\.(?:language_model\.)?)?layers\.\d+\."
    r"(?:mlp|block_sparse_moe|ffn)\.experts\.(\d+)\.(w[123])\.weight$"
)


def bf16_expert_projection(name: str) -> tuple[int, str] | None:
    """Read routing coordinates without rewriting the receiver's HF names."""
    match = _BF16_EXPERT.fullmatch(name)
    if match is None:
        return None
    expert, projection, other_expert, other_projection = match.groups()
    return int(expert or other_expert), projection or {"w1": "gate", "w2": "down", "w3": "up"}[other_projection]


def _fp8_weight_for_scale(name: str) -> str | None:
    if name.endswith(".weight_scale_inv"):
        return name.removesuffix("_scale_inv")
    if name.endswith(".scale"):
        return name.removesuffix(".scale") + ".weight"
    return None


@dataclasses.dataclass(frozen=True)
class ExpertLayout:
    num_experts: int
    ep_size: int
    engine_offsets: tuple[int, ...]
    world_size: int
    fp8_block_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.ep_size <= 0 or self.num_experts <= 0 or self.num_experts % self.ep_size:
            raise ValueError("Expert routing requires num_experts divisible by positive rollout EP size")
        if self.fp8_block_size is not None and (
            len(self.fp8_block_size) != 2 or any(type(value) is not int or value <= 0 for value in self.fp8_block_size)
        ):
            raise ValueError("Native FP8 routing requires a positive two-dimensional block shape")
        ranks = [rank for offset in self.engine_offsets for rank in range(offset, offset + self.ep_size)]
        if sorted(ranks) != list(range(self.world_size)) or not ranks:
            raise ValueError("Expert routing requires non-overlapping engine GPU ranges covering all training ranks")

    def destinations(self, name: str) -> tuple[int, ...]:
        match = _EXPERT.fullmatch(name)
        weight_name = _fp8_weight_for_scale(name) if self.fp8_block_size is not None else None
        coordinate = bf16_expert_projection(weight_name or name) if match is None else (int(match[1]), "")
        if coordinate is None:
            raise ValueError(f"Unsupported routed-expert tensor: {name}")
        return self.expert_destinations(coordinate[0])

    def expert_destinations(self, expert: int) -> tuple[int, ...]:
        if not 0 <= expert < self.num_experts:
            raise ValueError(f"Expert ID is outside the routed layout: {expert}")
        shard = expert // (self.num_experts // self.ep_size)
        return tuple(offset + shard for offset in self.engine_offsets)


def _validate_native_server_layout(config: dict, gpu_count: int, ep_size: int, num_experts: int) -> None:
    """Shared native loader topology and model-override contract."""
    required = dict(
        tp_size=gpu_count,
        ep_size=ep_size,
        pp_size=1,
        moe_dp_size=1,
        num_experts=num_experts,
        dtype="bfloat16",
        is_cuda=True,
    )
    if gpu_count != ep_size or any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Native expert routing requires CUDA, matching experts, TP=EP, PP1 and MoE DP1")
    overrides = json.loads(config.get("json_model_override_args") or "{}")
    if (
        not isinstance(overrides, dict)
        or set(overrides) - {"language_only", "language_model_only"}
        or any(type(v) is not bool for v in overrides.values())
    ):
        raise ValueError("Native expert routing cannot verify this model override")
    if (
        config.get("moe_runner_backend") not in ("deep_gemm", "triton", "flashinfer_cutlass")
        or config.get("speculative_algorithm") not in (None, "NEXTN", "EAGLE")
        or config.get("speculative_draft_model_path") not in (None, config.get("model_path"))
        or config.get("enable_eplb", False)
        or config.get("ep_num_redundant_experts", 0)
        or config.get("init_expert_location", "trivial") != "trivial"
        or config.get("ep_join_mode")
        or config.get("elastic_ep_backend")
        or config.get("dwdp_size", 1) != 1
        or config.get("enable_lora")
        or config.get("enforce_shared_experts_fusion")
        or config.get("kt_weight_path")
    ):
        raise ValueError("Native expert routing requires an unfused static expert layout without a separate draft")


def validate_bf16_server_layout(config: dict, gpu_count: int, ep_size: int, num_experts: int) -> None:
    """Route through the native HF loader only for verified static BF16 EP."""
    _validate_native_server_layout(config, gpu_count, ep_size, num_experts)
    if (
        config.get("quantization")
        or config.get("quantization_config")
        or not config.get("disable_shared_experts_fusion")
        and config.get("num_shared_experts") != 0
    ):
        raise ValueError("BF16 routing requires unfused shared experts and no quantization")


def validate_fp8_server_layout(config: dict, gpu_count: int, ep_size: int, num_experts: int) -> tuple[int, int]:
    """Use native FP8 conversion/loading with verified actual expert
    ownership."""
    _validate_native_server_layout(config, gpu_count, ep_size, num_experts)
    quantization = config.get("quantization_config") or {}
    block = quantization.get("weight_block_size")
    if (
        config.get("quantization") not in (None, "fp8")
        or quantization.get("quant_method") != "fp8"
        or quantization.get("fmt", "e4m3") != "e4m3"
        or not isinstance(block, (list, tuple))
        or len(block) != 2
        or any(type(value) is not int or value <= 0 for value in block)
    ):
        raise ValueError("Native FP8 routing requires the existing blockwise E4M3 conversion")
    profiles = config.get("native_fp8_experts")
    if not isinstance(profiles, list) or len(profiles) != gpu_count or not all(profiles):
        raise ValueError("Native FP8 receiver ownership is unavailable")
    modules = {item["module"] for item in profiles[0]}
    for tp_rank, profile in enumerate(profiles):
        if len(profile) != len(modules) or {item["module"] for item in profile} != modules:
            raise ValueError("Native FP8 workers disagree on their expert modules")
        for item in profile:
            expected = dict(
                moe_ep_rank=tp_rank,
                moe_ep_size=ep_size,
                moe_tp_size=1,
                _num_global_routed=num_experts,
                _num_local_routed=num_experts // ep_size,
                expert_begin=tp_rank * (num_experts // ep_size),
                expert_end=(tp_rank + 1) * (num_experts // ep_size),
                num_fused_shared_experts=0,
            )
            if any(item.get(key) != value for key, value in expected.items()):
                raise ValueError("Native FP8 worker ownership does not match its routed destination")
            if (
                not {"w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"}
                <= item["parameters"].keys()
            ):
                raise ValueError("Native FP8 worker is missing complete native weight/scale parameters")
    return tuple(block)


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
        if layout.fp8_block_size is not None:
            weights = {spec.name: spec for spec in schema if spec.name.endswith(".weight")}
            scales = {
                weight_name: spec for spec in schema if (weight_name := _fp8_weight_for_scale(spec.name)) is not None
            }
            if len(weights) + len(scales) != len(schema) or weights.keys() != scales.keys():
                raise ValueError("Native FP8 routing requires weight/scale pairs in the same source bucket")
            fused = {}
            for name, weight in weights.items():
                scale = scales[name]
                if (
                    weight.dtype != torch.float8_e4m3fn
                    or len(weight.shape) != 2
                    or scale.dtype != torch.float32
                    or scale.shape
                    != tuple(
                        (size + block - 1) // block
                        for size, block in zip(weight.shape, layout.fp8_block_size, strict=True)
                    )
                ):
                    raise ValueError("Native FP8 routing has an unsupported weight/scale schema")
                _, projection = bf16_expert_projection(name)
                fused.setdefault(name.rsplit(".", 2)[0], set()).add(projection)
            if any(("gate" in projections) != ("up" in projections) for projections in fused.values()):
                raise ValueError("Native FP8 routing requires complete fused gate/up projections")
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
        cached = self._plans.get(bucket)
        if (
            cached is not None
            and len(cached[0]) == len(named_tensors)
            and all(
                name == spec.name and tensor.shape == spec.shape and tensor.dtype == spec.dtype
                for (name, tensor), spec in zip(named_tensors, cached[0], strict=True)
            )
        ):
            schema = cached[0]
        else:
            schema = tuple(TensorSpec(name, tuple(tensor.shape), tensor.dtype) for name, tensor in named_tensors)
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
