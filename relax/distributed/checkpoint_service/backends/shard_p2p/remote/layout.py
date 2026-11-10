# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Actual source coordinates and native weight layouts for expert, vocabulary
and TP shards."""

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist


# Source topology.


def _validate_grid(records: list[dict], actors: tuple[int, ...], partition_axis: str, replica_axis: str) -> tuple:
    """Check the full Cartesian product and reciprocal group membership."""
    world = len(actors)
    by_global = {}
    shape = None
    for record in records:
        sizes = []
        for axis in (partition_axis, replica_axis):
            members = record.get(axis + "_members")
            index = record.get(axis + "_rank")
            if (
                not isinstance(members, (list, tuple))
                or not members
                or len(set(members)) != len(members)
                or any(type(member) is not int or member not in actors for member in members)
                or type(index) is not int
                or not 0 <= index < len(members)
                or members[index] != record["global_rank"]
            ):
                raise ValueError("Invalid source MPU coordinates")
            sizes.append(len(members))
        if sizes[0] * sizes[1] != world or (shape is not None and shape != sizes):
            raise ValueError("Source MPU groups do not form a complete partition by replica grid")
        shape = sizes
        by_global[record["global_rank"]] = record
    for record in records:
        for axis, other in ((partition_axis, replica_axis), (replica_axis, partition_axis)):
            members = record[axis + "_members"]
            for index, peer in enumerate(members):
                actual = by_global[peer]
                if (
                    tuple(actual[axis + "_members"]) != tuple(members)
                    or actual[axis + "_rank"] != index
                    or actual[other + "_rank"] != record[other + "_rank"]
                ):
                    raise ValueError("Source MPU peer membership disagrees")
    if len({(record[partition_axis + "_rank"], record[replica_axis + "_rank"]) for record in records}) != world:
        raise ValueError("Source MPU grid has duplicated or missing coordinates")
    return tuple(shape)


def _validate_pipeline_grid(
    records: list[dict], actors: tuple[int, ...], partition_axis: str, replica_axis: str
) -> tuple:
    """Validate each stage's grid and the lanes that connect those stages."""
    if not any("pp_members" in record or "pp_rank" in record for record in records):
        return _validate_grid(records, actors, partition_axis, replica_axis)
    by_global = {record["global_rank"]: record for record in records}
    stages, sizes = {}, set()
    for record in records:
        members, rank = record.get("pp_members"), record.get("pp_rank")
        if (
            not isinstance(members, (tuple, list))
            or not members
            or len(set(members)) != len(members)
            or any(type(peer) is not int or peer not in by_global for peer in members)
            or type(rank) is not int
            or not 0 <= rank < len(members)
            or members[rank] != record["global_rank"]
        ):
            raise ValueError("Invalid source pipeline coordinates")
        stages.setdefault(rank, []).append(record)
        sizes.add(len(members))
    if len(sizes) != 1 or set(stages) != set(range(sizes.pop())):
        raise ValueError("Incomplete source pipeline stages")
    shapes = {
        _validate_grid(peers, tuple(record["global_rank"] for record in peers), partition_axis, replica_axis)
        for peers in stages.values()
    }
    if len(shapes) != 1:
        raise ValueError("Source parallel grid differs between pipeline stages")
    for record in records:
        for index, peer in enumerate(record["pp_members"]):
            actual = by_global[peer]
            if (
                tuple(actual["pp_members"]) != tuple(record["pp_members"])
                or actual["pp_rank"] != index
                or any(actual[axis + "_rank"] != record[axis + "_rank"] for axis in (partition_axis, replica_axis))
            ):
                raise ValueError("Pipeline lane mixes source partition or replica coordinates")
    return shapes.pop()


def validate_source_topology(
    records: list[dict],
    actors: tuple[int, ...],
    engines: list[str],
    tp: int,
    *,
    partition_axis: str = "tp",
    replica_axis: str = "dp",
) -> dict:
    """Preserve each engine's baseline source, proven by actual MPU groups.

    The sender maps contiguous groups of ``tp`` Actor slots to an engine.
    Global process ranks and MPU group members need not be sorted.
    """
    if partition_axis == replica_axis or any(
        not isinstance(axis, str) or not axis for axis in (partition_axis, replica_axis)
    ):
        raise ValueError("Distinct source topology axes are required")
    world = len(actors)
    if (
        type(tp) is not int
        or tp <= 0
        or not world
        or len(set(actors)) != world
        or any(type(rank) is not int or rank < 0 for rank in actors)
        or len(records) != world
        or len(engines) * tp != world
        or len(set(engines)) != len(engines)
        or any(not isinstance(engine, str) or not engine for engine in engines)
    ):
        raise ValueError("Invalid source Actor or engine inventory")
    for rank, record in enumerate(records):
        if (
            type(record.get("rank")) is not int
            or record["rank"] != rank
            or type(record.get("global_rank")) is not int
            or record["global_rank"] != actors[rank]
        ):
            raise ValueError("Source gather slot differs from its actual Actor rank")
    shape = _validate_pipeline_grid(records, actors, partition_axis, replica_axis)
    engine_replica = {}
    for index, engine in enumerate(engines):
        owners = records[index * tp : (index + 1) * tp]
        if (
            len({record[replica_axis + "_rank"] for record in owners}) != 1
            or len({record.get("pp_rank", 0) for record in owners}) != 1
        ):
            raise ValueError("An existing inference engine mixes source replicas")
        engine_replica[engine] = owners[0][replica_axis + "_rank"]
    if set(engine_replica.values()) != set(range(shape[1])):
        raise ValueError("Source routing omits a source replica")
    return engine_replica


def dense_replica_rank(record: dict) -> int:
    """Read the validated replica coordinate, retaining CP1 plan support."""
    return record["dp_cp_rank"] if "dp_cp_members" in record else record["dp_rank"]


def validate_dense_topology(records: list[dict], actors: tuple[int, ...], engines: list[str], tp: int) -> dict:
    """A dense source is one TP group at a specific DP and CP coordinate."""
    combined = any("dp_cp_members" in record or "dp_cp_rank" in record for record in records)
    replica_axis = "dp_cp" if combined else "dp"
    routing = validate_source_topology(records, actors, engines, tp, replica_axis=replica_axis)
    if not combined:
        # Older CP1 descriptors omit CP entirely. They cannot describe CP>1.
        if any(
            ("cp_rank" in record or "cp_members" in record)
            and (
                type(record.get("cp_rank")) is not int
                or record["cp_rank"] != 0
                or record.get("cp_members") != [record["global_rank"]]
            )
            for record in records
        ):
            raise ValueError("Context replicas require explicit DP+CP coordinates")
        return routing
    by_global = {record["global_rank"]: record for record in records}
    shapes = set()
    for members in {tuple(record["dp_cp_members"]) for record in records}:
        peers = [by_global[rank] for rank in members]
        shapes.add(_validate_grid(peers, members, "cp", "dp"))
    if len(shapes) != 1:
        raise ValueError("Dense DP and CP sizes differ across TP ranks")
    for record in records:
        if any(
            (by_global[peer]["dp_rank"], by_global[peer]["cp_rank"]) != (record["dp_rank"], record["cp_rank"])
            for peer in list(record["tp_members"]) + list(record.get("pp_members", []))
        ):
            raise ValueError("A dense TP group or pipeline lane mixes DP or CP coordinates")
    return routing


# Source coordinates.


def _describe_axes(sync: Any, axes: tuple) -> dict:
    """Check MPU caches against actual groups without GPU work."""
    mpu = sync.backend._megatron.mpu
    record = dict(rank=sync.rank, global_rank=dist.get_global_rank(sync.group, sync.rank))
    for axis, prefix, configured in axes:
        options = dict(with_context_parallel=True, partial_data_parallel=False) if axis == "dp_cp" else {}
        group = getattr(mpu, f"get_{prefix}_parallel_group")(**options)
        members = list(dist.get_process_group_ranks(group))
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        mpu_rank = getattr(mpu, f"get_{prefix}_parallel_rank")(**options)
        mpu_size = getattr(mpu, f"get_{prefix}_parallel_world_size")(**options)
        if (
            type(mpu_rank) is not int
            or type(mpu_size) is not int
            or mpu_rank != rank
            or mpu_size != size
            or len(members) != size
            or not 0 <= rank < size
            or members[rank] != record["global_rank"]
            or (configured is not None and (type(configured) is not int or configured != size))
        ):
            raise ValueError(f"Source {axis} MPU coordinates differ from the actual process group or configuration")
        record[axis + "_rank"], record[axis + "_members"] = rank, members
    return record


def describe_dense_coordinates(sync: Any) -> dict:
    """Keep DP and CP explicit; dense replicas use the full DP+CP group."""
    args = sync.backend.args
    return _describe_axes(
        sync,
        (
            ("tp", "tensor_model", args.tensor_model_parallel_size),
            ("dp", "data", None),
            ("cp", "context", args.context_parallel_size),
            ("dp_cp", "data", None),
        ),
    )


def describe_source_coordinates(sync: Any) -> dict:
    """Read CPU metadata only; callers converge errors before collectives."""
    args = sync.backend.args
    return describe_dense_coordinates(sync) | _describe_axes(
        sync,
        (
            ("ep", "expert_model", args.expert_model_parallel_size),
            ("edp", "expert_data", None),
            ("etp", "expert_tensor", args.expert_tensor_parallel_size),
            ("pp", "pipeline_model", args.pipeline_model_parallel_size),
        ),
    )


# Expert catalog.


def describe_expert_sources(sources: dict, canonical: dict, dependencies: dict) -> dict:
    """Describe fresh local Bridge outputs before any PP or EP relay."""
    parameters = defaultdict(dict)
    for hf, (source, shape) in canonical.items():
        targets = dependencies.get(hf, ())
        if source not in sources or len(targets) != 1:
            raise ValueError("Expert canonical output has no original source or unique native dependency")
        parameters[next(iter(targets))][hf] = dict(
            source=source, shape=list(shape), dtype=str(sources[source]["dtype"])
        )
    return dict(
        sources={name: dict(shape=list(spec["shape"]), dtype=str(spec["dtype"])) for name, spec in sources.items()},
        parameters=dict(parameters),
    )


# Moe layout.

BLOCK_CAPABILITY = "fused-moe-bf16-ep1-block-v2"


def validate_expert_identity(metadata: Any, *, layers: set[int], experts: int) -> None:
    """Accept native EP1 identity maps using only their existing CPU copies."""
    if metadata is None:
        return
    physical = metadata.physical_to_logical_map_cpu
    logical = metadata.logical_to_all_physical_map_cpu
    if (
        metadata.ep_size != 1
        or type(metadata.num_layers) is not int
        or metadata.num_layers <= 0
        or not layers
        or any(type(layer) is not int or not 0 <= layer < metadata.num_layers for layer in layers)
        or metadata.num_logical_experts != experts
        or metadata.num_physical_experts != experts
        or not isinstance(physical, torch.Tensor)
        or not isinstance(logical, torch.Tensor)
        or physical.device.type != "cpu"
        or logical.device.type != "cpu"
        or physical.dtype not in (torch.int32, torch.int64)
        or logical.dtype not in (torch.int32, torch.int64)
        or physical.shape != (metadata.num_layers, experts)
        or logical.shape != (metadata.num_layers, experts, experts)
    ):
        raise NotImplementedError("Expert shard transfer requires complete CPU EP1 identity maps")
    identity = torch.arange(experts, device="cpu", dtype=physical.dtype).expand(metadata.num_layers, -1)
    invalid = torch.full((), -1, device="cpu", dtype=logical.dtype)
    # torch.equal returns a host boolean here; no GPU readback or synchronization.
    if (
        not torch.equal(physical, identity)
        or not torch.equal(logical[:, :, 0], identity.to(dtype=logical.dtype))
        or not torch.equal(logical[:, :, 1:], invalid.expand_as(logical[:, :, 1:]))
    ):
        raise NotImplementedError("Expert shard transfer does not support reordered or redundant experts")


def validate_native_model(
    model: torch.nn.Module, args: Any, tp_rank: int, layouts: Mapping | None = None
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], int]]:
    """Check the actual receiver model, or the replica with its reload
    metadata."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        FusedMoE,
        UnquantizedFusedMoEMethod,
        get_global_expert_location_metadata,
    )
    from sglang.srt.runtime_context import get_device

    expected_args = dict(
        tp_size=args.tp_size,
        ep_size=1,
        pp_size=1,
        dp_size=1,
        moe_dp_size=1,
        dwdp_size=1,
        moe_runner_backend="triton",
        moe_a2a_backend="none",
        disable_shared_experts_fusion=True,
        enforce_shared_experts_fusion=False,
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location="trivial",
        ep_join_mode=None,
        elastic_ep_backend=None,
        quantization=None,
        modelopt_quant=None,
        kt_weight_path=None,
        speculative_algorithm=None,
        quantize_and_serve=False,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        expert_distribution_recorder_mode=None,
    )
    mismatches = {name: getattr(args, name) for name, value in expected_args.items() if getattr(args, name) != value}
    # ServerArgs preserves raw input (device=None means auto); its resolved
    # device is published separately by SGLang. Require that actual device.
    device = get_device().device
    if (
        type(args.tp_size) is not int
        or args.tp_size <= 0
        or type(tp_rank) is not int
        or not 0 <= tp_rank < args.tp_size
        or device != "cuda"
        or mismatches
    ):
        raise NotImplementedError(
            f"Expert shard transfer requires unquantized TP/EP1 triton: tp_rank={tp_rank}, "
            f"device={device}, mismatched_fields={mismatches}"
        )
    if args.enable_lora or json.loads(args.json_model_override_args) != {}:
        raise NotImplementedError("Expert shard transfer does not support adapters or model overrides")
    params = dict(model.named_parameters(remove_duplicate=False))
    modules = {name: module for name, module in model.named_modules() if isinstance(module, FusedMoE)}
    if not modules:
        raise NotImplementedError("Expert shard transfer requires native FusedMoE modules")
    expected, owners, layer_ids, counts = {}, {}, set(), set()
    for prefix, module in modules.items():
        names = [f"{prefix}.{suffix}" if prefix else suffix for suffix in ("w13_weight", "w2_weight")]
        if any(name not in params for name in names):
            raise NotImplementedError("Native expert gate/up and down parameters must both exist")
        if any(params[name] is not getattr(module, name.rsplit(".", 1)[-1]) for name in names):
            raise NotImplementedError("Native expert parameters differ from their module attributes")
        if layouts is None and any(params[name].device.type != "cuda" for name in names):
            raise NotImplementedError("Native expert receiver parameter is not on CUDA")
        if {name for name in params if not prefix or name.startswith(prefix + ".")} != set(names):
            raise NotImplementedError("Native expert module has extra parameters or aliases")
        if layouts is not None and any(params[name].numel() or params[name].device.type != "cpu" for name in names):
            raise NotImplementedError("Reload metadata is only valid for empty CPU replicas")
        if type(module.layer_id) is not int or module.layer_id < 0 or module.layer_id in layer_ids:
            raise NotImplementedError("Native expert layer IDs must be distinct nonnegative integers")
        layer_ids.add(module.layer_id)
        shape = tuple(layouts[names[0]].shape if layouts is not None else params[names[0]].shape)
        if len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape) or shape[1] % 2:
            raise NotImplementedError("Invalid native gate/up dimensions")
        experts, doubled_width, hidden = shape
        width = doubled_width // 2
        counts.add(experts)
        for name, target_shape in zip(names, (shape, (experts, hidden, width)), strict=True):
            owners[name] = module
            expected[name] = (
                target_shape,
                (math.prod(target_shape[1:]), target_shape[2], 1),
                math.prod(target_shape) * 2,
            )
        if type(module) is not FusedMoE or type(module.quant_method) is not UnquantizedFusedMoEMethod:
            raise NotImplementedError("Expert shard transfer requires the native unquantized FusedMoE loader")
        module_fields = dict(
            use_presharded_weights=False,
            use_triton_kernels=False,
            use_padded_loading=False,
            moe_tp_size=args.tp_size,
            moe_tp_rank=tp_rank,
            moe_ep_size=1,
            _expert_storage_rank=0,
            _num_local_routed=experts,
            _num_global_routed=experts,
            num_local_experts=experts,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size_per_partition=width,
            _has_fused_shared=False,
            num_fused_shared_experts=0,
        )
        if any(getattr(module, name) != value for name, value in module_fields.items()):
            raise NotImplementedError("Expert shard transfer does not support reordered, padded or presharded experts")
        if (
            module.moe_runner_config.is_gated is not True
            or module.quant_config is not None
            or getattr(module, "scheme", None) is not None
            or getattr(module, "weight_load_method", None) is not None
        ):
            raise NotImplementedError("Expert shard transfer requires the plain gate/up layout")
        if getattr(module.quant_method, "load_up_proj_weight_first", False) or getattr(
            module.quant_method, "w13_swiglu_interleaved", False
        ):
            raise NotImplementedError("Expert shard transfer requires contiguous gate before up")
    if len(counts) != 1:
        raise NotImplementedError("Block expert routing requires a common expert count across layers")
    if {name for name in params if ".experts." in name} - expected.keys():
        raise NotImplementedError("Unknown expert parameters are outside the native gated layout")
    validate_expert_identity(get_global_expert_location_metadata(), layers=layer_ids, experts=next(iter(counts)))
    if len({id(params[name]) for name in expected}) != len(expected):
        raise NotImplementedError("Native expert parameters must not alias")
    devices = {params[name].device for name in expected}
    if layouts is None and len(devices) != 1:
        raise NotImplementedError("Native expert receiver parameters must share one CUDA device")
    for name, (shape, stride, nbytes) in expected.items():
        param = params[name]
        if layouts is None and param.device.type != "cuda":
            raise NotImplementedError(f"Expert shard transfer receiver parameter is not on CUDA: {name}")
        layout = layouts[name] if layouts is not None else param
        actual_stride = tuple(layout.stride) if layouts is not None else tuple(param.stride())
        actual_bytes = layout.occupied_nbytes if layouts is not None else param.numel() * param.element_size()
        if (tuple(layout.shape), actual_stride, actual_bytes, param.dtype) != (shape, stride, nbytes, torch.bfloat16):
            raise NotImplementedError(f"Unsupported native expert reload layout: {name}")
        if (
            getattr(param.weight_loader, "__func__", None) is not FusedMoE.weight_loader
            or getattr(param.weight_loader, "__self__", None) is not owners[name]
        ):
            raise NotImplementedError(f"Unsupported native expert loader: {name}")
        if getattr(param, "is_transposed", False) or getattr(param, "_sglang_require_global_experts", False):
            raise NotImplementedError(f"Unsupported native expert parameter attributes: {name}")
        if layouts is None and (
            param.storage_offset() * param.element_size() + nbytes > param.untyped_storage().nbytes()
        ):
            raise NotImplementedError("Native expert range exceeds its backing storage")

    if layouts is None:
        from .replica_overlap import validate_disjoint_ranges

        ranges = {}
        for name, param in params.items():
            if param.device in devices and param.numel():
                span = (
                    1 + sum((size - 1) * stride for size, stride in zip(param.shape, param.stride(), strict=True))
                ) * param.element_size()
                ranges[name] = dict(address=param.data_ptr(), nbytes=span)
        validate_disjoint_ranges(ranges, set(expected), set(ranges) - expected.keys())
    return expected


# Vocab source.
