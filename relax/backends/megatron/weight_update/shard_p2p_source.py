# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Qualified local expert IPC with fresh plans and native non-expert
fallback."""

import re
from collections.abc import Mapping
from math import prod

import torch
import torch.distributed as dist

from relax.backends.sglang.shard_p2p_weight_loader import (
    EXPERT_IPC_LOADER,
    TP_IPC_LOADER,
    expert_shapes,
    infer_expert_layers,
)
from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import group_storage_views
from relax.distributed.checkpoint_service.backends.shard_p2p.layout import ExpertGeometry
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import (
    coalesce_writes,
    region_write,
)
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger

from ..sglang import MultiprocessingSerializer
from .bridge_converter import _normalize_global_name
from .expert_layout import local_expert_parameters


logger = get_logger(__name__)


_EXPERT = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")


class ExpertViewPlanCache:
    """Cache only CPU layout descriptions; rebind and qualify live views.

    Addresses identify alias groups only within one call. Neither addresses,
    tensors nor storage objects survive in this cache. A changed layout must go
    through a fresh native seed instead of silently replacing the plan.
    """

    def __init__(self) -> None:
        self._signature = None
        self._plans = None

    @property
    def initialized(self) -> bool:
        return self._signature is not None

    def bind_sources(self, canonical: Mapping[str, torch.Tensor], *, qualification: tuple = (), **geometry):
        """Requalify fresh allocations without copying the static
        rectangles."""
        sources, views = group_storage_views(canonical)
        schema = []
        for name, tensor in sorted(canonical.items()):
            storage = tensor.untyped_storage()
            schema.append(
                (
                    name,
                    views[name]["source"],
                    str(tensor.dtype),
                    str(tensor.device),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.storage_offset(),
                    storage.nbytes(),
                )
            )
        signature = (tuple(sorted(geometry.items())), qualification, tuple(schema))
        if self._signature is None:
            _, plans = plan_expert_views(canonical, **geometry)
            self._signature, self._plans = signature, plans
        elif signature != self._signature:
            raise ValueError("Expert IPC logical layout changed; native reseeding is required")
        return sources

    def bind(self, canonical: Mapping[str, torch.Tensor], *, qualification: tuple = (), **geometry):
        sources = self.bind_sources(canonical, qualification=qualification, **geometry)
        # Entries contain scalars only. Copy the returned dictionaries so
        # callers cannot mutate the cached plan after its qualification.
        return sources, [[dict(write) for write in plan] for plan in self._plans]


def plan_expert_views(
    canonical: Mapping[str, torch.Tensor],
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    tp_size: int,
    source_rank: int,
    up_proj_first: bool = False,
    source_etp_size: int = 1,
    source_etp_rank: int = 0,
) -> tuple[dict[str, torch.Tensor], list[list[dict]]]:
    """Describe native TP shards without materializing packed expert tensors.

    Inputs must be canonical contiguous BF16 views, optionally sharded along
    the intermediate dimension by ETP. The caller qualifies Bridge's view
    contract and verifies aliasing with live training storage. Returned flat
    views keep source allocations alive only for this transfer.
    """
    if min(num_experts, hidden_size, intermediate_size, tp_size) <= 0 or intermediate_size % tp_size:
        raise ValueError("Unsupported expert geometry")
    if (
        type(source_etp_size) is not int
        or type(source_etp_rank) is not int
        or source_etp_size <= 0
        or intermediate_size % source_etp_size
        or not 0 <= source_etp_rank < source_etp_size
    ):
        raise ValueError("Invalid source expert tensor-parallel coordinate")
    geometry = ExpertGeometry(num_experts, hidden_size, intermediate_size, tp_size, up_proj_first)
    sources, views = group_storage_views(canonical)
    plans = [[] for _ in range(tp_size)]
    for name, tensor in sorted(canonical.items()):
        match = _EXPERT.fullmatch(name)
        if match is None or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError(f"Unsupported canonical expert view: {name}")
        layer, expert, component = match.groups()
        expert = int(expert)
        shape = geometry.component_shape(component, source_etp_size)
        if not 0 <= expert < num_experts or tuple(tensor.shape) != shape:
            raise ValueError(f"Invalid expert index or shape: {name}")
        key = views[name]["source"]
        offset = tensor.storage_offset()
        for tp in range(tp_size):
            target = f"model.layers.{layer}.mlp.experts."
            parameter = target + ("w2_weight" if component == "down" else "w13_weight")
            source_region, target_region = geometry.component_regions(
                expert,
                component,
                tp,
                source_offset=offset,
                source_etp=source_etp_size,
                source_etp_rank=source_etp_rank,
            )
            write = region_write(
                source_region,
                target_region,
                id=f"{source_rank}:{parameter}:{component}:{expert}:{tp}",
                source=key,
                source_rank=source_rank,
                param=parameter,
                component=component,
                target=f"tp-{tp}",
            )
            if write is not None:
                plans[tp].append(write)
    if not sources:
        raise ValueError("Empty local expert plan")
    return sources, [coalesce_writes(writes, allow_strided=True) for writes in plans]


def _validate_replicas(records: list[dict], *, label: str = "Expert") -> None:
    by_rank = {record["rank"]: record for record in records}
    if len(by_rank) != len(records):
        raise ValueError(f"Duplicate {label} replica rank")
    for record in records:
        ranks, replica, _ = record["role"]
        if not ranks or len(set(ranks)) != len(ranks) or not 0 <= replica < len(ranks):
            raise ValueError(f"Invalid {label} replica group")
        if ranks[replica] != record["rank"] or any(rank not in by_rank for rank in ranks):
            raise ValueError(f"{label} replica group is not covered by the Actor ranks")
        for index, rank in enumerate(ranks):
            peer = by_rank[rank]
            if (
                peer["role"][:2] != (ranks, index)
                or peer.get("shard") != record.get("shard")
                or peer["parameters"] != record["parameters"]
            ):
                raise ValueError(f"{label} replicas differ in membership or canonical weights")
        if len(ranks) > 1 and not record["parameters"]:
            raise ValueError(f"{label} replicas require nonempty canonical weight evidence")


def local_expert_plan(
    iterator,
    weights: Mapping[str, torch.Tensor],
    rank: int,
    *,
    plan_cache: ExpertViewPlanCache | None = None,
    sources_only: bool = False,
):
    layout = iterator._direct_expert_layout
    if layout is None or layout.ep_size != 1 or layout.moe_dp_size != 1 or layout.tp_size <= 0:
        raise ValueError("Colocated expert IPC requires a negotiated TP/EP1/DP1 layout")
    canonical, seen, mappings = {}, set(), set()
    for name, parameter in local_expert_parameters(iterator, weights):
        seen.add(name)
        views = iterator._bridge_converter.local_expert_views(name, parameter)
        if any(canonical_name in canonical for canonical_name, _ in views):
            raise ValueError("Duplicate canonical expert view")
        canonical.update(views)
        _, task = iterator._bridge_converter._lookup_task(_normalize_global_name(name))
        mapping = type(task.mapping).__name__
        mappings.add(mapping)
    geometry = dict(
        num_experts=layout.num_experts,
        hidden_size=layout.hidden_size,
        intermediate_size=layout.intermediate_size,
        tp_size=layout.tp_size,
        source_rank=rank,
        up_proj_first=layout.up_proj_first,
    )
    if plan_cache is None:
        if sources_only:
            raise ValueError("Source-only rebinding requires an initialized expert plan cache")
        sources, plans = plan_expert_views(canonical, **geometry)
    elif sources_only:
        sources = plan_cache.bind_sources(
            canonical, qualification=(tuple(sorted(seen)), tuple(sorted(mappings))), **geometry
        )
        plans = None
    else:
        sources, plans = plan_cache.bind(
            canonical, qualification=(tuple(sorted(seen)), tuple(sorted(mappings))), **geometry
        )
    summary = dict(
        local_parameters=len(seen),
        canonical_views=len(canonical),
        flat_allocations=len(sources),
        rectangles=[len(plan) for plan in plans] if plans is not None else None,
        mappings=sorted(mappings),
        up_proj_first=layout.up_proj_first,
    )
    return sources, plans, summary


class MegatronShardP2PSource:
    """Describe qualified training views and rebind each fresh snapshot."""

    def __init__(self, iterator, layout, *, process_group, num_layers: int, qualified: bool) -> None:
        self.iterator, self.layout = iterator, layout
        self.group = process_group
        self.rank = dist.get_rank(process_group)
        self.num_layers, self.qualified = num_layers, qualified
        self.raw_recipe_enabled = True
        self._raw_source_recipe = None

    @property
    def tp_size(self) -> int:
        return self.layout.tp_size

    def describe(self) -> dict:
        layout = self.layout
        if not self.qualified:
            raise ValueError("Expert IPC requires a negotiated local BF16 expert layout")
        self._replica_signature = self.source_role()
        coordinates = self._replica_signature[2]
        self.parallel_sizes = (coordinates[-1][0], coordinates[1][0])
        logger.info(
            "COLOCATE_EXPERT_SOURCE rank=%d edp_ranks=%s replica=%d owner=%s coordinates=%s",
            self.rank,
            self._replica_signature[0],
            self._replica_signature[1],
            self._replica_signature[1] == 0,
            coordinates,
        )
        return dict(
            num_experts=layout.num_experts,
            hidden_size=layout.hidden_size,
            intermediate_size=layout.intermediate_size,
            tp_size=layout.tp_size,
            num_layers=self.num_layers,
            source_world_size=dist.get_world_size(self.group),
        )

    def validate_axes(self) -> None:
        if any(self._replica_signature[2][axis][0] != 1 for axis in (1, 3, 4)):
            raise ValueError("Local expert IPC requires PP1/CP1/ETP1")

    def reset(self) -> None:
        self.raw_recipe_enabled = True
        self._source_plan_cache = ExpertViewPlanCache()
        self._raw_source_recipe = None

    @property
    def raw_recipe_available(self) -> bool:
        return self._replica_signature[1] != 0 or self._raw_source_recipe is not None

    def select_raw_recipe(self, enabled: bool) -> None:
        self.raw_recipe_enabled = enabled
        if not enabled:
            self._raw_source_recipe = None

    def raw_recipe_support(self, weights: Mapping[str, torch.Tensor]) -> str | None:
        from megatron.bridge.models.conversion import param_mapping as bridge

        for name, parameter in local_expert_parameters(self.iterator, weights):
            global_name, task = self.iterator._bridge_converter._lookup_task(_normalize_global_name(name))
            mapping = task.mapping
            if (
                type(mapping) not in (bridge.FusedExpertMapping, bridge.FusedGatedExpertMapping)
                or type(parameter) not in (torch.Tensor, torch.nn.Parameter)
                or parameter.ndim != 2
                or mapping.permute_dims is not None
                or mapping.transpose_on_export
                or getattr(mapping.local_hf_params, "__func__", None)
                is not bridge.MegatronParamMapping.local_hf_params
                or bridge._module_uses_fsdp(task.megatron_module)
            ):
                return "canonical Bridge views are supported; raw mapping is not implemented"
            for spec in mapping.local_hf_param_specs(global_name):
                if (
                    not isinstance(spec, bridge.LocalHFParamSpec)
                    or type(spec.split_count) is not int
                    or spec.split_count <= 0
                    or type(spec.split_index) is not int
                    or not 0 <= spec.split_index < spec.split_count
                    or (
                        spec.split_dim is not None
                        and (type(spec.split_dim) is not int or not -parameter.ndim <= spec.split_dim < parameter.ndim)
                    )
                ):
                    raise ValueError(f"Invalid expert split specification: {name}")
                if (
                    type(spec) is not bridge.LocalHFParamSpec
                    or spec.split_count not in (1, 2)
                    or (spec.split_count == 2 and (spec.split_dim is None or spec.split_dim % parameter.ndim != 0))
                ):
                    return "canonical Bridge views are supported; raw split is not implemented"
        return None

    def replica_record(self, weights: Mapping[str, torch.Tensor]) -> dict:
        from .utils import snapshot_digests

        role = self.source_role()
        if role != self._replica_signature:
            raise ValueError("Expert replica ownership changed; reconnect and reseed")
        parameters = {}
        if len(role[0]) > 1:
            for name, parameter in sorted(local_expert_parameters(self.iterator, weights)):
                views = dict(
                    self.iterator._bridge_converter.local_expert_views(
                        name, parameter, **({"allow_etp": True} if role[2][1][0] > 1 else {})
                    )
                )
                digests = snapshot_digests(views)
                for canonical, tensor in views.items():
                    if canonical in parameters:
                        raise ValueError("Duplicate canonical expert replica parameter")
                    parameters[canonical] = (tuple(tensor.shape), str(tensor.dtype), digests[canonical])
        return dict(
            rank=dist.get_global_rank(self.group, self.rank),
            role=role,
            shard=tuple(role[2][axis] for axis in (0, 1, 3)),
            parameters=parameters,
        )

    @staticmethod
    def validate_replicas(records: list[dict]) -> None:
        _validate_replicas(records)
        logger.info(
            "COLOCATE_EXPERT_REPLICA_AUDIT ranks=%d groups=%d parameter_views=%d exact=True",
            len(records),
            len({record["role"][0] for record in records}),
            sum(len(record["parameters"]) for record in records),
        )

    @staticmethod
    def complete_geometry(plans: list, geometry: dict) -> dict:
        layers = infer_expert_layers([write for owner in plans for write in owner[0]], geometry)
        return dict(geometry, expert_layers=layers)

    @staticmethod
    def target_layouts(geometry: dict) -> dict:
        return {name: dict(nbytes=prod(shape) * 2) for name, shape in expert_shapes(geometry).items()}

    def source_role(self) -> tuple:
        from megatron.core import parallel_state as mpu

        group = mpu.get_expert_data_parallel_group()
        ranks, replica = tuple(dist.get_process_group_ranks(group)), dist.get_rank(group)
        if not ranks or not 0 <= replica < len(ranks) or ranks[replica] != dist.get_global_rank(self.group, self.rank):
            raise ValueError("Expert data-parallel group does not contain this Actor coordinate")
        coordinates = tuple(
            (getattr(mpu, "get_" + axis + "_world_size")(), getattr(mpu, "get_" + axis + "_rank")())
            for axis in (
                "expert_model_parallel",
                "expert_tensor_parallel",
                "tensor_model_parallel",
                "pipeline_model_parallel",
                "context_parallel",
            )
        )
        return ranks, replica, coordinates

    def local_plan(
        self, weights: Mapping[str, torch.Tensor], *, sources_only: bool = False
    ) -> tuple[dict, list[list[dict]] | None]:
        iterator = self.iterator
        use_recipe = getattr(self, "raw_recipe_enabled", False)
        role = self.source_role()
        if role != self._replica_signature:
            raise ValueError("Expert replica ownership changed; reconnect and reseed")
        if role[1] != 0:
            return {}, None if sources_only else [[] for _ in range(self.tp_size)]
        if sources_only and (
            getattr(self, "_source_plan_cache", None) is None or not self._source_plan_cache.initialized
        ):
            raise ValueError("Source-only binding requires a fresh native seed")
        if use_recipe and sources_only and self._raw_source_recipe is not None:
            return self._raw_source_recipe.bind(iterator, weights, self.rank), None
        sources, plans, _ = local_expert_plan(
            iterator,
            weights,
            self.rank,
            plan_cache=getattr(self, "_source_plan_cache", None),
            sources_only=sources_only,
        )
        if use_recipe and not sources_only and self.raw_recipe_support(weights) is None:
            self._raw_source_recipe = RawExpertRecipe.compile(iterator, weights, self.rank, sources)
            logger.info("Colocated raw expert recipe compiled on Actor rank %d", self.rank)
        return sources, plans


def create_shard_p2p_synchronizer(owner):
    from relax.distributed.checkpoint_service.backends.shard_p2p.synchronizer import ShardP2PSynchronizer

    from .utils import validate_native_colocate_devices

    group, layout = get_gloo_group(), owner._direct_expert_layout
    qualified = (
        owner._has_retained_actor_weights()
        and not owner.lora_enabled
        and not owner.quantization_config
        and not owner._route_experts
        and not owner.use_distribute
        and layout is not None
        and (layout.ep_size, layout.moe_dp_size) == (1, 1)
        and layout.tp_size == owner.args.rollout_num_gpus_per_engine
        and layout.tp_size * len(owner.rollout_engines) == dist.get_world_size(group)
        and hasattr(owner._hf_weight_iterator, "get_non_expert_hf_weight_chunks")
        and getattr(owner, "_expert_ipc_receiver_support", False)
    )
    source = None
    if getattr(owner, "_colocate_expert_ipc_enabled", False):
        source = MegatronShardP2PSource(
            owner._hf_weight_iterator,
            layout,
            process_group=group,
            num_layers=owner.args.num_layers,
            qualified=qualified,
        )
    if getattr(owner, "_colocate_tp_ipc_enabled", False):
        from .combined_shard_source import CombinedShardSource

        source = CombinedShardSource(owner, process_group=group, expert=source)
    if source is None:
        raise ValueError("No qualified shard P2P source")
    return ShardP2PSynchronizer(
        source,
        process_group=group,
        engines=owner.rollout_engines,
        serialize=MultiprocessingSerializer.serialize,
        control_loader=TP_IPC_LOADER,
        update_loader=EXPERT_IPC_LOADER,
        validate_devices=validate_native_colocate_devices,
    )


def _identity(tensor: torch.Tensor) -> tuple:
    return str(tensor.device), tensor.untyped_storage().data_ptr()


def _describe(iterator, weights: Mapping[str, torch.Tensor], rank: int, *, audit: bool) -> tuple:
    from megatron.bridge.models.conversion import param_mapping as bridge
    from megatron.core import parallel_state as mpu

    from .bridge_converter import _normalize_global_name

    layout = iterator._direct_expert_layout
    topology = (
        mpu.get_expert_tensor_parallel_world_size(),
        mpu.get_expert_model_parallel_world_size(),
        mpu.get_tensor_model_parallel_world_size(),
        mpu.get_pipeline_model_parallel_world_size(),
        mpu.get_context_parallel_world_size(),
    )
    if layout is None or layout.tp_size <= 0 or (layout.ep_size, layout.moe_dp_size) != (1, 1) or topology[0] != 1:
        raise ValueError("Raw expert recipe requires ETP1 and a negotiated TP/EP1/DP1 layout")
    geometry = (
        rank,
        topology,
        layout.num_experts,
        layout.hidden_size,
        layout.intermediate_size,
        layout.tp_size,
        layout.up_proj_first,
    )
    converter = iterator._bridge_converter
    converter.init_tasks()
    if converter._quantization_config:
        raise ValueError("Raw expert recipe does not support quantization")
    records, canonical_names = {}, set()
    for name, param in sorted(local_expert_parameters(iterator, weights)):
        key = iterator._vanilla_key_map[name]
        if not isinstance(key, str):
            raise ValueError("Raw expert keys must be parameter names")
        global_name, task = converter._lookup_task(_normalize_global_name(name))
        mapping = task.mapping if task is not None else None
        if (
            type(param) not in (torch.Tensor, torch.nn.Parameter)
            or param.dtype != torch.bfloat16
            or param.ndim != 2
            or not param.is_contiguous()
            or type(mapping) not in (bridge.FusedExpertMapping, bridge.FusedGatedExpertMapping)
            or not mapping.is_expert
            or getattr(mapping, "is_adapter", False)
            or mapping.permute_dims is not None
            or mapping.transpose_on_export
            or getattr(mapping.local_hf_params, "__func__", None) is not bridge.MegatronParamMapping.local_hf_params
            or bridge._module_uses_fsdp(task.megatron_module)
        ):
            raise ValueError(f"Unsupported raw expert mapping or storage: {name}")
        shape, stride = tuple(param.shape), tuple(param.stride())
        offset, capacity = param.storage_offset(), param.untyped_storage().nbytes()
        if min(shape) <= 0 or offset < 0 or (offset + param.numel()) * param.element_size() > capacity:
            raise ValueError(f"Raw expert storage is out of bounds: {name}")
        specs = mapping.local_hf_param_specs(global_name)
        sharding = mapping._local_hf_sharding(param, task.megatron_module)
        if not specs or sharding not in (("replicated", None, 1), ("etp", 0, 1), ("etp", 1, 1)):
            raise ValueError(f"Raw expert recipe requires complete unsharded expert views: {name}")
        descriptors = []
        for spec in specs:
            if (
                type(spec) is not bridge.LocalHFParamSpec
                or not isinstance(spec.name, str)
                or spec.name in canonical_names
                or type(spec.split_count) is not int
                or type(spec.split_index) is not int
                or (spec.split_dim is not None and type(spec.split_dim) is not int)
                or not 0 <= spec.split_index < spec.split_count
                or spec.split_count not in (1, 2)
                or (spec.split_count == 2 and (type(spec.split_dim) is not int or spec.split_dim % 2 != 0))
            ):
                raise ValueError(f"Unsupported or duplicate raw expert split: {name}")
            canonical_names.add(spec.name)
            selected_shape = tuple(spec.selected_shape(shape))
            selected_offset = offset + spec.split_index * selected_shape[0] * shape[1]
            descriptors.append(
                (
                    spec.name,
                    spec.split_dim,
                    spec.split_index,
                    spec.split_count,
                    selected_shape,
                    stride,
                    selected_offset,
                )
            )
        if audit:
            views = converter.local_expert_views(name, param)
            actual = [(n, tuple(v.shape), tuple(v.stride()), v.storage_offset(), _identity(v)) for n, v in views]
            expected = [(d[0], d[4], d[5], d[6], _identity(param)) for d in descriptors]
            if actual != expected:
                raise ValueError(f"Raw expert recipe differs from Bridge views: {name}")
        qualification = (global_name, type(mapping).__name__, mapping.megatron_param, mapping.hf_param, sharding)
        if any(not isinstance(value, str) for value in qualification[:4]):
            raise ValueError(f"Unsupported raw expert mapping names: {name}")
        metadata = (
            key,
            qualification,
            str(param.dtype),
            str(param.device),
            shape,
            stride,
            offset,
            capacity,
            tuple(descriptors),
        )
        records[name] = (param, metadata)
    return geometry, records


class RawExpertRecipe:
    """Persist CPU descriptions only; every bind rechecks Bridge and live
    storage.

    Seed plans still come from the native canonical views. Slot representatives
    are parameter names, never pointers: new allocations are allowed, but
    changed aliasing, offsets, mappings or topology require another native
    seed. Returned flat views retain this round's sources until the existing
    receiver ACKs.
    """

    @classmethod
    def compile(cls, iterator, weights: Mapping[str, torch.Tensor], rank: int, sources: Mapping[str, torch.Tensor]):
        geometry, records = _describe(iterator, weights, rank, audit=True)
        for tensor in sources.values():
            capacity = tensor.untyped_storage().nbytes()
            if (
                tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()
                or tensor.storage_offset() != 0
                or capacity % 2
                or tuple(tensor.shape) != (1, capacity // 2)
            ):
                raise ValueError("Expert seed must expose complete BF16 storage from offset zero")
        identities = {_identity(tensor): slot for slot, tensor in sources.items()}
        if len(identities) != len(sources):
            raise ValueError("Expert seed contains duplicate storage slots")
        owners = {}
        for name, (tensor, _) in records.items():
            identity = _identity(tensor)
            if identity not in identities:
                raise ValueError("Expert recipe storage is absent from the native seed")
            owners.setdefault(identities[identity], name)
        if owners.keys() != sources.keys():
            raise ValueError("Expert recipe does not cover the native seed storage")
        recipe = cls()
        recipe._slot_owners = tuple(sorted(owners.items()))
        recipe._signature = recipe._schema(geometry, records)
        return recipe

    def _schema(self, geometry: tuple, records: dict) -> tuple:
        identities = {_identity(records[name][0]): slot for slot, name in self._slot_owners}
        if len(identities) != len(self._slot_owners):
            raise ValueError("Raw expert storage aliasing changed; native reseeding is required")
        schema = []
        for name, (tensor, metadata) in records.items():
            slot = identities.get(_identity(tensor))
            if slot is None:
                raise ValueError("Raw expert storage aliasing changed; native reseeding is required")
            schema.append((name, slot, metadata))
        return geometry, tuple(schema)

    def bind(self, iterator, weights: Mapping[str, torch.Tensor], rank: int) -> dict[str, torch.Tensor]:
        geometry, records = _describe(iterator, weights, rank, audit=False)
        if (
            any(name not in records for _, name in self._slot_owners)
            or self._schema(geometry, records) != self._signature
        ):
            raise ValueError("Raw expert layout changed; native reseeding is required")
        sources = {}
        for slot, name in self._slot_owners:
            tensor = records[name][0]
            storage = tensor.untyped_storage()
            count = storage.nbytes() // tensor.element_size()
            sources[slot] = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
                storage, 0, (1, count), (count, 1)
            )
        return sources
