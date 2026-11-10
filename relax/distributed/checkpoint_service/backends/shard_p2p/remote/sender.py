# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Export current expert, vocabulary and TP shards and retain original transfer
completions."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ..plan import TransferPlan
from .common import atomic_json, identity, named_sources, read_manifest
from .layout import describe_expert_sources, describe_source_coordinates, validate_native_model
from .loader_probe import HfNameMapping
from .plan import (
    RPC_RECEIPT_DELIVERY,
    OwnerBlock,
    ReceiptValidator,
    build_expert_catalog,
    build_expert_routing,
    fingerprint,
    owned_expert_engines,
    partition_expert_plans,
    plan_owner_blocks,
    validate_expert_catalog,
)
from .rpc_receipts import route_receipts


# Source coordinates.


def capture_expert_routing(sync: Any) -> dict:
    """Agree a complete routing catalog before any receiver prepare."""

    def local() -> tuple:
        return (
            describe_source_coordinates(sync),
            [None] * sync.world_size,
            tuple(dist.get_process_group_ranks(sync.group)),
        )

    record, records, actors = sync._phase(local)
    dist.all_gather_object(records, record, group=sync.group)

    def build() -> dict:
        route = build_expert_routing(
            records, actors, sync.engines, sync.backend.args.rollout_num_gpus_per_engine, sync.backend.args.num_experts
        )
        if sync._collective_expert_export and any(len(group) < 2 for group in route["collective_groups"]):
            raise NotImplementedError("Expert AllGather requires at least two engines per expert replica")
        return route

    return sync._phase(build)


def validate_expert_routing(sync: Any) -> None:
    """Reject stale owners before snapshots, receiver control or GPU writes."""
    route = sync._expert_routing
    args = sync.backend.args
    if (
        fingerprint(route) != sync._expert_routing_hash
        or list(route["engine_sources"]) != sync.engines
        or route["source_world_size"] != sync.world_size
        or route["experts"] != args.num_experts
        or len(sync.engines) * args.rollout_num_gpus_per_engine != sync.world_size
        or tuple(dist.get_process_group_ranks(sync.group))
        != tuple(record["global_rank"] for record in route["coordinates"])
        or describe_source_coordinates(sync) != route["coordinates"][sync.rank]
    ):
        raise RuntimeError("Expert source routing changed since qualification")


# Expert catalog.


# Moe blocks.


def pack_canonical_block(block: OwnerBlock, values: Mapping[str, Any]) -> Any:
    """Pack this version's Bridge outputs into one bounded temporary tensor."""
    import torch

    if block.projection not in ("gate_up", "down"):
        raise ValueError("Unknown gated expert projection")
    tensors = [values[name] for name in block.hf_names]
    device = tensors[0].device
    expected = (
        (block.hidden, block.intermediate)
        if block.projection == "down"
        else (
            block.intermediate,
            block.hidden,
        )
    )
    if any(t.dtype != torch.bfloat16 or t.device != device or tuple(t.shape) != expected for t in tensors):
        raise ValueError("Canonical expert value changed shape, dtype or device")
    output = torch.empty(block.canonical_shape, dtype=torch.bfloat16, device=device)
    # stack reads strided inputs directly; flattening each source first would
    # retain another block's worth of temporary copies for noncontiguous data.
    torch.stack(tensors, out=output.view(len(tensors), *expected))
    return output


def copy_owner_block(block: OwnerBlock, source: Any, target: Any, tp_rank: int) -> None:
    """Copy one TP slice directly into a registered writer buffer view."""
    import torch

    if (
        block.projection not in ("gate_up", "down")
        or not 0 <= tp_rank < block.tp_size
        or tuple(source.shape) != block.canonical_shape
        or tuple(target.shape) != block.native_shape
        or source.dtype != torch.bfloat16
        or target.dtype != torch.bfloat16
        or source.device != target.device
        or not target.is_contiguous()
        or source.untyped_storage().data_ptr() == target.untyped_storage().data_ptr()
    ):
        raise ValueError("Invalid or aliased canonical/native expert block")
    for source_slice, target_slice in block.geometry.block_copy_slices(block.projection, len(block.experts), tp_rank):
        target[target_slice].copy_(source[source_slice])


# Moe block direct.


class _ExpertTransferProtocol:
    """Skip EP gather for expert weights; retain P2P for all other
    parameters."""

    def __init__(self, sync: Any, manifest: dict) -> None:
        self.sync = sync
        self.tp_size = sync.backend.args.rollout_num_gpus_per_engine

        def describe_local() -> dict:
            writes = self._local_plan()
            return dict(writes=writes, catalog=getattr(self, "_native_source_catalog", None))

        record = sync._phase(describe_local)
        gathered = [None] * sync.world_size
        dist.all_gather_object(gathered, record, group=sync.group)
        self.plans = {
            f"{engine}-{tp}": dict(source_world_size=sync.world_size, writes=[])
            for engine in sync.engines
            for tp in range(self.tp_size)
        }
        routing = getattr(sync, "_expert_routing", None)
        if routing is not None:
            for plan in self.plans.values():
                # Every target binds the same catalog, including its replica's
                # actual source slots. Per-target fragments cannot bind peers.
                plan["source_routing"] = routing
        self.collective_groups = (
            routing["collective_groups"]
            if routing is not None
            else [[f"{engine}-{tp}" for engine in sync.engines] for tp in range(self.tp_size)]
        )
        for source in gathered:
            for write in source["writes"]:
                self.plans[write["target"]]["writes"].append(write)
        self.target_names = getattr(self, "_native_expert_names", None)
        if not self.target_names or any(
            {write["param"] for write in plan["writes"]} != self.target_names for plan in self.plans.values()
        ):
            raise ValueError("Expert shard transfer must cover every qualified native expert parameter")
        self.plans = TransferPlan.create_all(
            self.plans,
            {key: manifest["params"] for key in self.plans},
            {key: self.target_names for key in self.plans},
            max_bytes=sync.writer.buffers[0].numel(),
        )
        inventories = [source["catalog"] for source in gathered]
        if routing is None or any(inventory is None for inventory in inventories):
            raise ValueError("Expert shard transfer requires every original source catalog and actual routing")
        catalog = build_expert_catalog(
            routing,
            inventories,
            {name: sync.mapping.hf_names_by_param_name[name] for name in self.target_names},
        )
        for key, plan in self.plans.items():
            plan["source_catalog"] = catalog
            validate_expert_catalog(plan, key, manifest["params"], required=True)
        self.collective = sync._collective_expert_export
        for plan in self.plans.values():
            plan["receipt_delivery"] = RPC_RECEIPT_DELIVERY
        if self.collective:
            self.plans = partition_expert_plans(self.plans, manifest["params"], self.collective_groups)
        self.fallback_mapping = HfNameMapping.from_hf_names_by_param_name(
            {name: hf for name, hf in sync.mapping.hf_names_by_param_name.items() if name not in self.target_names}
        )
        self.local_writes = {}
        for write in (write for plan in self.plans.values() for write in plan["writes"]):
            if write["source_rank"] != sync.rank:
                continue
            self.local_writes.setdefault(write["source"], []).append(write)
        self.hashes = {key: fingerprint(plan) for key, plan in self.plans.items()}
        self.published = False
        self.rpc_validators = {key: ReceiptValidator(plan) for key, plan in self.plans.items()}

    def _local_plan(self) -> list[dict]:
        raise NotImplementedError("An expert payload adapter must provide its source plan")

    def agree(self) -> None:
        values = [None] * self.sync.world_size
        dist.all_gather_object(values, fingerprint(self.hashes), group=self.sync.group)
        if len(set(values)) != 1:
            raise RuntimeError("Actor ranks disagree on the MoE direct routing plan")

    def prepare(self, directory: Path, version: int) -> None:
        if not hasattr(self, "initial_version"):
            self.initial_version = version
        self.receipt_version = version
        self.rpc_receipts_active = version != self.initial_version
        self.local_rpc_receipts = self.gathered_rpc_receipts = self.routed_rpc_receipts = None
        if self.sync.rank == 0 and getattr(self.sync, "_snapshot_audit", None) is not None:
            (directory / ".retain-audit").touch(mode=0o600)
        previous = getattr(self, "manifests", {})
        self.rpc_plans_active = self.published
        self.plan_envelopes = None
        if self.rpc_plans_active and (
            self.sync.completed_version != version - 1
            or set(previous) != set(self.plans)
            or set(self.hashes) != set(self.plans)
            or set(self.rpc_validators) != set(self.plans)
            or any(previous[key]["version"] != version - 1 for key in self.plans)
            or any(self.rpc_validators[key].plan_hash != self.hashes[key] for key in self.plans)
        ):
            raise RuntimeError("RPC plan cache requires the preceding completed update")
        self.manifests = {
            key: read_manifest(directory / f"{key}.manifest.json", version, previous.get(key)) for key in self.plans
        }
        for key, manifest in self.manifests.items():
            if manifest["version"] != version or manifest["tp_rank"] != int(key.rsplit("-", 1)[1]):
                raise RuntimeError("Stale or misrouted MoE target manifest")
            self.sync.writer.connect_target(key, manifest)
        if self.rpc_plans_active:
            self.plan_envelopes = {
                key: dict(
                    generation=self.manifests[key]["generation"], version=version, cached_plan_hash=self.hashes[key]
                )
                for key in self.plans
            }
        elif self.sync.rank == 0:
            for key, plan in self.plans.items():
                payload = dict(generation=self.manifests[key]["generation"], version=version)
                payload["plan"] = plan
                atomic_json(directory / f"{key}.plan.json", payload)
        self.published = True
        profile = getattr(self.sync, "_host_profile", None)
        if profile is not None:
            profile["rpc_plan_envelopes"] = len(self.plan_envelopes) if self.rpc_plans_active else 0

    def command_plans(self, engine: str, version: int) -> dict:
        if not self.rpc_plans_active or self.receipt_version != version or self.plan_envelopes is None:
            raise RuntimeError("Plan RPC requires this version's original warm envelopes")
        keys = {f"{engine}-{tp}" for tp in range(self.tp_size)}
        if not keys <= self.plan_envelopes.keys():
            raise RuntimeError("Incomplete engine plan envelopes")
        return {key: self.plan_envelopes[key] for key in sorted(keys)}

    def finish(self, directory: Path, version: int) -> None:
        local = {}
        for key, plan in self.plans.items():
            completed = [write["id"] for write in plan["writes"] if write["source_rank"] == self.sync.rank]
            if getattr(self, "collective", False) and not completed:
                continue
            receipt = dict(
                source_rank=self.sync.rank,
                version=version,
                generation=self.manifests[key]["generation"],
                plan_hash=self.hashes[key],
                writes=completed,
            )
            if getattr(self, "rpc_receipts_active", False):
                if completed:
                    local[key] = receipt
            else:
                atomic_json(directory / f"{key}.source-{self.sync.rank}.done.json", receipt)
        if getattr(self, "rpc_receipts_active", False):
            # This entire method runs in _phase after every writer completed.
            # Serialize and allocate before the next collective, so any local
            # exception converges before another rank enters gather_object.
            self.local_rpc_receipts = json.dumps(local, separators=(",", ":"), allow_nan=False)
            self.gathered_rpc_receipts = [None] * self.sync.world_size if self.sync.rank == 0 else None
            self.receipt_root = dist.get_global_rank(self.sync.group, 0)

    def gather_receipts(self) -> None:
        dist.gather_object(
            self.local_rpc_receipts,
            self.gathered_rpc_receipts,
            dst=self.receipt_root,
            group=self.sync.group,
        )

    def route_receipts(self) -> None:
        if self.sync.rank == 0:
            self.routed_rpc_receipts = route_receipts(
                self.gathered_rpc_receipts,
                self.rpc_validators,
                {key: manifest["generation"] for key, manifest in self.manifests.items()},
                self.receipt_version,
            )

    def commit_receipts(self, engine: str, version: int) -> dict:
        if self.routed_rpc_receipts is None or version != self.receipt_version:
            raise RuntimeError("Commit requires this version's gathered source receipts")
        return {
            key: values for key, values in self.routed_rpc_receipts.items() if key.rsplit("-", 1)[0] == str(engine)
        }

    def replicate(self, directory: Path, version: int, audit: bool) -> None:
        if self.sync.rank == 0 and self.collective:
            self.sync._control("expert_allgather", directory, version, audit)

    def collective_command(self) -> dict:

        keys = [f"{engine}-{tp}" for tp in range(self.tp_size) for engine in self.sync.engines]
        return dict(
            collective_peers={
                key: dict(
                    identity=identity(self.manifests[key]),
                    layout_hash=fingerprint({name: self.manifests[key]["params"][name] for name in self.target_names}),
                )
                for key in keys
            },
            collective_bootstraps={
                group[0]: self.manifests[group[0]]["collective_bootstrap"] for group in self.collective_groups
            },
        )


class MoeBlockDirectShards(_ExpertTransferProtocol):
    """Retain remote shard receipts/fences and replace only its expert send
    packing.

    The planner supports varying dimensions and ownership ranges. This adapter
    requires a validated unquantized native gated-expert loader and an actual
    source ownership catalog.
    """

    def __init__(self, sync: Any, manifest: dict) -> None:
        self._native_layouts = {
            name: {key: layout[key] for key in ("shape", "stride", "dtype", "nbytes")}
            for name, layout in manifest["params"].items()
        }
        super().__init__(sync, manifest)

    @property
    def fallback_skip_sources(self) -> set[str]:
        return set(self.sources)

    def _current_sources(self) -> dict[str, torch.Tensor]:
        backend = self.sync.backend
        values = [
            (name, tensor)
            for name, tensor in named_sources(self.sync)
            if backend._should_sync_tensor(name, tensor) and ".experts." in name
        ]
        if len({name for name, _ in values}) != len(values):
            raise ValueError("Duplicate current Megatron expert source")
        return dict(values)

    def _convert(self, name: str, parameter: torch.Tensor) -> dict[str, torch.Tensor]:
        pairs = self.sync.backend._bridge_converter.convert(name, parameter)
        values = dict(pairs)
        if not values or len(values) != len(pairs) or any(t.dtype != torch.bfloat16 for t in values.values()):
            raise ValueError("Bridge must emit distinct BF16 canonical experts")
        return values

    def _local_plan(self) -> list[dict]:
        sync = self.sync
        self._native_expert_names = set(
            validate_native_model(
                sync.replica._model,
                sync.replica._config.server_args,
                sync.rank % sync.backend.args.rollout_num_gpus_per_engine,
                sync.replica.transfer_buffer_param_layouts,
            )
        )
        self.sources, canonical = {}, {}
        for name, parameter in self._current_sources().items():
            if parameter.dtype != torch.bfloat16:
                raise ValueError("Expert shard transfer requires BF16 expert sources")
            values = self._convert(name, parameter)
            if sum(value.numel() for value in values.values()) != parameter.numel():
                raise ValueError("Bridge changed the expert source byte budget")
            self.sources[name] = dict(
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                hf={hf: tuple(value.shape) for hf, value in values.items()},
            )
            for hf, value in values.items():
                if hf in canonical:
                    raise ValueError("Two local sources emit the same canonical expert")
                canonical[hf] = (name, tuple(value.shape))
        engines = owned_expert_engines(sync._expert_routing, sync.rank, canonical)
        boundaries = set()
        if sync._collective_expert_export:
            experts = sync._expert_routing["experts"]
            for group in sync._expert_routing["collective_groups"]:
                if len(group) < 2 or experts % len(group):
                    raise ValueError("Collective seeds must evenly partition the expert axis")
                boundaries.update(range(experts // len(group), experts, experts // len(group)))
        self.blocks = plan_owner_blocks(
            canonical,
            sync.mapping.param_names_by_hf_name,
            self._native_layouts,
            tp_size=sync.backend.args.rollout_num_gpus_per_engine,
            max_bytes=sync.writer.buffers[0].numel(),
            expert_boundaries=sorted(boundaries),
        )
        self._native_source_catalog = describe_expert_sources(
            self.sources, canonical, sync.mapping.param_names_by_hf_name
        )
        return [
            block.write(rank=sync.rank, target=f"{engine}-{tp}", tp_rank=tp)
            for block in self.blocks
            for engine in engines
            for tp in range(block.tp_size)
        ]

    def send(self) -> int:
        current = self._current_sources()
        if set(current) != set(self.sources) or any(
            tuple(current[name].shape) != spec["shape"] or current[name].dtype != spec["dtype"]
            for name, spec in self.sources.items()
        ):
            raise RuntimeError("Current expert ownership or layout changed")
        expected = {(write["target"], write["id"]) for writes in self.local_writes.values() for write in writes}
        emitted = set()
        writer = self.sync.writer
        index, buffer = writer.acquire()
        cursor, transfers, total = 0, [], 0

        def flush() -> None:
            nonlocal index, buffer, cursor, transfers
            if transfers:
                writer.submit_ranges(index, transfers, self.manifests)
                index, buffer = writer.acquire()
                cursor, transfers = 0, []

        for block in self.blocks:
            values = {}
            for source in block.sources:
                converted = self._convert(source, current[source])
                if {name: tuple(value.shape) for name, value in converted.items()} != self.sources[source]["hf"]:
                    raise RuntimeError("Bridge canonical layout changed during block export")
                if values.keys() & converted.keys():
                    raise RuntimeError("Duplicate canonical block value")
                values.update(converted)
            canonical = pack_canonical_block(block, values)
            block_key = block.write(rank=self.sync.rank, target="unused", tp_rank=0)["source"]
            packed = {}
            for write in self.local_writes[block_key]:
                identity = (write["target"], write["id"])
                if identity in emitted:
                    raise RuntimeError("Duplicate block delivery")
                tp, count = write["tp_rank"], write["row_bytes"]
                if tp not in packed:
                    if cursor + count > buffer.numel():
                        flush()
                        packed = {}
                    view = buffer[cursor : cursor + count]
                    native = view.view(torch.bfloat16).view(block.native_shape)
                    copy_owner_block(block, canonical, native, tp)
                    packed[tp] = view
                    cursor += count
                transfers.append((write, packed[tp]))
                emitted.add(identity)
                total += count
            del canonical, values
        flush()
        writer.finish()
        if emitted != expected:
            raise RuntimeError("Block send did not complete every planned delivery")
        return total
