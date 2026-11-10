# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shard transfer sessions, cold delivery negotiation and receiver
completion."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4

import torch
import torch.distributed as dist

from relax.utils.logging_utils import get_logger

from .common import worker_identity
from .ipc import export_tensor
from .plan import TransferPlan, coalesce_writes, fingerprint, validate_coverage


logger = get_logger(__name__)


@dataclass
class _PendingUpdate:
    """One version's live storage and submitted readers; never cached."""

    version: int
    sources: Mapping[str, torch.Tensor] | None
    refs: list = field(default_factory=list)
    error: str | None = None


class ShardP2PSynchronizer:
    """Own session identity, completed version and failure isolation.

    prepare() negotiates an address-free CPU plan and receiver identity.
    start()/finish() bind one fresh snapshot, dispatch, await ACKs and release
    its storage. C00 independently waits for the CPU backup before releasing
    its own reference; disabling backup overlap never disables reader safety.
    """

    def __init__(
        self,
        source,
        *,
        process_group,
        engines,
        serialize,
        control_loader: str,
        update_loader: str,
        validate_devices,
    ) -> None:
        self.source, self.group, self.engines = source, process_group, engines
        self.serialize, self.control_loader, self.update_loader = serialize, control_loader, update_loader
        self.validate_devices = validate_devices
        self.rank = dist.get_rank(self.group)
        self.needs_seed, self.failed = True, False
        self.completed_version = None
        self._pending = None
        self.geometry = self._phase(source.describe)
        self.context_parallel_size, self.expert_tensor_parallel_size = source.parallel_sizes
        self._agree_parallel_sizes()
        self._phase(source.validate_axes)
        session = [uuid4().hex if self.rank == 0 else None]
        dist.broadcast_object_list(session, src=0, group=self.group)
        self.session = session[0]

    @property
    def retained_sources(self) -> Mapping[str, torch.Tensor] | None:
        return self._pending.sources if self._pending is not None else None

    def _wait_for_receivers(self, refs: list, *, error: str | None = None) -> list:
        import ray

        try:
            results = ray.get(refs) if refs else []
            for result in results:
                if isinstance(result, Mapping) and result.get("success") is False:
                    raise RuntimeError(f"SGLang rejected weights: {result}")
        except Exception as exc:
            error = str(exc)
            self._drain_receivers(refs)
        failed = torch.tensor([int(error is not None)], dtype=torch.int32, device="cpu")
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.group)
        if failed.item():
            errors = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(errors, error, group=self.group)
            raise RuntimeError(f"Shard P2P weight load failed: {errors}")
        return results

    @staticmethod
    def _drain_receivers(refs: list) -> None:
        import ray

        for ref in refs:
            try:
                ray.get(ref)
            except Exception:
                pass  # A prior error is reported only after all submitted readers drain.

    def _audit_expert_replicas(self, weights: Mapping[str, torch.Tensor]) -> None:
        records = self._gather(self._phase(lambda: self.source.replica_record(weights)))
        self._phase(lambda: self.source.validate_replicas(records) if self.rank == 0 else None)

    @property
    def tp_size(self) -> int:
        return self.geometry["tp_size"]

    @property
    def engine_count(self) -> int:
        return len(self.engines)

    def _agree_parallel_sizes(self) -> None:
        """Agree once before branching into CP or ETP audit collectives."""
        sizes = self._gather((self.context_parallel_size, self.expert_tensor_parallel_size))

        def check():
            if self.rank == 0 and (
                any(type(size) is not int or size < 1 for pair in sizes for size in pair) or len(set(sizes)) != 1
            ):
                raise ValueError("Actor context/expert tensor-parallel sizes disagree")

        self._phase(check)

    def _phase(self, operation):
        value, error = None, None
        try:
            if self.failed:
                raise RuntimeError("Expert IPC session failed; source storage remains retained")
            value = operation()
        except Exception as exc:
            error = str(exc)
        try:
            self._wait_for_receivers([], error=error)
        except Exception:
            self.failed = True
            raise
        return value

    def _gather(self, value):
        output = [None] * self.geometry["source_world_size"] if self.rank == 0 else None
        dist.gather_object(value, output, dst=0, group=self.group)
        return output

    def _dispatch(self, commands, *, engines=None) -> tuple[list, str | None]:
        refs, error = [], None
        if self.rank == 0:
            try:
                for engine, pair in zip(self.engines if engines is None else engines, commands, strict=True):
                    serialized = [
                        self.serialize([(json.dumps(command, separators=(",", ":")), torch.empty(0))], output_str=True)
                        for command in pair
                    ]
                    refs.append(
                        engine.update_weights_from_tensor.remote(
                            serialized_named_tensors=serialized,
                            load_format=self.control_loader
                            if pair[0]["op"] in ("describe_tp", "prepare", "seed")
                            else self.update_loader,
                            flush_cache=False,
                        )
                    )
            except Exception as exc:
                error = str(exc)
        return refs, error

    def _fanout(self, commands, *, engines=None) -> list:
        refs, error = self._dispatch(commands, engines=engines)
        try:
            return self._wait_for_receivers(refs, error=error)
        except Exception:
            self.failed = True
            raise

    def invalidate_receivers(self, engines=None) -> None:
        if self.failed or self._pending is not None:
            raise RuntimeError("Cannot reconnect an incomplete expert IPC transfer")
        if engines is not None:
            self.engines = engines
        self.needs_seed = True

    def prepare(self, weights: Mapping[str, torch.Tensor], *, version: int, audit_native: bool) -> None:
        """Build the first direct plan inside an active weight update."""
        self._audit_expert_replicas(weights)
        self.source.reset()
        if hasattr(self.source, "prepare_targets"):
            self.source.prepare_targets(self, weights, version=version)
        self.disabled = bool(getattr(self.source, "empty", False))
        if self.disabled:
            self.needs_seed = False
            return
        sources, plans = self._phase(lambda: self.source.local_plan(weights))
        raw = [None] * self.geometry["source_world_size"]
        local_raw = self.source.raw_recipe_available
        dist.all_gather_object(raw, local_raw, group=self.group)
        if not all(raw):
            self.source.select_raw_recipe(False)
            if self.rank == 0:
                logger.info("ShardP2P source recipe=canonical; raw mapping unsupported on an owner")

        def endpoint():
            if not sources:
                self._source_device = None
                return None
            devices = {value.device for value in sources.values()}
            if len(devices) != 1:
                raise ValueError("Expert source endpoint requires one actual source device")
            self._source_device = next(iter(devices))
            return worker_identity(self.session, self._source_device)

        endpoints = self._gather(self._phase(endpoint))
        sources = None
        gathered = self._gather(plans)
        layers = [
            self._phase(lambda: self.source.complete_geometry(gathered, self.geometry) if self.rank == 0 else None)
        ]
        # Reject an invalid logical inventory collectively before probing
        # devices. EDP non-owners still participate in physical reuse checks;
        # the expert-only source catalog omits their device identities.
        self._actor_endpoints = self._gather(self._phase(lambda: self._actor_endpoint(weights)))
        dist.broadcast_object_list(layers, src=0, group=self.group)
        self.geometry = layers[0]
        geometries = (
            self.source.geometries(self.geometry)
            if hasattr(self.source, "geometries")
            else [self.geometry] * self.tp_size
        )

        def collect():
            if self.rank != 0:
                return None
            result = []
            for tp_rank in range(self.tp_size):
                layouts = self.source.target_layouts(geometries[tp_rank])
                writes = []
                for rank, owner in enumerate(gathered):
                    if any(
                        write["source_rank"] != rank or write["target"] != f"tp-{tp_rank}" for write in owner[tp_rank]
                    ):
                        raise ValueError("Expert plan has a misrouted source or target")
                    writes.extend(owner[tp_rank])
                if validate_coverage(writes, layouts, source_world_size=self.geometry["source_world_size"]) != set(
                    layouts
                ):
                    raise ValueError("Expert plan must cover every native parameter")
                result.append(writes)
            return result

        # Keep both original union checks. Source rank is a merge boundary,
        # so the final coalescing pass can run on each owner without changing
        # IDs or owner-concatenation order, including non-idempotent plans.
        self._phase(collect)
        plans = self._phase(lambda: [coalesce_writes(plan) for plan in plans])
        gathered = self._gather(plans)
        prepared_writes = self._phase(collect)
        delivery_builders = None

        def prepare():
            nonlocal delivery_builders
            if self.rank != 0:
                return None
            self._plan_hashes = [
                fingerprint(dict(geometry=geometry, writes=plan))
                for geometry, plan in zip(geometries, prepared_writes, strict=True)
            ]
            source_endpoints = {rank: endpoint for rank, endpoint in enumerate(endpoints) if endpoint is not None}
            delivery_builders = [
                TransferPlan(f"tp-{tp_rank}", self.geometry["source_world_size"], tuple(plan)).bind_sources(
                    source_endpoints
                )
                for tp_rank, plan in enumerate(prepared_writes)
            ]
            commands = [
                [
                    dict(
                        op="seed",
                        session=self.session,
                        engine=engine,
                        tp_rank=tp_rank,
                        version=version,
                        previous_version=self.completed_version,
                        nonce=uuid4().hex,
                        geometry=geometries[tp_rank],
                        writes=prepared_writes[tp_rank],
                        plan_hash=self._plan_hashes[tp_rank],
                        batched=True,
                        cache_plan=True,
                        audit_native=audit_native,
                    )
                    for tp_rank in range(self.tp_size)
                ]
                for engine in range(self.engine_count)
            ]
            return commands

        commands = self._phase(prepare)
        self._negotiate_delivery(commands, endpoints, delivery_builders)
        self._fanout(commands)
        self.needs_seed = False

    def _actor_endpoint(self, weights: Mapping[str, torch.Tensor]) -> dict:
        device = torch.device("cuda", torch.cuda.current_device())
        if any(value.is_cuda and value.device != device for value in weights.values()):
            raise ValueError("Actor weights differ from their current CUDA device")
        return worker_identity(self.session, device)

    def _negotiate_delivery(self, commands, endpoints, delivery_builders) -> None:
        """Negotiate actual worker locality through the loader RPC response."""

        def prepare():
            if self.rank != 0:
                return None
            sources = [dict(rank=rank, endpoint=value) for rank, value in enumerate(endpoints) if value is not None]
            return [
                [
                    dict(
                        {key: value for key, value in command.items() if key != "tp"},
                        op="prepare",
                        sources=sources,
                    )
                    for command in pair
                ]
                for pair in commands
            ]

        probes = self._phase(prepare)
        responses = self._fanout(probes)

        def read():
            if self.rank != 0:
                return None
            bindings, needed = [], [[] for _ in range(self.geometry["source_world_size"])]
            for pair, response in zip(commands, responses, strict=True):
                if not isinstance(response, Mapping) or response.get("success") is not True:
                    raise ValueError("Missing successful expert prepare response")
                replies = json.loads(response["message"])["shard_p2p_receivers"]
                for command, reply in zip(pair, replies, strict=True):
                    tp_rank = command["tp_rank"]
                    result = dict(
                        logical_plan_hash=self._plan_hashes[tp_rank],
                        delivery=delivery_builders[tp_rank](reply["receiver"]),
                    )
                    binding = dict(result, plan_hash=fingerprint(result))
                    identity = {key: command[key] for key in ("session", "engine", "tp_rank", "version", "nonce")}
                    if reply != dict(
                        identity=identity,
                        receiver=binding["delivery"]["receiver"],
                        logical_plan_hash=binding["logical_plan_hash"],
                        plan_hash=binding["plan_hash"],
                    ):
                        raise ValueError("Expert receiver feedback differs from the actual source delivery")
                    command["delivery"] = binding
                    bindings.append(binding)
                    if binding["delivery"]["routes"]["nixl"]:
                        raise ValueError("Local expert IPC cannot import remote source allocations")
                    local = set(binding["delivery"]["routes"]["cuda_ipc"])
                    by_source = [set() for _ in needed]
                    for write in command["writes"]:
                        if write["id"] in local:
                            by_source[write["source_rank"]].add(write["source"])
                    for slots, keys in zip(needed, by_source, strict=True):
                        slots.append(frozenset(keys))
            width = len(commands[0])
            self.validate_devices(
                self._actor_endpoints,
                [
                    [binding["delivery"]["receiver"] for binding in bindings[start : start + width]]
                    for start in range(0, len(bindings), width)
                ],
            )
            self._receiver_plan_hashes = tuple(binding["plan_hash"] for binding in bindings)
            return needed

        needed = [self._phase(read)]
        dist.broadcast_object_list(needed, src=0, group=self.group)
        self._needed_keys_by_slot = tuple(needed[0][self.rank])

    def update(self, weights: Mapping[str, torch.Tensor], *, version: int) -> None:
        if getattr(self, "disabled", False):

            def complete_native():
                if self.needs_seed or self._pending is not None:
                    raise RuntimeError("Native fallback requires a complete seed")
                if self.completed_version is not None and version != self.completed_version + 1:
                    raise ValueError("Native fallback version did not advance once")
                self.completed_version = version

            self._phase(complete_native)
            return
        try:
            self.start(weights, version=version)
            self.finish(version=version)
        except Exception:
            self.abort()
            raise

    def start(self, weights: Mapping[str, torch.Tensor], *, version: int) -> None:
        def preflight():
            if self._pending is not None:
                raise RuntimeError("Expert IPC update is already in flight")
            if self.needs_seed:
                raise RuntimeError("Expert IPC requires a fresh native seed")

        def bind():
            preflight()
            sources, plans = self.source.local_plan(weights, sources_only=True)
            if {value.device for value in sources.values()} != (
                set() if self._source_device is None else {self._source_device}
            ):
                raise ValueError("Expert source device changed; native reseeding is required")
            self._pending = _PendingUpdate(version, sources)
            exports = {
                engine * self.tp_size + tp: {
                    key: export_tensor(value)
                    for key, value in sources.items()
                    if key in self._needed_keys_by_slot[engine * self.tp_size + tp]
                }
                for engine in range(self.engine_count)
                for tp in range(self.tp_size)
            }

            return exports

        exports = self._phase(bind)
        gathered = self._gather(exports)

        def prepare():
            if self.rank != 0:
                return None
            commands = [
                [
                    dict(
                        op="apply",
                        session=self.session,
                        engine=engine,
                        tp_rank=tp_rank,
                        version=version,
                        nonce=uuid4().hex,
                        plan_hash=self._plan_hashes[tp_rank],
                        delivery_hash=self._receiver_plan_hashes[engine * self.tp_size + tp_rank],
                        bindings=[
                            dict(source_rank=rank, source=key, payload=payload)
                            for rank, owner in enumerate(gathered)
                            for key, payload in owner[engine * self.tp_size + tp_rank].items()
                        ],
                    )
                    for tp_rank in range(self.tp_size)
                ]
                for engine in range(self.engine_count)
            ]

            return commands

        commands = self._phase(prepare)
        self._pending.refs, self._pending.error = self._dispatch(commands)

    def _drain_pending(self) -> None:
        # This is intentionally local: a converter failure can leave peer
        # Actors inside a different collective, so abort must not rendezvous.
        pending = getattr(self, "_pending", None)
        if pending is not None:
            self._drain_receivers(pending.refs)
            pending.refs.clear()

    def abort(self) -> None:
        """Drain local readers and fail closed; retain source storage for
        teardown."""
        if getattr(self, "_pending", None) is not None:
            self.failed = True
            self._drain_pending()

    def finish(self, *, version: int) -> None:
        pending = getattr(self, "_pending", None)
        error = None
        if self.failed:
            error = "Expert IPC session failed; source storage remains retained"
        elif pending is None or pending.version != version:
            error = "Expert IPC finish has no matching pending version"
        else:
            error = pending.error
        try:
            # Validate and wait in one shared phase. The ordinary synchronous
            # path retains its original number of error-agreement collectives.
            self._wait_for_receivers(pending.refs if pending is not None else [], error=error)
        except Exception:
            self.failed = True
            self.abort()
            raise
        pending.sources = None
        self._phase(torch.cuda.ipc_collect)
        self.completed_version = pending.version
        self._pending = None
