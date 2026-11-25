# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shard transfer sessions, cold delivery negotiation and receiver
completion."""

import json
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from uuid import uuid4

import torch
import torch.distributed as dist

from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger

from .common import validate_worker_identity, worker_identity
from .ipc import export_tensor
from .plan import TransferPlan, coalesce_writes, fingerprint, validate_coverage


logger = get_logger(__name__)


def _compile_receiver_plans(slots: list[tuple], source_endpoints: dict, source_world_size: int) -> list[tuple]:
    """Compile scalar snapshots only; never capture an adapter, tensor or
    PG."""
    result, shared = [], {}
    for tp_rank, geometry, layouts, writes, localized in slots:
        if localized or tp_rank not in shared:
            if localized:
                writes = coalesce_writes(writes)
            if validate_coverage(writes, layouts, source_world_size=source_world_size) != set(layouts):
                raise ValueError("Receiver plan must cover every native parameter")
            builder = TransferPlan(f"tp-{tp_rank}", source_world_size, tuple(writes)).bind_sources(
                source_endpoints, geometry=geometry
            )
            value = (geometry, writes, builder.geometry_hash, builder)
            if not localized:
                shared[tp_rank] = value
        else:
            value = shared[tp_rank]
        result.append(value)
    return result


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
        agent_future: Future | None = None,
    ) -> None:
        self.source, self.group, self.engines = source, process_group, engines
        self.serialize, self.control_loader, self.update_loader = serialize, control_loader, update_loader
        self.validate_devices = validate_devices
        self._agent_future = agent_future
        self.rank = dist.get_rank(self.group)
        self.needs_seed, self.failed = True, False
        self.completed_version = None
        self._pending = None
        self._prefetched_version = None
        self._plan_future = None
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
        from relax.backends.sglang.weight_loader import COLOCATE_NIXL_LOADER

        def encode(command: dict) -> str:
            if command["op"] == "seed" and "delivery" in command:
                command = {**command, "prepared_delivery_hash": command["delivery"]["plan_hash"]}
                for key in ("geometry", "writes", "delivery"):
                    del command[key]
            return self.serialize([(json.dumps(command, separators=(",", ":")), torch.empty(0))], output_str=True)

        refs, error = [], None
        if self.rank == 0:
            try:
                for engine, pair in zip(self.engines if engines is None else engines, commands, strict=True):
                    serialized = [encode(command) for command in pair]
                    refs.append(
                        engine.update_weights_from_tensor.remote(
                            serialized_named_tensors=serialized,
                            load_format=(
                                COLOCATE_NIXL_LOADER
                                if pair[0]["op"] == "bind_epoch" or "epoch_generation" in pair[0]
                                else self.control_loader
                                if pair[0]["op"] in ("describe_tp", "prepare", "seed")
                                else self.update_loader
                            ),
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
        if getattr(self, "_plan_future", None) is not None:
            # A running task owns scalar snapshots only and cannot publish state.
            self._plan_future.cancel()
        self._plan_future = None
        self._prefetched_version = None
        self.needs_seed = True

    def _decode_tp_replies(self, requests, responses) -> list[dict] | None:
        if self.rank != 0:
            return None
        replies = []
        self._described_receivers = []
        identity = ("session", "engine", "version", "nonce", "description_hash", "tp_rank")
        for pair, response in zip(requests, responses, strict=True):
            if not isinstance(response, dict) or response.get("success") is not True:
                raise ValueError("Missing ordinary TP layout response")
            values = json.loads(response["message"])["shard_p2p_tp_targets"]
            if len(values) != self.tp_size:
                raise ValueError("Incomplete ordinary TP target inventory")
            for command, reply in zip(pair, values, strict=True):
                if any(reply.get(key) != command[key] for key in identity):
                    raise ValueError("Misrouted ordinary TP layout reply")
                receiver = reply.get("receiver") if command.get("describe_source_locality") is True else None
                if receiver is not None:
                    validate_worker_identity(receiver)
                self._described_receivers.append(receiver)
            replies.append(values)

        def payload(values):
            return [
                {key: value for key, value in reply.items() if key not in (*identity, "receiver")} for reply in values
            ]

        if any(payload(pair) != payload(replies[0]) for pair in replies[1:]):
            raise ValueError("Inference engines disagree on ordinary TP geometry")
        return replies[0]

    def _prepare_targets(self, weights: Mapping[str, torch.Tensor], version: int) -> None:
        """Negotiate ordinary TP layouts before building the shared IPC
        plan."""
        source = self.source
        records = self._gather(self._phase(lambda: source.target_catalogue(weights)))
        description = self._phase(lambda: source.target_description(records))

        def commands():
            if self.rank != 0:
                return None
            digest = fingerprint(description)
            return [
                [
                    dict(
                        op="describe_tp",
                        session=self.session,
                        engine=engine,
                        tp_rank=tp,
                        version=version,
                        nonce=uuid4().hex,
                        geometry=self.geometry,
                        description=description,
                        description_hash=digest,
                        describe_source_locality=True,
                    )
                    for tp in range(self.tp_size)
                ]
                for engine in range(self.engine_count)
            ]

        requests = self._phase(commands)
        responses = self._fanout(requests)
        replies = self._phase(lambda: self._decode_tp_replies(requests, responses))
        selection = [self._phase(lambda: source.select_targets(description, replies, engine_count=self.engine_count))]
        dist.broadcast_object_list(selection, src=0, group=self.group)
        self._phase(lambda: source.apply_targets(selection[0], weights))

    def prepare(
        self, weights: Mapping[str, torch.Tensor], *, version: int, audit_native: bool, prefetch: bool = False
    ) -> None:
        """Compile layouts early; publish bindings only in the real update."""
        # All adapters share this numerical-audit gate; layout and lifetime
        # validation remain unconditional.
        audit_native = bool(audit_native and Envs.RELAX_P2P_WEIGHT_CONSISTENCY_CHECK)
        if prefetch and (audit_native or Envs.RELAX_P2P_WEIGHT_CONSISTENCY_CHECK):
            raise ValueError("Speculative planning requires unaudited, nonpersistent targets")
        prefetched_version = getattr(self, "_prefetched_version", None)
        if (
            prefetched_version is not None
            and prefetched_version == version
            and not prefetch
            and not audit_native
            and not Envs.RELAX_P2P_WEIGHT_CONSISTENCY_CHECK
        ):
            result = self._phase(lambda: self._plan_future.result() if self.rank == 0 else None)

            def changed():
                try:
                    sources, _ = self.source.local_plan(weights, sources_only=True)
                except ValueError:
                    return True
                schema = {
                    name: (tuple(value.shape), tuple(value.stride()), str(value.dtype))
                    for name, value in sources.items()
                }
                return schema != self._source_schema or any(
                    value.device != self._source_device for value in sources.values()
                )

            stale = torch.tensor([int(self._phase(changed))], dtype=torch.int32, device="cpu")
            dist.all_reduce(stale, op=dist.ReduceOp.MAX, group=self.group)
            self._plan_future = None
            self._prefetched_version = None
            if not stale.item():
                self._finish_preparation(result, version=version, audit_native=audit_native)
                return
        elif prefetched_version is not None:
            if self._plan_future is not None:
                self._plan_future.cancel()
            self._plan_future = None
            self._prefetched_version = None
        self._audit_expert_replicas(weights)
        self.source.reset()
        self._actor_endpoints = self._gather(self._phase(lambda: self._actor_endpoint(weights)))
        if getattr(self.source, "has_ordinary", False):
            self._prepare_targets(weights, version)
        self.disabled = bool(getattr(self.source, "empty", False))
        if self.disabled:
            if not prefetch:
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
            self._source_schema = {
                name: (tuple(value.shape), tuple(value.stride()), str(value.dtype)) for name, value in sources.items()
            }
            if not sources:
                self._source_device = None
                return None
            devices = {value.device for value in sources.values()}
            if len(devices) != 1:
                raise ValueError("Expert source endpoint requires one actual source device")
            self._source_device = next(iter(devices))
            return worker_identity(self.session, self._source_device)

        endpoints = self._gather(self._phase(endpoint))
        self._prepared_endpoints = endpoints
        sources = None
        gathered = self._gather(plans)
        layers = [
            self._phase(lambda: self.source.complete_geometry(gathered, self.geometry) if self.rank == 0 else None)
        ]
        # EDP non-owners also participate in physical reuse checks; the
        # expert-only source catalog omits their device identities.
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

        def snapshot():
            if self.rank != 0:
                return None
            slots = []
            source_endpoints = {rank: endpoint for rank, endpoint in enumerate(endpoints) if endpoint is not None}
            for engine in range(self.engine_count):
                for tp_rank, writes in enumerate(prepared_writes):
                    slot = engine * self.tp_size + tp_rank
                    receiver = self._described_receivers[slot] if getattr(self.source, "has_ordinary", False) else None
                    localized = receiver is not None and hasattr(self.source, "localize_plan")
                    if localized:
                        writes = self.source.localize_plan(writes, receiver, self._actor_endpoints)
                    slots.append(
                        (
                            tp_rank,
                            geometries[tp_rank],
                            self.source.target_layouts(geometries[tp_rank]),
                            writes,
                            localized,
                        )
                    )
            return slots, source_endpoints, self.geometry["source_world_size"]

        inputs = self._phase(snapshot)
        if prefetch:

            def submit():
                if self.rank != 0:
                    return None
                pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p2p-cold-plan")
                future = pool.submit(_compile_receiver_plans, *inputs)
                pool.shutdown(wait=False)
                return future

            self._plan_future = self._phase(submit)
            self._prefetched_version = version
            return
        result = self._compile_cold_snapshot(inputs)
        self._finish_preparation(result, version=version, audit_native=audit_native)

    def _compile_cold_snapshot(self, inputs: tuple | None) -> list[tuple] | None:
        """Share heavy localized CPU plans over the existing Actor group."""
        world = dist.get_world_size(self.group)
        if world <= 1 or dist.get_backend(self.group) != "gloo":
            return self._phase(lambda: _compile_receiver_plans(*inputs) if self.rank == 0 else None)

        def parallel_slot_count() -> int:
            if self.rank != 0:
                return 0
            slots = inputs[0]
            # Object collectives pad to their largest payload. Reserve them
            # for heavy plans with work for every rank, as in the replay.
            if (
                len(slots) >= world
                and all(slot[4] for slot in slots)
                and sum(len(slot[3]) for slot in slots) >= 240_000
            ):
                return len(slots)
            return 0

        root = dist.get_global_rank(self.group, 0)
        decision = [self._phase(parallel_slot_count)]
        dist.broadcast_object_list(decision, src=root, group=self.group)
        slot_count = decision[0]
        if not slot_count:
            return self._phase(lambda: _compile_receiver_plans(*inputs) if self.rank == 0 else None)

        def assign() -> list[tuple] | None:
            if self.rank != 0:
                return None
            slots, endpoints, source_world = inputs
            indexed = list(enumerate(slots))
            return [(indexed[rank::world], endpoints, source_world) for rank in range(world)]

        assignments = self._phase(assign)
        own = [None]
        dist.scatter_object_list(own, assignments, src=root, group=self.group)

        def compile_assigned() -> list[tuple]:
            indexed, endpoints, source_world = own[0]
            plans = _compile_receiver_plans([slot for _, slot in indexed], endpoints, source_world)
            return [(index, plan) for (index, _), plan in zip(indexed, plans, strict=True)]

        compiled = self._phase(compile_assigned)
        gathered = [None] * world if self.rank == 0 else None
        dist.gather_object(compiled, gathered, dst=root, group=self.group)

        def restore_order() -> list[tuple] | None:
            if self.rank != 0:
                return None
            ordered = sorted((entry for worker in gathered for entry in worker), key=lambda entry: entry[0])
            if [index for index, _ in ordered] != list(range(slot_count)):
                raise ValueError("Parallel compilation did not return every receiver slot exactly once")
            return [plan for _, plan in ordered]

        return self._phase(restore_order)

    def _finish_preparation(self, plans, *, version: int, audit_native: bool) -> None:
        delivery_builders = None

        def commands():
            nonlocal delivery_builders
            if self.rank != 0:
                return None
            self._plan_hashes = [value[2] for value in plans]
            delivery_builders = [value[3] for value in plans]
            return [
                [
                    dict(
                        op="seed",
                        session=self.session,
                        engine=engine,
                        tp_rank=tp_rank,
                        version=version,
                        previous_version=self.completed_version,
                        nonce=uuid4().hex,
                        geometry=plans[engine * self.tp_size + tp_rank][0],
                        writes=plans[engine * self.tp_size + tp_rank][1],
                        plan_hash=self._plan_hashes[engine * self.tp_size + tp_rank],
                        batched=True,
                        cache_plan=True,
                        audit_native=audit_native,
                        consistency_check=bool(Envs.RELAX_P2P_WEIGHT_CONSISTENCY_CHECK),
                    )
                    for tp_rank in range(self.tp_size)
                ]
                for engine in range(self.engine_count)
            ]

        requests = self._phase(commands)
        endpoints = self._prepared_endpoints
        self._negotiate_delivery(requests, endpoints, delivery_builders)
        self._fanout(requests)
        self.needs_seed = False

    def _actor_endpoint(self, weights: Mapping[str, torch.Tensor]) -> dict:
        device = torch.device("cuda", torch.cuda.current_device())
        if any(value.is_cuda and value.device != device for value in weights.values()):
            raise ValueError("Actor weights differ from their current CUDA device")
        return worker_identity(self.session, device)

    def _negotiate_delivery(self, commands, endpoints, delivery_builders) -> None:
        """Negotiate actual worker locality through the loader RPC response."""
        from .remote.plan import PACKED_TARGET_ENCODING

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
                        **(
                            {"describe_tp_replicas": True, "target_encoding": PACKED_TARGET_ENCODING}
                            if Envs.RELAX_COLOCATE_PINNED_NIXL_TARGET
                            else {}
                        ),
                    )
                    for command in pair
                ]
                for pair in commands
            ]

        probes = self._phase(prepare)
        responses = self._fanout(probes)

        def read():
            from .remote.plan import ReceiptValidator, packed_target_regions

            if self.rank != 0:
                return None
            bindings, capabilities, needed = [], [], [[] for _ in range(self.geometry["source_world_size"])]
            remote_by_owner = [{} for _ in needed]
            remote_plans = {}
            remote_layouts = {}
            self._remote_validators = {}
            self._remote_encodings = {}
            self._remote_manifest_layouts = {}
            for pair, response in zip(commands, responses, strict=True):
                if not isinstance(response, Mapping) or response.get("success") is not True:
                    raise ValueError("Missing successful expert prepare response")
                replies = json.loads(response["message"])["shard_p2p_receivers"]
                for command, reply in zip(pair, replies, strict=True):
                    tp_rank = command["tp_rank"]
                    slot = command["engine"] * self.tp_size + tp_rank
                    described = (
                        self._described_receivers[slot] if getattr(self.source, "has_ordinary", False) else None
                    )
                    if described is not None and any(
                        reply["receiver"][key] != value for key, value in described.items() if key != "generation"
                    ):
                        raise ValueError("Ordinary TP receiver changed after layout discovery")
                    result = dict(
                        logical_plan_hash=self._plan_hashes[slot],
                        delivery=delivery_builders[slot](reply["receiver"]),
                    )
                    binding = dict(result, plan_hash=fingerprint(result))
                    identity = {key: command[key] for key in ("session", "engine", "tp_rank", "version", "nonce")}
                    encoding = reply.get("identity", {}).get("target_encoding")
                    if encoding not in (None, PACKED_TARGET_ENCODING):
                        raise ValueError("Receiver selected an unsupported remote target encoding")
                    if encoding is not None:
                        identity["target_encoding"] = encoding
                        command["target_encoding"] = encoding
                    self._remote_encodings[f"{command['engine']}-{tp_rank}"] = encoding
                    if reply != dict(
                        identity=identity,
                        receiver=binding["delivery"]["receiver"],
                        logical_plan_hash=binding["logical_plan_hash"],
                        plan_hash=binding["plan_hash"],
                        **(
                            {"replica_parameters": reply.get("replica_parameters")}
                            if "replica_parameters" in reply
                            else {}
                        ),
                    ):
                        raise ValueError("Expert receiver feedback differs from the actual source delivery")
                    command["delivery"] = binding
                    bindings.append(binding)
                    capabilities.append(reply.get("replica_parameters", []))
                    local = set(binding["delivery"]["routes"]["cuda_ipc"])
                    remote = set(binding["delivery"]["routes"]["nixl"])
                    remote_writes = [write for write in command["writes"] if write["id"] in remote]
                    target = f"{command['engine']}-{tp_rank}"
                    remote_plans[target] = dict(writes=remote_writes)
                    shapes = self.source.target_layouts(command["geometry"])
                    remote_layouts[target] = {
                        name: shapes[name] for name in {write["param"] for write in remote_writes}
                    }
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
            from .replicas import partition_writes, tp_replica_configurations

            tp_replicas = (
                tp_replica_configurations(commands, capabilities) if Envs.RELAX_COLOCATE_PINNED_NIXL_TARGET else {}
            )
            for pair in commands:
                for command in pair:
                    config = tp_replicas.get(f"{command['engine']}-{command['tp_rank']}")
                    if config is not None:
                        command["tp_replicas"] = config

            for target, plan in remote_plans.items():
                if target in tp_replicas:
                    config = tp_replicas[target]
                    plan["writes"] = partition_writes(
                        plan["writes"],
                        config["layouts"],
                        config["parameters"],
                        int(target.split("-")[1]),
                        self.tp_size,
                    )
                validator = self._remote_validators[target] = ReceiptValidator(plan)
                expected = frozenset(write["param"] for write in plan["writes"])
                self._remote_manifest_layouts[target, None] = expected, None, dict.fromkeys(expected)
                encoding = self._remote_encodings[target]
                packed = None
                if encoding is not None and expected:
                    packed = packed_target_regions(
                        plan["writes"], {name: remote_layouts[target][name]["nbytes"] for name in expected}
                    )
                    wire_sizes = {
                        name: sum(region["rows"] * region["row_bytes"] for region in regions.values())
                        for name, regions in packed.items()
                    }
                    self._remote_manifest_layouts[target, encoding] = expected, fingerprint(packed), wire_sizes
                for write in plan["writes"]:
                    owner = remote_by_owner[write["source_rank"]].setdefault(
                        target, dict(writes=[], plan_hash=validator.plan_hash)
                    )
                    owner["writes"].append(write)
                    if packed is not None:
                        name, identity = write["param"], write["id"]
                        owner.setdefault("packed_regions", {}).setdefault(name, {})[identity] = packed[name][identity]
            self._receiver_plan_hashes = tuple(binding["plan_hash"] for binding in bindings)
            self._remote_expected_by_owner = tuple(frozenset(owner) for owner in remote_by_owner)
            return needed, remote_by_owner, remote_plans, remote_layouts

        details = self._phase(read)
        needed = [details[0] if self.rank == 0 else None]
        dist.broadcast_object_list(needed, src=0, group=self.group)
        # Cleanup enters a collective, so share the same decision on every rank.
        self._has_ipc = any(keys for owner in needed[0] for keys in owner)
        self._needed_keys_by_slot = tuple(needed[0][self.rank])
        slots = [None]
        dist.scatter_object_list(
            slots,
            details[1] if self.rank == 0 else None,
            src=0,
            group=self.group,
        )
        self._remote_local_plans = slots[0]
        self._remote_plans = details[2] if self.rank == 0 else None
        self._remote_layouts = details[3] if self.rank == 0 else None
        remote_active = [any(plan["writes"] for plan in details[2].values()) if self.rank == 0 else None]
        dist.broadcast_object_list(remote_active, src=0, group=self.group)
        self._has_remote = remote_active[0]

    def _send_remote(self, sources: Mapping[str, torch.Tensor], version: int) -> tuple[dict, dict | None]:
        """Bind fresh receiver addresses, then collect actual owner DONE
        receipts."""
        from .remote.rpc_receipts import index_receipts

        def commands():
            if self.rank != 0:
                return None
            return [
                [
                    dict(
                        op="bind_epoch",
                        session=self.session,
                        engine=engine,
                        tp_rank=tp,
                        version=version,
                        nonce=uuid4().hex,
                        delivery_hash=self._receiver_plan_hashes[engine * self.tp_size + tp],
                        target_encoding=self._remote_encodings[f"{engine}-{tp}"],
                    )
                    for tp in range(self.tp_size)
                ]
                for engine in range(self.engine_count)
            ]

        responses = self._fanout(self._phase(commands))

        def read_manifests():
            if self.rank != 0:
                return None
            manifests = {}
            for engine, response in enumerate(responses):
                if not isinstance(response, Mapping) or response.get("success") is not True:
                    raise RuntimeError("Missing colocate NIXL epoch response")
                replies = json.loads(response["message"])["shard_p2p_epochs"]
                if len(replies) != self.tp_size:
                    raise RuntimeError("Incomplete colocate NIXL TP epoch response")
                for tp, manifest in enumerate(replies):
                    key = f"{engine}-{tp}"
                    memory_type, device_id = manifest["target_memory_type"], manifest["target_device_id"]
                    encoding = self._remote_encodings[key] if memory_type == "DRAM" else None
                    expected, packed_hash, wire_sizes = self._remote_manifest_layouts[key, encoding]
                    if (
                        manifest["tp_rank"] != tp
                        or manifest["version"] != version
                        or not isinstance(manifest.get("generation"), str)
                        or not manifest["generation"]
                        or type(manifest.get("gpu_id")) is not int
                        or manifest["gpu_id"] < 0
                        or memory_type not in ("VRAM", "DRAM")
                        or (
                            memory_type == "DRAM"
                            and (
                                type(manifest.get("persistent_registration_key")) is not str
                                or len(manifest["persistent_registration_key"]) != 32
                            )
                        )
                        or (memory_type == "VRAM" and manifest.get("persistent_registration_key") is not None)
                        or type(device_id) is not int
                        or (device_id != (0 if memory_type == "DRAM" else manifest["gpu_id"]))
                        or set(manifest["params"]) != expected
                        or manifest.get("target_encoding") != encoding
                        or manifest.get("packed_regions_hash") != packed_hash
                        or "packed_regions" in manifest
                        or bool(manifest["agent_metadata"]) != bool(expected)
                        or any(
                            manifest["params"][name].get("nbytes") != self._remote_layouts[key][name]["nbytes"]
                            or manifest["params"][name].get("dtype") != str(torch.bfloat16)
                            or type(manifest["params"][name].get("address")) is not int
                            or manifest["params"][name]["address"] <= 0
                            or manifest["params"][name].get("wire_nbytes") != wire_sizes[name]
                            for name in expected
                        )
                    ):
                        raise RuntimeError("Colocate NIXL epoch differs from the negotiated plan")
                    manifests[key] = manifest
            return manifests

        manifests = [self._phase(read_manifests)]
        dist.broadcast_object_list(manifests, src=0, group=self.group)

        def send():
            from .colocate_remote import send_remote_shards

            local_manifests = {}
            for target, plan in self._remote_local_plans.items():
                manifest = manifests[0][target]
                if manifest.get("target_encoding") is not None:
                    manifest = dict(manifest, packed_regions=plan["packed_regions"])
                local_manifests[target] = manifest
            receipts, self._writer = send_remote_shards(
                sources,
                self._remote_local_plans,
                local_manifests,
                rank=self.rank,
                version=version,
                writer=getattr(self, "_writer", None),
                agent_future=self._agent_future,
            )
            return receipts

        receipts = self._gather(self._phase(send))

        def route():
            if self.rank != 0:
                return None
            routed = {key: [] for key in self._remote_plans}
            for rank, owner in enumerate(receipts):
                expected = self._remote_expected_by_owner[rank]
                if set(owner) != expected:
                    raise RuntimeError("Colocate NIXL owner omitted or misrouted a receipt")
                for key, receipt in owner.items():
                    routed[key].append(receipt)
            for key in self._remote_plans:
                self._remote_validators[key].validate(
                    index_receipts(routed[key]), manifests[0][key]["generation"], version
                )
            return routed

        receipts = self._phase(route)
        return manifests[0], receipts

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
            if (
                self._has_remote
                and {
                    name: (tuple(value.shape), tuple(value.stride()), str(value.dtype))
                    for name, value in sources.items()
                }
                != self._source_schema
            ):
                raise ValueError("Remote shard source schema changed; native reseeding is required")
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
        remote_manifests, remote_receipts = (
            self._send_remote(self._pending.sources, version) if self._has_remote else (None, None)
        )

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
                        plan_hash=self._plan_hashes[engine * self.tp_size + tp_rank],
                        delivery_hash=self._receiver_plan_hashes[engine * self.tp_size + tp_rank],
                        bindings=[
                            dict(source_rank=rank, source=key, payload=payload)
                            for rank, owner in enumerate(gathered)
                            for key, payload in owner[engine * self.tp_size + tp_rank].items()
                        ],
                        **(
                            dict(
                                epoch_generation=remote_manifests[f"{engine}-{tp_rank}"]["generation"],
                                source_receipts=remote_receipts[f"{engine}-{tp_rank}"],
                            )
                            if self._has_remote
                            else {}
                        ),
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
        pending = self._pending
        if pending is not None:
            self._drain_receivers(pending.refs)
            pending.refs.clear()

    def abort(self) -> None:
        """Drain local readers and fail closed; retain source storage for
        teardown."""
        self.failed = True
        self._drain_pending()

    def wait_for_apply(self, *, version: int) -> None:
        """Await this apply without releasing sources or completing the
        version."""
        pending = self._pending
        error = None
        if self.failed:
            error = "Expert IPC session failed; source storage remains retained"
        elif pending is None or pending.version != version:
            error = "Expert IPC finish has no matching pending version"
        else:
            error = pending.error
        try:
            self._wait_for_receivers(pending.refs if pending is not None else [], error=error)
        except Exception:
            self.abort()
            raise

    def finish(self, *, version: int) -> None:
        try:
            # Validate receiver ACKs before retiring any sender storage.
            self.wait_for_apply(version=version)
            writer = getattr(self, "_writer", None)
            if getattr(self, "_has_remote", False):
                self._phase(lambda: writer.release_buffers() if writer is not None else None)
        except Exception:
            self.abort()
            raise
        pending = self._pending
        pending.sources = None
        if getattr(self, "_has_ipc", True):
            self._phase(torch.cuda.ipc_collect)
        self.completed_version = pending.version
        self._pending = None
