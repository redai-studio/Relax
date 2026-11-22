# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Resident Async weights using the shared shard plan and transfer session."""

import json
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import ray
import torch
import torch.distributed as dist

from relax.backends.sglang.weight_loader import COLOCATE_NIXL_LOADER, EXPERT_IPC_LOADER, TP_IPC_LOADER
from relax.distributed.checkpoint_service.backends.shard_p2p.common import validate_worker_identity
from relax.distributed.checkpoint_service.backends.shard_p2p.remote.common import create_agent
from relax.distributed.checkpoint_service.backends.shard_p2p.synchronizer import ShardP2PSynchronizer
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.env import Envs

from ..sglang import MultiprocessingSerializer
from .combined_shard_source import CombinedShardSource
from .expert_layout import DirectExpertLayout
from .hf_weight_iterator_bridge import HfWeightIteratorBridge
from .shard_p2p_source import MegatronShardP2PSource
from .utils import run_with_distributed_error_check


def observe_initial_native(backend, values: list[tuple[str, torch.Tensor]]) -> None:
    """Retain only the actual initial HF schema, never initial weight
    storage."""
    seed = getattr(backend, "_async_shard_initial_seed", None)
    if seed is None or seed["ready"]:
        return
    for name, value in values:
        if ".experts." in name or name.startswith("relax.expert_slice.v1:"):
            continue
        spec = dict(shape=list(value.shape), dtype=str(value.dtype))
        previous = seed["specs"].setdefault(name, spec)
        if previous != spec:
            raise ValueError("Initial native HF schema changed during weight synchronization")


def validate_async_devices(actors: list[dict], engines: list[list[dict]]) -> None:
    """Qualify distinct resident allocations from the actual CUDA workers."""
    receivers = [value for engine in engines for value in engine]
    if not actors or not receivers or any(len(engine) != len(engines[0]) for engine in engines):
        raise ValueError("Async shard receivers have incomplete device coverage")
    for value in [*actors, *receivers]:
        validate_worker_identity(value)
    source_ids = {(value["boot_id"], value["gpu_uuid"]) for value in actors}
    target_ids = {(value["boot_id"], value["gpu_uuid"]) for value in receivers}
    if len(source_ids) != len(actors) or len(target_ids) != len(receivers) or source_ids & target_ids:
        raise ValueError("Async weight sync requires distinct training and inference GPUs")


def supports_async_shards(backend, *, rollout_only: bool, actor_fwd_only: bool) -> bool:
    """Other recipients and unsupported conversion semantics stay native."""
    args = backend.args
    return bool(
        getattr(args, "use_shard_p2p_weight_sync", False)
        and getattr(args, "fully_async", False)
        and not getattr(args, "hybrid", False)
        and not getattr(args, "offload_train", False)
        and not getattr(args, "offload_rollout", False)
        and rollout_only
        and not actor_fwd_only
        and backend._use_bridge
        and backend._weights_getter is None
        and not backend._lora_enabled
    )


def probe_receivers(backend, group, *, persistent_targets: bool = True) -> dict:
    """The actual guarded GPU workers advertise their current session."""
    result = [None]

    def query():
        if dist.get_rank(group) != 0:
            return
        session = uuid4().hex
        refs = []
        for engine, handle in enumerate(backend.rollout_engines.values()):
            commands = [
                MultiprocessingSerializer.serialize(
                    [
                        (
                            json.dumps(
                                dict(
                                    op="probe_nixl",
                                    session=session,
                                    engine=engine,
                                    tp_rank=tp,
                                    version=0,
                                    nonce=uuid4().hex,
                                    persistent_targets=persistent_targets,
                                    geometry=dict(tp_size=backend.args.rollout_num_gpus_per_engine),
                                )
                            ),
                            torch.empty(0),
                        )
                    ],
                    output_str=True,
                )
                for tp in range(backend.args.rollout_num_gpus_per_engine)
            ]
            refs.append(handle.update_weights_from_tensor.remote(commands, load_format=COLOCATE_NIXL_LOADER))
        values = ray.get(refs, timeout=600)
        replies = [json.loads(value["message"])["shard_p2p_nixl_capabilities"] for value in values]
        result[0] = dict(
            signature=backend._rollout_topology_signature,
            native_flat_bucket=bool(replies)
            and all(
                value.get("success") is True
                and len(pair) == backend.args.rollout_num_gpus_per_engine
                and all(
                    reply.get("supported") is True and reply.get("native_flat_bucket_v1") is True for reply in pair
                )
                for value, pair in zip(values, replies, strict=True)
            ),
        )
        if not persistent_targets:
            # Initial loading needs only capability. Do not cache dummy dtype or
            # require resident sessions before the first native weight load.
            return
        if not replies or any(
            value.get("success") is not True
            or len(pair) != backend.args.rollout_num_gpus_per_engine
            or any(reply.get("supported") is not True or not reply.get("layout") for reply in pair)
            for value, pair in zip(values, replies, strict=True)
        ):
            raise RuntimeError("Async shard receivers did not prove native NIXL capability")
        layouts = [reply["layout"] for pair in replies for reply in pair]
        if any(layout != layouts[0] for layout in layouts[1:]):
            raise ValueError("Async receivers disagree on their native weight layout")
        result[0] = dict(
            layout=layouts[0],
            sessions=[reply["session"] for pair in replies for reply in pair],
            signature=backend._rollout_topology_signature,
        )

    run_with_distributed_error_check(query, group)
    dist.broadcast_object_list(result, src=0, group=group)
    return result[0]


def prefetch_sender_agent() -> Future:
    """Initialize transport during native seeding, without exporting
    weights."""
    device = torch.cuda.current_device()

    def initialize():
        torch.cuda.set_device(device)
        return create_agent()

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-sender-init")
    future = pool.submit(initialize)
    pool.shutdown(wait=False)
    return future


def prepare_async_shard_iterator(backend, *, prepare_native: bool = False) -> HfWeightIteratorBridge:
    """Collectively prepare export metadata while Actor initialization can
    overlap rollout loading."""
    group = get_gloo_group()
    iterator = getattr(backend, "_async_shard_iterator", None)
    ready = [None] * dist.get_world_size(group)
    dist.all_gather_object(ready, iterator is not None, group=group)
    # Rebuild on every rank if any cache is missing: construction contains PP/EP collectives.
    if not all(ready):
        iterator = HfWeightIteratorBridge(backend.args, backend.model, backend.model_name, backend.quantization_config)
        backend._async_shard_iterator = iterator
    if prepare_native and not Envs.RELAX_P2P_WEIGHT_CONSISTENCY_CHECK:
        from .tp_shard_source import observe_native_specs

        weights = dict(
            backend._megatron.named_params_and_buffers(
                backend.args, backend.model, convert_to_global_name=False, include_persistent_buffers=True
            )
        )
        specs = {}
        # Complete the real conversion while rollout loading can overlap it.
        # Keep scalar schema only; dummy receiver contents are not a native seed.
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            observe_native_specs(specs, iterator.iter_initial_hf_weights(weights))
        seed = getattr(backend, "_async_shard_initial_seed", None)
        backend._async_shard_initial_seed = dict(
            specs=specs,
            ready=False,
            version=0,
            signature=None,
            iterator=id(iterator),
            agent_future=seed["agent_future"] if seed is not None else prefetch_sender_agent(),
        )
    return iterator


def initial_rollout_only(backend) -> bool:
    """Prove every current PP sender has no ActorFwd or Ref recipients."""
    sender = getattr(backend, "_is_pp_src_rank", None)
    group = backend._model_update_groups_for_actor_fwd_ref
    supported = getattr(backend, "_actor_fwd_ref_world_size", None) == 1 and (
        sender is False or sender is True and group is not None and dist.get_world_size(group) == 1
    )
    ready = torch.tensor([int(supported)], dtype=torch.int32, device="cpu")
    dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=get_gloo_group())
    return bool(ready.item())


class AsyncShardSync:
    """Keep Async scheduling separate from the common data/transport core.

    Generation remains paused through native fallback, direct writes, GPU
    completion, ACK and version publication. The Actor cannot train again
    before this synchronous API has released every source reader.
    """

    def __init__(self, backend) -> None:
        self.backend = backend
        self.group = get_gloo_group()
        self.rank = dist.get_rank(self.group)
        self.source = self.core = None
        self.native_phase = False
        self.signature = None
        self.iterator = self._checked(lambda: prepare_async_shard_iterator(backend))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-sender")
        seed = getattr(backend, "_async_shard_initial_seed", None)
        future = seed.get("agent_future") if seed is not None else None
        self.agent_future = future if future is not None else prefetch_sender_agent()
        self.cache_future: Future | None = None

    def release_native_cache(self) -> None:
        """Release only unused allocator blocks; live model storage stays
        held."""
        if self.cache_future is not None:
            raise RuntimeError("A previous native cache release is still pending")
        device = torch.cuda.current_device()

        def release() -> None:
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()

        self.cache_future = self.pool.submit(release)

    def _finish_native_cache(self) -> None:
        future = self.cache_future
        self._checked(lambda: future.result() if future is not None else None)
        self.cache_future = None

    def _checked(self, action):
        return run_with_distributed_error_check(action, self.group)

    def _probe(self) -> dict:
        return probe_receivers(self.backend, self.group)

    def _initialize(self, layout: dict) -> None:
        self.native_flat_bucket = layout.get("native_flat_bucket_v1") is True
        expert = None
        args = self.backend.args
        if (
            layout.get("num_experts")
            and not self.backend.quantization_config
            and args.pipeline_model_parallel_size == 1
            and args.context_parallel_size == 1
            and args.expert_tensor_parallel_size == 1
        ):
            geometry = DirectExpertLayout.from_receiver(layout)

            qualified = self._checked(lambda: self.iterator.configure_direct_experts(geometry))
            enabled = torch.tensor([int(qualified)], dtype=torch.int32, device="cpu")
            dist.all_reduce(enabled, op=dist.ReduceOp.MIN, group=self.group)
            if enabled.item():
                expert = MegatronShardP2PSource(
                    self.iterator,
                    geometry,
                    process_group=self.group,
                    num_layers=self.backend.args.num_layers,
                    qualified=True,
                )
            else:
                self._checked(lambda: self.iterator.configure_direct_experts(None))
        owner = SimpleNamespace(args=self.backend.args, _hf_weight_iterator=self.iterator)
        self.source = self._checked(lambda: CombinedShardSource(owner, process_group=self.group, expert=expert))
        count = [len(self.backend.rollout_engines) if self.rank == 0 else None]
        dist.broadcast_object_list(count, src=0, group=self.group)
        handles = list(self.backend.rollout_engines.values()) if self.rank == 0 else [None] * count[0]
        self.engine_indices = {rank: engine for engine, rank in enumerate(self.backend.rollout_engines)}
        self.core = ShardP2PSynchronizer(
            self.source,
            process_group=self.group,
            engines=handles,
            serialize=MultiprocessingSerializer.serialize,
            control_loader=TP_IPC_LOADER,
            update_loader=EXPERT_IPC_LOADER,
            validate_devices=validate_async_devices,
            agent_future=self.agent_future,
            persistent_targets=True,
            staging_bytes=min(self.backend.args.update_weight_buffer_size, 128 << 20),
        )

    def _retire(self, sessions: list[str | None]) -> None:
        core = self.core
        if core.failed or core.retained_sources is not None or core.completed_version is None:
            raise RuntimeError("Cannot replace an incomplete Async transfer session")
        if any(session not in (None, core.session) for session in sessions):
            raise RuntimeError("Receiver belongs to an unrelated Async transfer session")
        count = len(sessions) // core.tp_size
        # Proxies may have been recreated, but surviving receivers still hold
        # their old session's engine number, even after removal or reordering.
        indices = [None]

        def identify():
            if self.rank == 0:
                indices[0] = [
                    self.engine_indices[rank]
                    if core.session in sessions[engine * core.tp_size : (engine + 1) * core.tp_size]
                    else engine
                    for engine, rank in enumerate(self.backend.rollout_engines)
                ]

        self._checked(identify)
        dist.broadcast_object_list(indices, src=0, group=self.group)
        commands = [
            [
                dict(
                    op="retire",
                    session=core.session,
                    engine=indices[0][engine],
                    tp_rank=tp,
                    version=core.completed_version,
                    nonce=uuid4().hex,
                    geometry=core.geometry,
                )
                for tp in range(core.tp_size)
            ]
            for engine in range(count)
        ]
        handles = list(self.backend.rollout_engines.values()) if self.rank == 0 else [None] * count
        core._fanout(commands, engines=handles)
        writer = getattr(core, "_writer", None)
        self._checked(lambda: writer.disconnect_targets() if writer is not None else None)

    def observe_native(self, values) -> None:
        if self.source is not None and self.core.needs_seed:
            self.source.observe_native(values)

    def skips_native(self, name: str) -> bool:
        if not self.native_phase or self.source is None or self.core.needs_seed:
            return False
        return name in self.source.native_skip_names or self.source.has_experts and ".experts." in name

    def _gather_native_specs(self, local_specs: dict | None) -> dict | None:
        records = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(records, (self.backend.weight_version, self.signature, local_specs), group=self.group)

        def merge():
            if any(record[:2] != records[0][:2] for record in records):
                raise ValueError("Async native schema version or topology differs across Actor ranks")
            if any(record[2] is None for record in records):
                return None
            specs = {}
            for _, _, values in records:
                for name, spec in values.items():
                    if specs.setdefault(name, spec) != spec:
                        raise ValueError("Async native HF schema differs across Actor stages")
            return specs or None

        return self._checked(merge)

    def update(self, *, rollout_only: bool, actor_fwd_only: bool) -> None:
        backend = self.backend
        if self.core is None and not Envs.RELAX_P2P_WEIGHT_CONSISTENCY_CHECK:
            # The first optimizer step leaves unused training allocations.
            # Reclaim them during cold metadata negotiation, before native
            # fallback needs fresh gather buffers.
            self.release_native_cache()
        backend._rollout_control_request("/pause_generation")
        self._checked(
            lambda: (
                ray.get([value.flush_cache.remote() for value in backend.rollout_engines.values()], timeout=600)
                if self.rank == 0
                else None
            )
        )
        probe = self._probe()
        fresh = (
            self.core is None
            or self.signature != probe["signature"]
            or self.core.completed_version != backend.weight_version
        )
        if not fresh and any(session != self.core.session for session in probe["sessions"]):
            fresh = True
        if fresh:
            writer = getattr(self.core, "_writer", None)
            if self.core is not None:
                self._retire(probe["sessions"])
            self._initialize(probe["layout"])
            if writer is not None:
                self.core._writer = writer
            self.signature = probe["signature"]
            self._checked(self.source.begin_native_seed)
        weights = self._checked(
            lambda: dict(
                backend._megatron.named_params_and_buffers(
                    backend.args,
                    backend.model,
                    convert_to_global_name=False,
                    include_persistent_buffers=True,
                )
            )
        )
        # Real init-time export or native initialization established the HF
        # schema. With content auditing disabled, qualify current allocations
        # before the native fallback, avoiding a full-model native broadcast.
        seed = getattr(backend, "_async_shard_initial_seed", None)
        backend._async_shard_initial_seed = None
        if self.core.needs_seed:
            valid_seed = (
                seed is not None
                and seed["version"] == backend.weight_version
                and (
                    seed["ready"]
                    and seed["signature"] == probe["signature"]
                    or backend.weight_version == 0
                    and not seed["ready"]
                    and seed.get("iterator") == id(self.iterator)
                )
                and not Envs.RELAX_P2P_WEIGHT_CONSISTENCY_CHECK
            )
            specs = self._gather_native_specs(seed["specs"] if valid_seed else None)
            if specs is not None:
                self.source.ordinary.native_specs = specs
                self.core.prepare(weights, version=backend.weight_version + 1, audit_native=False)
        self._finish_native_cache()
        self.native_phase = True
        try:
            backend.update_weights_for_rollout(rollout_only=rollout_only, actor_fwd_only=actor_fwd_only)
        finally:
            self.native_phase = False
        version = backend.weight_version
        if self.core.needs_seed:
            self.source.ordinary.native_specs = self._gather_native_specs(self.source.ordinary.native_specs) or {}
            self.core.prepare(weights, version=version, audit_native=bool(Envs.RELAX_P2P_WEIGHT_CONSISTENCY_CHECK))
        self.core.update(weights, version=version)
        # The native gather/conversion temporaries must be reclaimed before
        # the Actor can train again, even when this overlapped direct copies.
        self._finish_native_cache()
        backend._rollout_control_request(
            "/update_weight_version", {"new_version": str(version), "abort_all_requests": False}
        )
        backend._rollout_control_request("/continue_generation")
