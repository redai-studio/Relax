# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Online canonical HF conversion and lossless TCP memory updates."""

from __future__ import annotations

from time import monotonic, sleep

from relax.distributed.weight_sync.config import MemorySyncConfig
from relax.distributed.weight_sync.memory_delta import MemoryDeltaEncoder
from relax.distributed.weight_sync.memory_transport import MemorySource
from relax.distributed.weight_sync.plan import SyncPlan
from relax.distributed.weight_sync.streaming_delta import (
    SnapshotBudget,
    StreamingDeltaEncoder,
    steady_state_snapshot_bytes,
)
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class UpdateWeightFromMemory:
    def __new__(cls, args, model, weights_getter, *, model_name, quantization_config):
        if MemorySyncConfig.load(args.memory_weight_sync_config).layout == "native_gated_moe":
            from .update_weight_from_sharded_memory import UpdateWeightFromShardedMemory

            return UpdateWeightFromShardedMemory(
                args, model, weights_getter, model_name=model_name, quantization_config=quantization_config
            )
        return super().__new__(cls)

    def __init__(self, args, model, weights_getter, *, model_name, quantization_config) -> None:
        if quantization_config:
            raise ValueError("Memory delta requires unquantized BF16/FP32 export weights")
        from .hf_weight_iterator_base import HfWeightIteratorBase

        self.config = MemorySyncConfig.load(args.memory_weight_sync_config)
        self.weights_getter = weights_getter
        self.iterator = HfWeightIteratorBase.create(
            args, model, model_name=model_name, quantization_config=quantization_config
        )
        self.canonical_hf = self.config.layout == "canonical_hf"
        self.capture_dense_bridge = (
            self.canonical_hf and args.megatron_to_hf_mode == "bridge" and not getattr(args, "num_experts", None)
        )
        self.encoder = (
            StreamingDeltaEncoder(SnapshotBudget(self.config.gpu_budget_bytes), bucket_bytes=self.config.bucket_bytes)
            if self.canonical_hf
            else MemoryDeltaEncoder(
                bucket_bytes=self.config.bucket_bytes, snapshot_budget_bytes=self.config.gpu_budget_bytes
            )
        )
        self.receiver_plan: SyncPlan | None = None
        self.source: MemorySource | None = None
        self.members: set[str] | None = None
        self.engines: tuple = ()
        self.failed = False

    def update_weights(self, rollout_manager) -> None:
        import torch
        import torch.distributed as dist

        from relax.utils.distributed_utils import get_gloo_group

        if self.failed:
            raise RuntimeError("Failed memory stream must be discarded; restart with a new stream and FULL")
        started = monotonic()
        group = get_gloo_group()
        rank = dist.get_rank(group)
        self.source_ranks = tuple(dist.get_process_group_ranks(group))
        error, buckets = None, None
        local_weights = None
        try:
            local_weights = self.weights_getter()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size(group)
        # Do not let peers enter converter collectives if a rank cannot even
        # access its stopped optimizer snapshot.
        dist.all_gather_object(errors, error, group=group)

        def capture_gate(local_error: str | None) -> None:
            capture_errors = [None] * dist.get_world_size(group)
            dist.all_gather_object(capture_errors, local_error, group=group)
            if any(capture_errors):
                raise RuntimeError(f"Dense Bridge memory capture failed across source ranks: {capture_errors}")

        # Every rank runs the existing converter/collectives while the optimizer
        # is stopped. Encoder freezes each yielded value before buffer reuse.
        named = None
        try:
            if any(errors):
                raise RuntimeError(f"Cannot access a training snapshot: {errors}")
            with torch.no_grad():
                if getattr(self, "capture_dense_bridge", False):
                    preparation_error = None
                    if rank == 0:
                        try:
                            version = self.encoder.version + 1
                            full = self.config.full_only or version % self.config.anchor_interval == 0
                            self.encoder.begin(full=full, compare_changes=not self.config.full_only)
                        except Exception as exc:
                            preparation_error = f"{type(exc).__name__}: {exc}"
                    capture_gate(preparation_error)
                    self.iterator.capture_gated_dense_tensors(
                        local_weights,
                        emit=self.encoder.add if rank == 0 else lambda _name, _tensor: None,
                        error_gate=capture_gate,
                        metadata_group=group,
                    )
                    if rank == 0:
                        buckets = self.encoder.finish()
                        if steady_state_snapshot_bytes(buckets) > self.config.gpu_budget_bytes:
                            raise ValueError("Canonical HF snapshots cannot fit the next delta within the GPU budget")
                elif rank == 0:
                    named = (pair for chunk in self.iterator.get_hf_weight_chunks(local_weights) for pair in chunk)
                    version = self.encoder.version + 1
                    full = self.config.full_only or version % self.config.anchor_interval == 0
                    if self.canonical_hf:
                        self.encoder.begin(full=full, compare_changes=not self.config.full_only)
                        for name, tensor in named:
                            self.encoder.add(name, tensor)
                        buckets = self.encoder.finish()
                        if steady_state_snapshot_bytes(buckets) > self.config.gpu_budget_bytes:
                            raise ValueError("Canonical HF snapshots cannot fit the next delta within the GPU budget")
                    else:
                        buckets = self.encoder.prepare(named, full=full)
                else:
                    named = (pair for chunk in self.iterator.get_hf_weight_chunks(local_weights) for pair in chunk)
                    for _ in named:
                        pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            # A local capture/packing failure must not strand peers inside the
            # remaining conversion collectives. Producer failures still obey
            # the distributed backend's collective timeout.
            try:
                if named is not None:
                    for _ in named:
                        pass
            except Exception:
                pass
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
        result = [None]
        if rank == 0:
            try:
                if any(errors):
                    raise RuntimeError(f"Memory HF capture failed: {errors}")
                capture_s = monotonic() - started
                metrics = self._install(rollout_manager, buckets, started)
                metrics.update(
                    version=version,
                    capture_encode_s=capture_s,
                    sync_with_probe_s=monotonic() - started,
                    elements=sum(b.elements for b in buckets),
                    changed=None if self.canonical_hf and self.config.full_only else sum(b.changed for b in buckets),
                    payload_bytes=sum(len(b.payload) for b in buckets),
                    protocol_control_bytes=self.source.last_control_bytes,
                    logical_delivery_bytes=sum(len(b.payload) for b in buckets) * len(self.members),
                    payload_scope="Logical member deliveries; not physical NIC traffic",
                    members=len(self.members),
                    full=all(b.encoding == "full" for b in buckets),
                )
                result[0] = metrics
                logger.info("Memory weight sync: %s", metrics)
            except Exception as exc:
                self.failed = True
                self.encoder.abort()
                closure_errors = self._close_generation(rollout_manager)
                if self.source is not None:
                    try:
                        self.source.close()
                    except Exception as close_error:
                        logger.warning("Memory source cleanup failed: %s", close_error)
                    self.source = None
                result[0] = {"error": f"{type(exc).__name__}: {exc}", "closure_errors": closure_errors}
        dist.broadcast_object_list(result, src=0, group=group)
        if "error" in result[0]:
            self.failed = True
            raise RuntimeError(f"Memory weight sync failed: {result[0]}")

    def _close_generation(self, rollout_manager) -> list[str]:
        import ray

        errors = []
        try:
            engines, *_ = ray.get(rollout_manager.get_rollout_engines_and_lock.remote(), timeout=10)
            engines = tuple(engines)
        except Exception as exc:
            errors.append(f"Cannot rediscover inference group: {exc}")
            engines = ()
        # Include the original group when membership changed or discovery died.
        known = {e._actor_id.hex(): e for e in (*self.engines, *engines)}
        for member, engine in known.items():
            try:
                ray.get(engine.pause_generation.remote(timeout=5), timeout=10)
            except Exception as exc:
                try:
                    ray.kill(engine, no_restart=True)
                except Exception as kill_error:
                    errors.append(f"Cannot close generation for {member}: {exc}; kill: {kill_error}")
        for error in errors:
            logger.error("Memory failure quarantine unconfirmed: %s", error)
        return errors

    def _install(self, rollout_manager, buckets, update_started: float) -> dict:
        import ray

        deadline = monotonic() + self.config.sync_timeout_s

        def remaining():
            value = deadline - monotonic()
            if value <= 0:
                raise TimeoutError("Memory group installation deadline exceeded")
            return value

        resolve_started = monotonic()
        engines, lock, *metadata = ray.get(rollout_manager.get_rollout_engines_and_lock.remote(), timeout=remaining())
        engines = tuple(engines)
        previous_engines = self.engines
        self.engines = tuple({e._actor_id.hex(): e for e in (*previous_engines, *engines)}.values())
        identities = tuple(engine._actor_id.hex() for engine in engines)
        if not identities or len(set(identities)) != len(engines):
            raise RuntimeError("Memory sync requires distinct, nonempty inference members")
        if self.canonical_hf:
            counts = metadata[1] if len(metadata) >= 2 else None
            if not isinstance(counts, (list, tuple)) or len(counts) != len(engines):
                raise ValueError("Canonical HF synchronization requires actual engine worker counts")
            if any(type(count) is not int or count <= 0 for count in counts):
                raise ValueError("Canonical HF worker counts must be positive integers")
            capabilities = ray.get(
                [engine.get_hf_memory_capability.remote(remaining()) for engine in engines], timeout=remaining()
            )
            runtime_counts = [capability["workers"] for capability in capabilities]
            if runtime_counts != list(counts):
                raise ValueError("Canonical HF runtime worker topology changed or disagrees with manager allocation")
            capability_fingerprints = {
                identity: capability["fingerprint"]
                for identity, capability in zip(identities, capabilities, strict=True)
            }
            plan = SyncPlan.build(
                {
                    identity: tuple(f"{identity}:tp:{tp}" for tp in range(count))
                    for identity, count in zip(identities, runtime_counts, strict=True)
                },
                tensor_schema=self.encoder.pending_schema,
                source_ranks=self.source_ranks,
                receiver_capabilities=capability_fingerprints,
            )
        else:
            counts = metadata[1] if len(metadata) >= 2 else None
            if not isinstance(counts, (list, tuple)) or list(counts) != [1] * len(engines):
                raise ValueError("The direct-copy Dense adapter requires one allocated worker per engine")
            capabilities = ray.get(
                [engine.get_dense_memory_capability.remote(remaining()) for engine in engines], timeout=remaining()
            )
            capability_fingerprints = {
                identity: capability["fingerprint"]
                for identity, capability in zip(identities, capabilities, strict=True)
            }
            plan = SyncPlan.build(
                {identity: (identity,) for identity in identities},
                tensor_schema=self.encoder.pending_schema,
                source_ranks=self.source_ranks,
                receiver_capabilities=capability_fingerprints,
            )
        if self.receiver_plan is not None and plan != self.receiver_plan:
            raise RuntimeError("Receiver worker topology changed; start a new memory stream")
        self.receiver_plan = plan
        members = plan.members
        if self.members is not None and members != self.members:
            raise RuntimeError("Inference membership changed; restart with a new memory stream")
        self.members = members
        resolved_at = monotonic()
        if self.source is None:
            self.source = MemorySource(
                ray.util.get_node_ip_address(),
                timeout_s=self.config.sync_timeout_s,
                host_budget_bytes=self.config.host_budget_bytes,
            )
        descriptor = self.source.publish(
            buckets,
            stream=self.config.stream,
            version=self.encoder.version + 1,
            base=self.encoder.version,
            schema=self.encoder.pending_schema,
            members=members,
            reset=self.encoder.version == -1,
            group_update=self.receiver_plan.fingerprint if self.canonical_hf else None,
        )
        published_at = monotonic()
        while not ray.get(lock.acquire.remote(), timeout=remaining()):
            sleep(min(0.05, remaining()))
        paused = monotonic()
        try:
            ray.get([e.pause_generation.remote(timeout=remaining()) for e in engines], timeout=remaining())
            pause_returned = monotonic()
            if self.canonical_hf:
                # Reserve one full HF directory for ordinary loader temporaries.
                # This is an explicit admission estimate, verified by GPU runs;
                # arbitrary model-specific loader workspace is not inferred.
                workspace_bytes = sum(
                    bucket.elements * {"bfloat16": 2, "float32": 4}[bucket.dtype] for bucket in buckets
                )
                installs = [
                    e.install_hf_memory_update.remote(
                        descriptor,
                        self.config.stream,
                        self.receiver_plan.group_members(e._actor_id.hex()),
                        self.receiver_plan.fingerprint,
                        remaining(),
                        self.config.host_budget_bytes,
                        self.config.receiver_gpu_budget_bytes,
                        workspace_bytes,
                        capability_fingerprints[e._actor_id.hex()],
                    )
                    for e in engines
                ]
            else:
                installs = [
                    e.install_memory_update.remote(
                        descriptor,
                        e._actor_id.hex(),
                        remaining(),
                        self.config.host_budget_bytes,
                        capability_fingerprints[e._actor_id.hex()],
                    )
                    for e in engines
                ]
            loaded = ray.get(installs, timeout=remaining())
            receivers_returned = monotonic()
            if any(not isinstance(item, dict) or item.get("success") is not True for item in loaded):
                raise RuntimeError(f"Inference member failed memory update: {loaded}")
            acknowledged = self.source.wait(remaining())
            if self.canonical_hf:
                self.encoder.prepare_commit(acknowledged, members)
                self.encoder.commit()
            else:
                self.encoder.commit(acknowledged, members)
            self.source.finish()
            loaded_at = monotonic()
            ray.get([e.continue_generation.remote(timeout=remaining()) for e in engines], timeout=remaining())
            ready_at = monotonic()
            # Verify the explicit generation-ready
            # probe. This probe is reported separately from loader time.
            ray.get([e.memory_weight_generation_probe.remote(remaining()) for e in engines], timeout=remaining())
            return dict(
                install_s=loaded_at - paused,
                continue_s=ready_at - loaded_at,
                sync_to_ready_s=ready_at - update_started,
                probe_s=monotonic() - ready_at,
                generation_pause_s=ready_at - paused,
                source_host=descriptor["host"],
                receivers=loaded,
                resolve_group_s=resolved_at - resolve_started,
                publish_s=published_at - resolved_at,
                lock_s=paused - published_at,
                pause_calls_s=pause_returned - paused,
                receiver_calls_s=receivers_returned - pause_returned,
                commit_s=loaded_at - receivers_returned,
            )
        finally:
            ray.get(lock.release.remote(), timeout=10)
