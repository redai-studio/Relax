# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Online HF conversion and lossless TCP memory updates for synchronous Dense
RL."""

from __future__ import annotations

from time import monotonic, sleep

from relax.distributed.weight_sync.config import MemorySyncConfig
from relax.distributed.weight_sync.memory_delta import MemoryDeltaEncoder
from relax.distributed.weight_sync.memory_transport import MemorySource
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class UpdateWeightFromMemory:
    def __init__(self, args, model, weights_getter, *, model_name, quantization_config) -> None:
        if quantization_config:
            raise ValueError("Memory delta requires unquantized BF16 weights")
        from .hf_weight_iterator_base import HfWeightIteratorBase

        self.config = MemorySyncConfig.load(args.memory_weight_sync_config)
        self.weights_getter = weights_getter
        self.iterator = HfWeightIteratorBase.create(
            args, model, model_name=model_name, quantization_config=quantization_config
        )
        self.encoder = MemoryDeltaEncoder(
            bucket_bytes=self.config.bucket_bytes, snapshot_budget_bytes=self.config.gpu_budget_bytes
        )
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
        # Every rank runs the existing converter/collectives while the optimizer
        # is stopped. Encoder freezes each yielded value before buffer reuse.
        named = None
        try:
            if any(errors):
                raise RuntimeError(f"Cannot access a training snapshot: {errors}")
            named = (pair for chunk in self.iterator.get_hf_weight_chunks(local_weights) for pair in chunk)
            with torch.no_grad():
                if rank == 0:
                    version = self.encoder.version + 1
                    full = self.config.full_only or version % self.config.anchor_interval == 0
                    buckets = self.encoder.prepare(named, full=full)
                else:
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
                    changed=sum(b.changed for b in buckets),
                    payload_bytes=sum(len(b.payload) for b in buckets),
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
        engines, lock, *_ = ray.get(rollout_manager.get_rollout_engines_and_lock.remote(), timeout=remaining())
        engines = tuple(engines)
        previous_engines = self.engines
        self.engines = tuple({e._actor_id.hex(): e for e in (*previous_engines, *engines)}.values())
        members = {engine._actor_id.hex() for engine in engines}
        if not members or len(members) != len(engines):
            raise RuntimeError("Memory sync requires distinct, nonempty inference members")
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
        )
        published_at = monotonic()
        while not ray.get(lock.acquire.remote(), timeout=remaining()):
            sleep(min(0.05, remaining()))
        paused = monotonic()
        try:
            ray.get([e.pause_generation.remote(timeout=remaining()) for e in engines], timeout=remaining())
            pause_returned = monotonic()
            loaded = ray.get(
                [
                    e.install_memory_update.remote(
                        descriptor,
                        e._actor_id.hex(),
                        remaining(),
                        self.config.host_budget_bytes,
                    )
                    for e in engines
                ],
                timeout=remaining(),
            )
            receivers_returned = monotonic()
            if any(not isinstance(item, dict) or item.get("success") is not True for item in loaded):
                raise RuntimeError(f"Inference member failed memory update: {loaded}")
            self.encoder.commit(self.source.wait(remaining()), members)
            self.source.finish()
            loaded_at = monotonic()
            ray.get([e.continue_generation.remote(timeout=remaining()) for e in engines], timeout=remaining())
            ready_at = monotonic()
            # Match the existing disk prototype's explicit generation-ready
            # probe. This probe is reported separately from loader time.
            ray.get([e.disk_weight_generation_probe.remote(remaining()) for e in engines], timeout=remaining())
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
