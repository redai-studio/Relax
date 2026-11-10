# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Owner-local Qwen3.5 experts and bounded nonexpert HF streams over TCP."""

from __future__ import annotations

import secrets
from contextlib import closing
from dataclasses import replace
from time import monotonic, sleep
from typing import Any

from relax.distributed.weight_sync.config import MemorySyncConfig
from relax.distributed.weight_sync.memory_transport import MemorySource
from relax.distributed.weight_sync.streaming_delta import SnapshotBudget, StreamingDeltaEncoder
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class UpdateWeightFromShardedMemory:
    """All source ranks commit only after every fixed engine applied all
    owners.

    Source-local capture errors are gated before entering nonexpert converter
    collectives. Any failure invalidates the entire updater; recovery uses a
    fresh stream and complete FULL, never a partially committed retry.
    """

    def __init__(
        self, args: Any, model: Any, weights_getter: Any, *, model_name: str, quantization_config: Any
    ) -> None:
        from .hf_weight_iterator_base import HfWeightIteratorBase

        if quantization_config:
            raise ValueError("Owner-local memory sync requires unquantized weights")
        self.config = MemorySyncConfig.load(args.memory_weight_sync_config)
        self.weights_getter = weights_getter
        self.iterator = HfWeightIteratorBase.create(args, model, model_name=model_name, quantization_config=None)
        self.budget = SnapshotBudget(self.config.gpu_budget_bytes)
        self.encoders = {
            f"expert:{tp}": StreamingDeltaEncoder(self.budget, bucket_bytes=self.config.bucket_bytes)
            for tp in range(2)
        }
        self.sources: dict[str, MemorySource] = {}
        self.engines: tuple = ()
        self.members: tuple[str, ...] | None = None
        self.version = -1
        self.delta_updates = 0
        self.failed = False

    def _gate(self, error: str | None) -> None:
        import torch.distributed as dist

        from relax.utils.distributed_utils import get_gloo_group

        group = get_gloo_group()
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            raise RuntimeError(f"Sharded memory update failed across source ranks: {errors}")

    def update_weights(self, rollout_manager: Any, *, full_only: bool | None = None) -> dict:
        import torch.distributed as dist

        from relax.utils.distributed_utils import get_gloo_group

        if self.failed:
            raise RuntimeError("Discard the failed sharded updater and start a new FULL stream")
        rank = dist.get_rank(get_gloo_group())
        original_config = self.config
        if full_only is not None:
            self.config = replace(self.config, full_only=full_only)
        try:
            return self._update(rollout_manager)
        except Exception as original_error:
            self.failed = True
            cleanup_errors = []
            for encoder in self.encoders.values():
                try:
                    encoder.abort()
                except Exception as exc:
                    cleanup_errors.append(f"Encoder abort: {type(exc).__name__}: {exc}")
            if rank == 0:
                from .update_weight_from_memory import UpdateWeightFromMemory

                try:
                    cleanup_errors.extend(UpdateWeightFromMemory._close_generation(self, rollout_manager))
                except Exception as exc:
                    cleanup_errors.append(f"Group quarantine: {type(exc).__name__}: {exc}")
            for name, source in self.sources.items():
                try:
                    source.close()
                except Exception as exc:
                    cleanup_errors.append(f"Source {name} close: {type(exc).__name__}: {exc}")
            if cleanup_errors:
                logger.error("Sharded update cleanup failures: %s", cleanup_errors)
                add_note = getattr(original_error, "add_note", None)
                if callable(add_note):
                    add_note(f"Sharded update cleanup failures: {cleanup_errors}")
            raise
        finally:
            self.config = original_config

    def _update(self, manager: Any) -> dict:
        import ray
        import torch
        import torch.distributed as dist
        from megatron.core import mpu

        from relax.utils.distributed_utils import get_gloo_group

        from .expert_layout import DirectExpertLayout, expert_slice_chunk
        from .owner_expert_blocks import iter_owner_expert_blocks

        group = get_gloo_group()
        rank, world = dist.get_rank(group), dist.get_world_size(group)
        started = monotonic()
        deadline = started + self.config.sync_timeout_s

        def remaining() -> float:
            value = deadline - monotonic()
            if value <= 0:
                raise TimeoutError("Sharded group synchronization deadline exceeded")
            return value

        plan = [None]
        lock = None
        if rank == 0:
            try:
                engines, lock, *_ = ray.get(manager.get_rollout_engines_and_lock.remote(), timeout=remaining())
                engines = tuple(engines)
                identities = tuple(engine._actor_id.hex() for engine in engines)
                self.engines = tuple({e._actor_id.hex(): e for e in (*self.engines, *engines)}.values())
                if len(identities) != 4 or len(set(identities)) != 4 or self.members not in (None, identities):
                    raise ValueError("Sharded synchronization requires four unchanged TP2 engines")
                layouts = ray.get([e.get_expert_weight_sync_layout.remote() for e in engines], timeout=remaining())
                direct = [DirectExpertLayout.from_receiver(layout) for layout in layouts]
                if any(item != direct[0] for item in direct) or direct[0].tp_size != 2:
                    raise ValueError("Sharded receivers disagree on their native expert layout")
                self.members = identities
                plan[0] = dict(layout=layouts[0], members=identities, nonce=secrets.token_hex(24))
            except Exception as exc:
                plan[0] = dict(error=f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(plan, src=0, group=group)
        if "error" in plan[0]:
            raise RuntimeError(plan[0]["error"])
        members, nonce = plan[0]["members"], plan[0]["nonce"]
        layout = DirectExpertLayout.from_receiver(plan[0]["layout"])
        planned_at = monotonic()
        self.members = tuple(members)
        if world != 8:
            raise ValueError("Owner-local Qwen3.5 synchronization requires exactly eight source ranks")
        self._gate(
            None if mpu.get_expert_model_parallel_rank() == rank else "Expert owners must follow global rank order"
        )
        if rank == 0 and "nonexpert" not in self.encoders:
            self.encoders["nonexpert"] = StreamingDeltaEncoder(self.budget, bucket_bytes=self.config.bucket_bytes)
        version = self.version + 1
        full = self.config.full_only or self.delta_updates % self.config.anchor_interval == 0
        error, weights = None, None
        try:
            free, _ = torch.cuda.mem_get_info()
            reusable = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            # Snapshots already live in the allocator are counted in used_bytes.
            # Keep an additional margin for conversion/library workspace.
            required = self.budget.limit_bytes - self.budget.used_bytes + (4 << 30)
            if required > free + reusable:
                raise ValueError("Insufficient GPU headroom for the configured shard budget and conversion margin")
            weights = self.weights_getter()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._gate(error)
        error = None
        try:
            self.iterator._bridge_converter.broadcast_and_apply_configs()
            for encoder in self.encoders.values():
                encoder.begin(full=full, compare_changes=not self.config.full_only)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._gate(error)
        local: dict[str, list] = {}
        error = None
        try:
            with torch.no_grad():
                with closing(self.iterator.iter_owner_local_expert_tensors(weights, grouped=True)) as named:
                    with closing(iter_owner_expert_blocks(named, layout, rank, world, self.budget)) as blocks:
                        for name, packed in blocks:
                            for tp in range(2):
                                chunk = expert_slice_chunk([(name, packed)], layout, tp)
                                if chunk is None or len(chunk) != 1:
                                    raise ValueError("Owner block did not map to one contiguous native TP slice")
                                descriptor, shard = chunk[0]
                                self.encoders[f"expert:{tp}"].add(descriptor, shard)
                                del chunk, shard
                            del packed
                for tp in range(2):
                    key = f"expert:{tp}"
                    local[key] = self.encoders[key].finish()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # No rank enters the collective nonexpert converter after an owner error.
        self._gate(error)
        error = None
        named = self.iterator.iter_gated_nonexpert_tensors(weights, error_gate=self._gate)
        try:
            with torch.no_grad():
                for name, tensor in named:
                    if rank == 0:
                        self.encoders["nonexpert"].add(name, tensor)
                if rank == 0:
                    local["nonexpert"] = self.encoders["nonexpert"].finish()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                for _ in named:
                    pass
            except Exception:
                pass
        self._gate(error)
        captured_at = monotonic()
        error = None
        descriptors = {}
        stream_metadata = {}
        expected_members = {}
        try:
            if (
                sum(len(bucket.payload) for buckets in local.values() for bucket in buckets)
                > self.config.host_budget_bytes
            ):
                raise ValueError("Encoded owner streams exceed their shared source host budget")
            for key, buckets in local.items():
                source = self.sources.get(key)
                if source is None:
                    source = MemorySource(
                        ray.util.get_node_ip_address(),
                        timeout_s=self.config.sync_timeout_s,
                        host_budget_bytes=self.config.host_budget_bytes,
                    )
                    self.sources[key] = source
                if key == "nonexpert":
                    stream = f"{self.config.stream}:nonexpert"
                    fixed = {f"{engine}:tp:{tp}" for engine in members for tp in range(2)}
                else:
                    tp = int(key.split(":")[1])
                    stream = f"{self.config.stream}:expert:{rank}:tp:{tp}"
                    fixed = {f"{engine}:tp:{tp}" for engine in members}
                expected_members[key] = fixed
                descriptor = source.publish(
                    buckets,
                    stream=stream,
                    version=version,
                    base=version - 1,
                    schema=self.encoders[key].pending_schema,
                    members=fixed,
                    reset=version == 0,
                    group_update=nonce,
                )
                descriptors[stream] = descriptor
                stream_metadata[stream] = dict(
                    schema=self.encoders[key].pending_schema,
                    group_update=nonce,
                    update=descriptor["update"],
                    members=sorted(fixed),
                    version=version,
                    base=version - 1,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._gate(error)
        all_descriptors = [None] * world
        dist.all_gather_object(all_descriptors, descriptors, group=group)
        published_at = monotonic()
        result = [None]
        acquired = False
        if rank == 0:
            try:
                combined = {key: value for item in all_descriptors for key, value in item.items()}
                if len(combined) != world * 2 + 1:
                    raise ValueError("Incomplete owner stream descriptor set")
                lock_started = monotonic()
                while not ray.get(lock.acquire.remote(), timeout=remaining()):
                    sleep(min(0.05, remaining()))
                acquired = True
                locked_at = monotonic()
                ray.get(
                    [engine.pause_generation.remote(timeout=remaining()) for engine in engines], timeout=remaining()
                )
                paused_at = monotonic()
                controls = []
                for identity in members:
                    per_tp = []
                    for tp in range(2):
                        names = [f"{self.config.stream}:expert:{owner}:tp:{tp}" for owner in range(world)]
                        names.append(f"{self.config.stream}:nonexpert")
                        per_tp.append(
                            dict(
                                stream=self.config.stream,
                                version=version,
                                reset=version == 0,
                                group_update=nonce,
                                owners=world,
                                tp_rank=tp,
                                tp_size=2,
                                host_budget_bytes=self.config.host_budget_bytes,
                                gpu_budget_bytes=self.config.receiver_gpu_budget_bytes,
                                timeout_s=remaining(),
                                streams=[
                                    dict(stream=name, descriptor=combined[name], member=f"{identity}:tp:{tp}")
                                    for name in names
                                ],
                            )
                        )
                    controls.append(per_tp)
                dispatched_at = monotonic()
                loaded = ray.get(
                    [
                        engine.install_sharded_memory_update.remote(control, remaining())
                        for engine, control in zip(engines, controls, strict=True)
                    ],
                    timeout=remaining(),
                )
                if any(not isinstance(item, dict) or item.get("success") is not True for item in loaded):
                    raise RuntimeError(f"Incomplete sharded engine install: {loaded}")
                installed_at = monotonic()
                result[0] = dict(
                    receivers=loaded,
                    receiver_calls_s=installed_at - paused_at,
                    stream_union_s=lock_started - published_at,
                    lock_wait_s=locked_at - lock_started,
                    pause_s=paused_at - locked_at,
                    receiver_control_prepare_s=dispatched_at - paused_at,
                    install_calls_s=installed_at - dispatched_at,
                )
            except Exception as exc:
                result[0] = dict(error=f"{type(exc).__name__}: {exc}")
        broadcast_started = monotonic()
        dist.broadcast_object_list(result, src=0, group=group)
        broadcasted_at = monotonic()
        error = result[0].get("error")
        if error is None:
            try:
                for key, source in self.sources.items():
                    self.encoders[key].prepare_commit(source.wait(remaining()), expected_members[key])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        completion_error = None
        lock_release_s = 0.0
        try:
            self._gate(error)
            ack_prepared_at = monotonic()
            error = None
            try:
                for encoder in self.encoders.values():
                    encoder.commit()
                self.version = version
                if not self.config.full_only:
                    self.delta_updates += 1
                for source in self.sources.values():
                    source.finish()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._gate(error)
            committed_at = monotonic()
            if rank == 0:
                ready = [None]
                try:
                    ray.get(
                        [engine.continue_generation.remote(timeout=remaining()) for engine in engines],
                        timeout=remaining(),
                    )
                    ready_at = monotonic()
                    ray.get(
                        [engine.disk_weight_generation_probe.remote(remaining()) for engine in engines],
                        timeout=remaining(),
                    )
                    ready[0] = dict(
                        sync_to_ready_s=ready_at - started,
                        sync_with_probe_s=monotonic() - started,
                        resume_s=ready_at - committed_at,
                        probe_s=monotonic() - ready_at,
                    )
                except Exception as exc:
                    ready[0] = dict(error=f"{type(exc).__name__}: {exc}")
            else:
                ready = [None]
            dist.broadcast_object_list(ready, src=0, group=group)
            if "error" in ready[0]:
                raise RuntimeError(ready[0]["error"])
        except Exception as exc:
            completion_error = f"{type(exc).__name__}: {exc}"
        finally:
            if rank == 0 and acquired:
                release_started = monotonic()
                try:
                    ray.get(lock.release.remote(), timeout=10)
                except Exception as exc:
                    completion_error = f"Group lock release failed: {type(exc).__name__}: {exc}"
                lock_release_s = monotonic() - release_started
        completion_gate_started = monotonic()
        self._gate(completion_error)
        completion_gate_s = monotonic() - completion_gate_started
        totals = [None] * world
        groups = {}
        for key, buckets in local.items():
            for bucket in buckets:
                label = ("expert" if key.startswith("expert:") else "nonexpert") + ":" + bucket.dtype
                item = groups.setdefault(
                    label,
                    dict(
                        elements=0,
                        changed=0,
                        payload_bytes=0,
                        full_buckets=0,
                        sparse_buckets=0,
                        zero_changed_buckets=0,
                        full_fallback_buckets=0,
                    ),
                )
                item["elements"] += bucket.elements
                item["changed"] += bucket.changed
                item["payload_bytes"] += len(bucket.payload)
                item[bucket.encoding + "_buckets"] += 1
                item["zero_changed_buckets"] += int(bucket.changed == 0)
                item["full_fallback_buckets"] += int(bucket.encoding == "full" and not full)
        if self.config.full_only:
            for item in groups.values():
                item["changed"] = None
                item["zero_changed_buckets"] = None
                item["full_fallback_buckets"] = None
        local_stats = dict(
            groups=groups,
            streams=stream_metadata,
            layout_and_membership_s=planned_at - started,
            capture_body_s=captured_at - planned_at,
            install_result_broadcast_s=broadcasted_at - broadcast_started,
            ack_wait_and_prepare_s=ack_prepared_at - broadcasted_at,
            commit_and_gate_s=committed_at - ack_prepared_at,
            lock_release_s=lock_release_s,
            completion_gate_s=completion_gate_s,
            elements=sum(b.elements for values in local.values() for b in values),
            changed=None if self.config.full_only else sum(b.changed for values in local.values() for b in values),
            payload_bytes=sum(len(b.payload) for values in local.values() for b in values),
            expert_payload_bytes=sum(
                len(b.payload) for key, values in local.items() if key.startswith("expert:") for b in values
            ),
            nonexpert_payload_bytes=sum(len(b.payload) for b in local.get("nonexpert", [])),
            tcp_control_bytes=sum(source.last_control_bytes for source in self.sources.values()),
            clone_enqueue_s=sum(encoder.clone_enqueue_s for encoder in self.encoders.values()),
            pack_diff_to_host_s=sum(encoder.pack_diff_to_host_s for encoder in self.encoders.values()),
            snapshot_peak_bytes=self.budget.peak_bytes,
            cuda_allocated_bytes=torch.cuda.memory_allocated(),
            cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        dist.all_gather_object(totals, local_stats, group=group)
        metrics = dict(
            stream=self.config.stream,
            version=version,
            full=full,
            members=len(members),
            capture_encode_s=captured_at - started,
            publish_s=published_at - captured_at,
            source_ranks=totals,
            **result[0],
            **ready[0],
        )
        if rank == 0:
            logger.info("Sharded memory weight sync: %s", metrics)
        return metrics
