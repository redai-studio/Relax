# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Ordinary training timing: no snapshots, complements or extra FULL
controls."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import Any


def install(args: Any) -> None:
    import ray
    import torch
    import torch.distributed as dist

    from relax.distributed.weight_sync.provenance import runtime_identity
    from relax.distributed.weight_sync.store import atomic_write
    from relax.utils.distributed_utils import get_gloo_group
    from relax.utils.logging_utils import get_logger
    from task_delta_weight_sync.native_call import guarded_native_call
    from task_delta_weight_sync.prepared_nccl import wire_metadata, wire_schema_digest
    from task_delta_weight_sync.sync_metrics import (
        allocator_metrics,
        optimizer_boundary,
        receiver_metrics,
        validate_stream_receipts,
    )

    memory = bool(getattr(args, "memory_weight_sync_config", None))
    if memory:
        from relax.backends.megatron.weight_update.update_weight_from_sharded_memory import (
            UpdateWeightFromShardedMemory,
        )

        cls = UpdateWeightFromShardedMemory
        from relax.distributed.weight_sync.config import MemorySyncConfig

        if MemorySyncConfig.load(args.memory_weight_sync_config).layout != "qwen35_owner_local":
            raise ValueError("Ordinary sharded measurements require the owner-local 35B layout")
    else:
        from relax.backends.megatron.weight_update.update_weight_from_distributed import UpdateWeightFromDistributed

        cls = UpdateWeightFromDistributed
    group, logger = get_gloo_group(), get_logger(__name__)
    rank = dist.get_rank(group)
    observe = os.environ.get("DWS_COLLECT_METRICS") == "1"
    root = Path(os.environ["TENSORBOARD_DIR"]).resolve().parents[1] / "sync-measurements"

    def gate(error):
        errors = [None] * dist.get_world_size(group)
        dist.all_gather_object(errors, error, group=group)
        if any(errors):
            raise RuntimeError(f"Ordinary synchronization measurement failed: {errors}")

    error = None
    try:
        atomic_write(root / "ranks" / f"{rank}.json", json.dumps(runtime_identity(), indent=2).encode())
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    gate(error)
    args._dws_optimizer_observer = True
    args._dws_optimizer_result = None
    if not memory:
        supervision = dist.new_group(
            ranks=dist.get_process_group_ranks(group), backend="gloo", timeout=timedelta(seconds=60)
        )
        original_bucket = cls._update_bucket_weights_from_distributed

        def bucket(self, named, *call_args, **kwargs):
            self._normal_source_bytes += sum(t.numel() * t.element_size() for _, t in named)
            self._normal_wire_metadata.append(wire_metadata(named))
            return original_bucket(self, named, *call_args, **kwargs)

        cls._update_bucket_weights_from_distributed = bucket
    original = cls.update_weights

    def quarantine(self):
        errors = []
        engines = (self.engines or getattr(self, "_normal_metric_engines", ())) if memory else self.rollout_engines
        for engine in engines:
            try:
                ray.get(engine.pause_generation.remote(timeout=5), timeout=10)
            except Exception:
                try:
                    ray.kill(engine, no_restart=True)
                except Exception as exc:
                    errors.append(str(exc))
        return errors

    def update(self, *call_args, **kwargs):
        entered_at = monotonic()
        if not memory:
            error = None
            if rank == 0:
                identities = [engine._actor_id.hex() for engine in self.rollout_engines]
                if len(identities) != 4 or len(set(identities)) != 4 or list(self._engine_gpu_counts or []) != [2] * 4:
                    error = "Ordinary native comparison requires four distinct TP2 engines"
            try:
                gate(error)
            except Exception:
                if rank == 0:
                    quarantine(self)
                raise
        self._normal_source_bytes = 0
        self._normal_wire_metadata = []
        self._normal_step = getattr(self, "_normal_step", -1) + 1
        metric_started = monotonic()
        metric_id = f"optimizer-{self._normal_step:06d}"
        metric_receivers_start = None
        source_start = None
        source_starts = None
        metric_arm = "memory" if memory else "nccl"
        error = None
        if observe:
            try:
                if rank == 0:
                    if memory:
                        manager = call_args[0] if call_args else kwargs["rollout_manager"]
                        engines, _, _, counts, _ = ray.get(manager.get_rollout_engines_and_lock.remote(), timeout=60)
                        self._normal_metric_engines = tuple(engines)
                        if list(counts) != [2] * 4 or len({e._actor_id.hex() for e in engines}) != 4:
                            raise ValueError("Ordinary measurements require four distinct TP2 engines")
                    else:
                        engines = self.rollout_engines
                    self._normal_metric_engines = tuple(engines)
                    metric_receivers_start = receiver_metrics(
                        engines, root, metric_id, f"{metric_arm}-metrics-start", start=True
                    )
                source_start = allocator_metrics(reset=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            try:
                gate(error)
            except Exception:
                if rank == 0:
                    quarantine(self)
                raise
            source_starts = [None] * dist.get_world_size(group)
            dist.all_gather_object(source_starts, source_start, group=group)
        metric_setup_s = monotonic() - metric_started
        started = monotonic()
        value = (
            original(self, *call_args, **kwargs)
            if memory
            else guarded_native_call(
                lambda: original(self, resume_generation=False),
                process_group=supervision,
                timeout_s=600,
                failure_path=root / f"native-failure-rank-{rank}.json",
                quarantine=(lambda: quarantine(self)) if rank == 0 else None,
            )
        )
        returned = monotonic()
        error, report = None, None
        if rank == 0:
            try:
                if memory:
                    report = dict(value, arm="full_tcp" if self.config.full_only else "delta_tcp")
                    expert = sum(r["expert_payload_bytes"] for r in value["source_ranks"])
                    nonexpert = sum(r["nonexpert_payload_bytes"] for r in value["source_ranks"])
                    report["logical_payload_bytes"] = 4 * expert + 8 * nonexpert
                else:
                    engines = self.rollout_engines
                    fence_started = monotonic()
                    control = dict(
                        command="fence",
                        phase="ordinary-fence",
                        update_id=f"optimizer-{self._normal_step:06d}",
                        paused=True,
                        directory=str(root.parent / "completion-fences"),
                        node_budget_bytes=384 << 30,
                    )
                    ray.get([engine.run_memory_audit.remote(control, 120) for engine in engines], timeout=130)
                    fenced_at = monotonic()
                    ray.get([engine.continue_generation.remote(timeout=120) for engine in engines], timeout=130)
                    ready = monotonic()
                    ray.get([engine.disk_weight_generation_probe.remote(120) for engine in engines], timeout=130)
                    report = dict(
                        arm="nccl",
                        source_tensor_bytes=self._normal_source_bytes,
                        logical_payload_bytes=sum(self._engine_gpu_counts) * self._normal_source_bytes,
                        receiver_workers=sum(self._engine_gpu_counts),
                        engine_gpu_counts=list(self._engine_gpu_counts),
                        receivers=[engine._actor_id.hex() for engine in engines],
                        version=self.weight_version,
                        sync_to_ready_s=ready - started,
                        sync_with_probe_s=monotonic() - started,
                        fence_s=fenced_at - fence_started,
                        resume_s=ready - fenced_at,
                        probe_s=monotonic() - ready,
                        changed=None,
                        changed_scope="No extra diff on the native FULL control",
                    )
                completed_at = monotonic()
                report.update(
                    optimizer_update=self._normal_step,
                    initialization=self._normal_step == 0,
                    updater_return_s=returned - started,
                    completion_return_s=completed_at - started,
                    optimizer_boundary=optimizer_boundary(args, entered_at, completed_at),
                    native_return_scope=None
                    if memory
                    else "resume_generation=False; continue/fence/probe included in common completion timings",
                    no_audit=True,
                    payload_scope="Logical shard/member deliveries; not NIC traffic",
                    memory_peak_scope="Process lifetime allocator high-water marks; not per-arm incremental peaks",
                )
                if not memory:
                    report["wire_schema_sha256"] = wire_schema_digest(self._normal_wire_metadata)
            except Exception as exc:
                errors = quarantine(self)
                error = f"{type(exc).__name__}: {exc}; quarantine_errors={errors}"
        gate(error)
        if observe:
            metric_started = monotonic()
            error, local = None, None
            try:
                local = allocator_metrics(reset=False)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            try:
                gate(error)
            except Exception:
                if rank == 0:
                    quarantine(self)
                raise
            sources = [None] * dist.get_world_size(group)
            dist.all_gather_object(sources, local, group=group)
            error = None
            if rank == 0:
                try:
                    report["observations"] = receiver_metrics(
                        self._normal_metric_engines, root, metric_id, f"{metric_arm}-metrics-end", start=False
                    )
                    report["observations"]["source_ranks"] = sources
                    report["observations_start"] = dict(metric_receivers_start, source_ranks=source_starts)
                    if memory:
                        report["wire_schema_check"] = validate_stream_receipts(
                            report, report["observations"]["workers"]
                        )
                    report["metrics_setup_s"] = metric_setup_s
                    report["metrics_collection_s"] = monotonic() - metric_started
                    report["memory_peak_scope"] = "Per-arm GPU reset; lifetime host peaks explicitly labeled"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}; quarantine_errors={quarantine(self)}"
            gate(error)
        elif not memory:
            local_memory = dict(
                rank=rank,
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            memories = [None] * dist.get_world_size(group)
            dist.all_gather_object(memories, local_memory, group=group)
            if rank == 0:
                report["source_ranks"] = memories
        error = None
        if rank == 0:
            try:
                atomic_write(root / f"{self._normal_step:06d}.json", json.dumps(report, indent=2).encode())
                logger.info("Ordinary sharded sync measurement: %s", report)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}; quarantine_errors={quarantine(self)}"
        gate(error)
        return value

    cls.update_weights = update
