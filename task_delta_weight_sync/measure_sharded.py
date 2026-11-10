# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Matched frozen-target TCP/NCCL trials and a disk-free raw GPU oracle.

Delta always runs first. Full controls alternate order; retained CPU snapshots
have the same residency during all stable measurements. Audit GPU copies,
complement and the extra ordinary FULL reference stay outside all arm timers.
"""

from __future__ import annotations

import importlib
import json
import os
from datetime import timedelta
from pathlib import Path
from time import monotonic, sleep
from typing import Any


def install(args: Any) -> None:
    import ray
    import torch
    import torch.distributed as dist

    from relax.backends.megatron.weight_update.update_weight_from_memory import UpdateWeightFromMemory
    from relax.backends.megatron.weight_update.update_weight_from_sharded_memory import UpdateWeightFromShardedMemory
    from relax.distributed.weight_sync.config import MemorySyncConfig
    from relax.distributed.weight_sync.provenance import runtime_identity
    from relax.distributed.weight_sync.store import atomic_write
    from relax.utils.distributed_utils import get_gloo_group
    from relax.utils.logging_utils import get_logger
    from task_delta_weight_sync.native_call import guarded_native_call
    from task_delta_weight_sync.prepared_nccl import wire_bucket

    config = MemorySyncConfig.load(args.memory_weight_sync_config)
    if config.layout != "qwen35_owner_local" or config.full_only:
        raise ValueError("Matched sharded measurement requires the delta owner-local configuration")
    cls = UpdateWeightFromShardedMemory
    if getattr(cls, "_dws_paired", False):
        return
    group, logger = get_gloo_group(), get_logger(__name__)
    rank = dist.get_rank(group)
    root = Path(os.environ["TENSORBOARD_DIR"]).resolve().parents[1]
    measurements, audit_root = root / "paired-measurements", root / "audit"
    native_module = importlib.import_module("relax.backends.megatron.weight_update.update_weight_from_distributed")
    native_cls = native_module.UpdateWeightFromDistributed
    # Capture this bound implementation before any measurement wrapper. The
    # reference call must stay paused and must never trigger an implicit probe.
    native_update = native_cls.update_weights
    events: list[dict] = []

    def broadcast(group_name, process_group, version, engines, named):
        refs = [
            engine.update_weights_from_distributed.remote(
                names=[name for name, _ in named],
                dtypes=[tensor.dtype for _, tensor in named],
                shapes=[tensor.shape for _, tensor in named],
                group_name=group_name,
                weight_version=str(version),
            )
            for engine in engines
        ]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        host_started = monotonic()
        handles = [
            native_module.dist.broadcast(tensor.data, 0, group=process_group, async_op=True) for _, tensor in named
        ]
        for handle in handles:
            handle.wait()
        end.record()
        events.append(
            dict(
                start=start,
                end=end,
                host_enqueue_s=monotonic() - host_started,
                bytes=sum(t.numel() * t.element_size() for _, t in named),
                wire_plan=wire_bucket(named),
            )
        )
        return refs

    native_module.update_weights_from_distributed = broadcast
    original = cls.update_weights

    def root_action(self, action):
        error, value = None, None
        if rank == 0:
            try:
                value = action()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        self._gate(error)
        return value

    def paused_action(self, command: str, update_id: str, phase: str, *, stay_paused: bool = False):
        def act():
            deadline = monotonic() + config.sync_timeout_s
            lock = self._ab_lock
            acquired = False

            def remaining():
                value = deadline - monotonic()
                if value <= 0:
                    raise TimeoutError("Paused audit group deadline exceeded")
                return value

            try:
                while not ray.get(lock.acquire.remote(), timeout=remaining()):
                    sleep(min(0.05, remaining()))
                acquired = True
                ray.get(
                    [e.pause_generation.remote(timeout=remaining()) for e in self._ab_engines], timeout=remaining()
                )
                control = dict(
                    command=command,
                    update_id=update_id,
                    phase=phase,
                    paused=True,
                    directory=str(audit_root),
                    node_budget_bytes=384 << 30,
                )
                replies = ray.get(
                    [e.run_memory_audit.remote(control, remaining()) for e in self._ab_engines], timeout=remaining()
                )
                if any(item.get("success") is not True for item in replies):
                    raise RuntimeError(f"Incomplete TP audit: {replies}")
                if command != "fence":
                    for engine in self._ab_engines:
                        for tp in range(2):
                            member = f"{engine._actor_id.hex()}:tp:{tp}"
                            report = json.loads((audit_root / member / f"{update_id}-{phase}.json").read_text())
                            if report.get("success") is not True:
                                raise RuntimeError(f"Failed worker audit: {report}")
                            if command == "compare":
                                inventory = json.loads((audit_root / member / "inventory.json").read_text())
                                if (
                                    report.get("compared_bytes") != inventory["snapshot_bytes"]
                                    or report.get("mismatched_bytes") != 0
                                ):
                                    raise RuntimeError("Incomplete raw GPU parameter comparison")
                if not stay_paused:
                    ray.get(
                        [e.continue_generation.remote(timeout=remaining()) for e in self._ab_engines],
                        timeout=remaining(),
                    )
            finally:
                if acquired:
                    ray.get(lock.release.remote(), timeout=10)

        return root_action(self, act)

    def initialize(self, manager):
        setup = [None]
        if rank == 0:
            try:
                values = ray.get(manager.get_rollout_engines_and_lock.remote(), timeout=config.sync_timeout_s)
                engines, lock, _, counts, offsets = values
                if len(engines) != 4 or list(counts) != [2] * 4:
                    raise ValueError("Matched trials require exactly four fixed TP2 engines")
                setup[0] = dict(engines=engines, lock=lock, counts=counts, offsets=offsets)
            except Exception as exc:
                setup[0] = dict(error=f"{type(exc).__name__}: {exc}")
        dist.broadcast_object_list(setup, src=0, group=group)
        if "error" in setup[0]:
            raise RuntimeError(setup[0]["error"])
        values = setup[0]
        self._ab_engines, self._ab_lock = tuple(values["engines"]), values["lock"]
        self.engines = self._ab_engines
        native = native_cls(
            args,
            self.iterator.model,
            self.weights_getter,
            model_name=self.iterator.model_name,
            quantization_config=None,
        )
        error = None
        try:
            native.connect_rollout_engines(
                values["engines"],
                values["lock"],
                engine_gpu_counts=values["counts"],
                engine_gpu_offsets=values["offsets"],
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._gate(error)
        self._ab_native = native
        self._ab_native_control = dist.new_group(
            ranks=dist.get_process_group_ranks(group),
            backend="gloo",
            timeout=timedelta(seconds=min(config.sync_timeout_s, 60)),
        )
        self._ab_step = -1
        error = None
        try:
            atomic_write(
                measurements / "source-runtime" / f"rank-{rank}.json",
                json.dumps(runtime_identity(), indent=2).encode(),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._gate(error)
        paused_action(self, "prepare", "initialization", "prepare")
        root_action(
            self,
            lambda: atomic_write(measurements / "runtime.json", json.dumps(runtime_identity(), indent=2).encode()),
        )

    def native_call(self, update_id: str, phase: str):
        return guarded_native_call(
            lambda: native_update(self._ab_native, resume_generation=False),
            process_group=self._ab_native_control,
            timeout_s=config.sync_timeout_s,
            failure_path=measurements / f"{update_id}-{phase}-rank-{rank}-failure.json",
            quarantine=(lambda: UpdateWeightFromMemory._close_generation(self, self._ab_manager))
            if rank == 0
            else None,
        )

    def timed_native(self, update_id: str) -> dict | None:
        events.clear()
        started = monotonic()
        native_call(self, update_id, "timed-nccl")
        returned = monotonic()
        ready = {}

        def finish():
            control = dict(
                command="fence",
                update_id=update_id,
                phase="nccl-fence",
                paused=True,
                directory=str(audit_root),
                node_budget_bytes=384 << 30,
            )
            ray.get(
                [e.run_memory_audit.remote(control, config.sync_timeout_s) for e in self._ab_engines],
                timeout=config.sync_timeout_s,
            )
            ray.get(
                [e.continue_generation.remote(timeout=config.sync_timeout_s) for e in self._ab_engines],
                timeout=config.sync_timeout_s,
            )
            ready["sync_to_ready_s"] = monotonic() - started
            ray.get(
                [e.disk_weight_generation_probe.remote(config.sync_timeout_s) for e in self._ab_engines],
                timeout=config.sync_timeout_s,
            )

        root_action(self, finish)
        elapsed = monotonic() - started

        def collect():
            rows = []
            for event in events:
                event["end"].synchronize()
                rows.append(
                    dict(
                        sender_collective_stream_s=event["start"].elapsed_time(event["end"]) / 1000,
                        host_enqueue_s=event["host_enqueue_s"],
                        bytes=event["bytes"],
                    )
                )
            payload = sum(item["bytes"] for item in rows)
            return dict(
                arm="nccl",
                sync_call_s=elapsed,
                native_return_s=returned - started,
                native_supervision="ordinary native implementation in its original CUDA stream, plus independent Gloo polling at 20ms intervals",
                collective_buckets=rows,
                sender_collective_stream_s=sum(r["sender_collective_stream_s"] for r in rows),
                source_tensor_bytes=payload,
                logical_payload_bytes=payload * 8,
                receiver_workers=8,
                payload_scope="full unsharded tensors delivered to each TP worker; not measured NIC traffic",
                **ready,
            )

        return root_action(self, collect)

    def tcp_metrics(value: dict, elapsed: float, arm: str) -> dict:
        ranks = value["source_ranks"]
        expert = sum(item["expert_payload_bytes"] for item in ranks)
        nonexpert = sum(item["nonexpert_payload_bytes"] for item in ranks)
        return dict(
            value,
            arm=arm,
            sync_call_s=elapsed,
            logical_payload_bytes=4 * expert + 8 * nonexpert,
            tcp_control_bytes=sum(item["tcp_control_bytes"] for item in ranks),
            engine_control_json_bytes=sum(item["control_json_bytes"] for item in value["receivers"]),
            payload_scope="expert TP shards to four engines; full nonexpert HF to eight TP workers; not NIC traffic",
        )

    def paired(self, manager):
        total_started = monotonic()
        try:
            self._ab_manager = manager
            if not hasattr(self, "_ab_native"):
                initialize(self, manager)
            self._ab_step += 1
            update_id = f"optimizer-{self._ab_step:06d}"
            started = monotonic()
            delta = original(self, manager)
            delta = tcp_metrics(delta, monotonic() - started, "delta_tcp")
            paused_action(self, "capture", update_id, "delta-before-full")
            controls = ("nccl", "full_tcp") if self._ab_step % 2 else ("full_tcp", "nccl")
            results = dict(delta_tcp=delta)
            for arm in controls:
                if arm == "nccl":
                    results[arm] = timed_native(self, update_id)
                else:
                    started = monotonic()
                    value = original(self, manager, full_only=True)
                    results[arm] = tcp_metrics(value, monotonic() - started, arm)
                paused_action(self, "compare", update_id, f"{arm}-compare")
            oracle_started = monotonic()
            paused_action(self, "complement", update_id, "complement-before-reference", stay_paused=True)
            events.clear()
            native_call(self, update_id, "independent-full")
            paused_action(self, "compare", update_id, "independent-full-reference")
            # Only the above successful all-worker reference compare resumes.
            oracle_s = monotonic() - oracle_started
            report = dict(
                optimizer_update=self._ab_step,
                initialization=self._ab_step == 0,
                status="VERIFIED",
                full_control_order=list(controls),
                arms=results,
                extra_reference_s=oracle_s,
                total_with_audit_s=monotonic() - total_started,
                extra_reference_source_tensor_bytes=sum(event["bytes"] for event in events),
                extra_reference_logical_payload_bytes=8 * sum(event["bytes"] for event in events),
                extra_reference_weight_version=self._ab_native.weight_version,
                comparison="all actual native parameter bytes; delta captured before either FULL",
                limitation="delta first; full controls alternate; audits perturb caches; shared CPU/GPU baselines remain resident",
            )
            root_action(
                self,
                lambda: atomic_write(
                    measurements / f"{self._ab_step:06d}.json", json.dumps(report, indent=2).encode()
                ),
            )
            if rank == 0:
                self._ab_wire_plan = [event["wire_plan"] for event in events]
                logger.info(
                    "Matched sharded sync verified: update=%d delta=%.6f full_tcp=%.6f nccl=%.6f",
                    self._ab_step,
                    results["delta_tcp"]["sync_call_s"],
                    results["full_tcp"]["sync_call_s"],
                    results["nccl"]["sync_call_s"],
                )
            return delta
        except Exception as original_error:
            self.failed = True
            cleanup_errors = []
            if rank == 0:
                try:
                    cleanup_errors.extend(UpdateWeightFromMemory._close_generation(self, manager))
                except Exception as exc:
                    cleanup_errors.append(f"Group quarantine: {type(exc).__name__}: {exc}")
            for key, source in self.sources.items():
                try:
                    source.close()
                except Exception as exc:
                    cleanup_errors.append(f"Source {key} close: {type(exc).__name__}: {exc}")
            failure = dict(error=f"{type(original_error).__name__}: {original_error}", cleanup_errors=cleanup_errors)
            try:
                atomic_write(measurements / f"failure-rank-{rank}.json", json.dumps(failure, indent=2).encode())
            except Exception as exc:
                cleanup_errors.append(f"Failure receipt: {type(exc).__name__}: {exc}")
            raise RuntimeError(f"Matched weight-sync experiment failed: {failure}") from original_error

    cls.update_weights = paired
    cls._dws_paired = True
