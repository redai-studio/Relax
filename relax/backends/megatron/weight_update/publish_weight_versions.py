# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Training-side full conversion with CPU publication and collective
finalization."""

import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

from relax.distributed.weight_sync import traffic
from relax.distributed.weight_sync.checkpoint import snapshot_tensors
from relax.distributed.weight_sync.diagnostics import save_diagnostic
from relax.distributed.weight_sync.metrics import SyncMetrics
from relax.distributed.weight_sync.publication import PublicationHandle, PublishedVersion, VersionPublisher
from relax.distributed.weight_sync.scratch import ScratchWorkspace
from relax.distributed.weight_sync.source_io import agree_errors, agree_io_state, coordinated_io
from relax.distributed.weight_sync.store import VersionStore, atomic_write, canonical, digest
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


class MegatronVersionPublisher:
    """All converter participants call submit/finalize on the training thread.

    The iterator performs its existing training-local collectives. Only rank
    zero persists logical tensors. The background executor receives files,
    never CUDA tensors or process groups.
    """

    def __init__(
        self,
        iterator: Any,
        weights_getter: Callable[[], Mapping[str, Any]],
        process_group: Any,
        store: VersionStore,
        stream: str,
        scratch: str | Path,
        *,
        model_config: dict,
        converter_revision: str,
        compression: bool = False,
        encode_workers: int = 4,
        anchor_interval: int = 20,
        full_only: bool = False,
        source_cache_root: str | Path | None = None,
        source_limits: dict[str, int] | None = None,
        source_io_timeout: float = 3600,
    ) -> None:
        import torch.distributed as dist

        if dist.get_backend(process_group) != "gloo":
            raise ValueError("Publication requires an explicit Gloo control group")
        self.iterator, self.weights_getter = iterator, weights_getter
        self.group, self.store = process_group, store
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.source = dist.get_global_rank(process_group, 0)
        self.scratch = Path(scratch)
        self.source_cache_root = Path(source_cache_root) if source_cache_root is not None else self.scratch
        self.source_limits = source_limits
        self.encode_workers = encode_workers
        self.source_io_timeout = source_io_timeout
        self.model_metadata_bytes = len(canonical(model_config))
        self.anchor_interval, self.full_only = anchor_interval, full_only
        self.identity = digest(
            canonical(
                {
                    "stream": stream,
                    "config": model_config,
                    "converter": converter_revision,
                    "compression": compression,
                    "encode_workers": encode_workers,
                    "anchor_interval": anchor_interval,
                    "full_only": full_only,
                    "source_io_timeout": source_io_timeout,
                }
            )
        )
        self.publisher = None
        self.export_workspace = None

        def open_export_workspace():
            # Own the lease before collective agreement can raise on a peer.
            self.export_workspace = ScratchWorkspace(self.source_cache_root, f"export:{stream}:{self.rank}")

        try:
            self._file_io(open_export_workspace, stage="export scratch recovery")
        except BaseException:
            if self.export_workspace:
                self.export_workspace.close()
            raise
        # Admit every rank's workspace before initializing the publication owner.
        error = None
        try:
            if self.rank == 0:
                self.publisher = VersionPublisher(
                    store,
                    stream,
                    self.scratch,
                    model_config=model_config,
                    converter_revision=converter_revision,
                    compression=compression,
                    encode_workers=encode_workers,
                    anchor_interval=anchor_interval,
                    full_only=full_only,
                )
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
        failures = self._agree(error)
        if any(failures):
            # A restored candidate may still be writing when another rank's
            # setup fails. Keep the healthy control group active while its
            # owner waits for the background writer to release the lease.
            self._owner_io(lambda: self.publisher.close(finalize=False) if self.publisher else None)
            self.export_workspace.close()
            raise RuntimeError(f"Source publication initialization failed: {failures}")
        pending = self._broadcast(
            asdict(self.publisher.pending) if self.publisher and self.publisher.pending else None
        )
        self.source_epoch = self._broadcast(uuid.uuid4().hex if self.rank == 0 else None)
        self.stream = stream
        self.pending: PublicationHandle | None = PublicationHandle(**pending) if pending else None
        self.metrics: SyncMetrics | None = SyncMetrics(self.pending.update_id) if self.pending else None
        if self.metrics:
            self.metrics.source_participant = {"rank": self.rank, "world_size": self.world_size}

    def _agree(self, value: object) -> list:
        import torch.distributed as dist

        if self.world_size == 1:
            return [value]
        values = [None] * dist.get_world_size(self.group)
        with traffic.capture(traffic.current() or getattr(self, "metrics", None)):
            traffic.all_gather_object(values, value, group=self.group)
        return values

    def _broadcast(self, value: object) -> object:
        values = [value]
        with traffic.capture(traffic.current() or getattr(self, "metrics", None)):
            traffic.broadcast_object_list(values, src=self.source, group=self.group)
        return values[0]

    def _check(self, error: str | None) -> None:
        with traffic.capture(traffic.current() or getattr(self, "metrics", None)):
            failures = agree_errors(error, self.group)
        if any(failures):
            raise RuntimeError(f"Logical export failed: {failures}")

    def _file_io(self, operation, *, stage, committing=False):
        def agreement(state):
            with traffic.capture(traffic.current() or getattr(self, "metrics", None)):
                return agree_io_state(state, self.group)

        return coordinated_io(operation, agreement, stage=stage, timeout=self.source_io_timeout, committing=committing)

    def _owner_io(self, operation: Callable[[], Any]) -> Any:
        try:
            value = self._file_io(
                operation if self.rank == 0 else None, stage="publication owner I/O", committing=True
            )
        except RuntimeError as error:
            if str(error).startswith("publication owner I/O failed:"):
                # File workers have drained and the control group is healthy.
                # A submit failure after journaling must expose its retryable
                # candidate on every rank, even though submit returned no handle.
                pending = self._broadcast(
                    asdict(self.publisher.pending) if self.publisher and self.publisher.pending else None
                )
                self.pending = PublicationHandle(**pending) if pending else None
            raise
        return self._broadcast(value)

    def submit(self, update_id: str, *, source: dict | None = None) -> PublicationHandle:
        try:
            return self._submit(update_id, source=source)
        except Exception:
            self._save_failed_metrics(str(update_id))
            raise

    def _save_failed_metrics(self, update_id: str) -> None:
        if self.metrics and self.metrics.update_id == update_id:
            self.metrics.unmeasured(
                "G/W/C", "failed operation: partial collective or filesystem traffic requires calibration"
            )
            try:
                self.metrics.save(self.scratch / f"metrics-failed-{self.metrics.operation_id}-rank-{self.rank}.json")
            except OSError:
                logger.warning("Failed source attempt; diagnostic output is unavailable", exc_info=True)

    def _submit(self, update_id: str, *, source: dict | None = None) -> PublicationHandle:
        metrics = traffic.current() or SyncMetrics(str(update_id))
        metrics.update_id = str(update_id)
        with traffic.capture(metrics):
            identities = self._agree(
                (str(update_id), self.identity, source, asdict(self.pending) if self.pending else None)
            )
        if any(value != identities[0] for value in identities):
            raise ValueError("Source ranks disagree on update identity or pending publication")
        if self.pending:
            self.finalize_pending(self.pending)
        self.metrics = metrics
        self.metrics.source_participant = {"rank": self.rank, "world_size": self.world_size}
        frozen, error, temporary = {}, None, None
        weights = None
        record = None
        try:
            import torch
            from megatron.core import mpu

            from relax.distributed.weight_sync.source_budget import source_demand, source_preflight_record

            weights = self.weights_getter()
            size = sum(value.numel() * value.element_size() for value in weights.values())
            if size > self.store.max_weight_bytes:
                raise ValueError("Source snapshot exceeds configured weight limit")
            iterator = self.iterator
            buckets = getattr(iterator, "megatron_local_param_info_buckets", None)
            if buckets is None:
                buckets = getattr(iterator, "_expert_buckets", []) + getattr(iterator, "_non_expert_buckets", [])
            from relax.distributed.weight_sync.codec import MAX_CHUNK_BYTES

            # Names, tensor/chunk descriptors and multiple durable manifests
            # share this allowance. Exact per-file quotas remain enforced by
            # VersionStore as each encoded record becomes available.
            metadata_bytes = (
                64 * 1024
                + 4 * self.model_metadata_bytes
                + sum(8192 + 4 * len(info.name.encode()) for bucket in buckets for info in bucket)
                + 4096 * ((self.store.max_weight_bytes + MAX_CHUNK_BYTES - 1) // MAX_CHUNK_BYTES)
            )
            # Use the source shapes before Bridge cropping or tied-weight
            # conversion. The raw cache and logical checkpoint have different
            # sizes even when they describe the same model.
            bucket_sizes = [
                sum(
                    info.size
                    * (
                        (
                            mpu.get_expert_tensor_parallel_world_size()
                            if ".experts." in info.name
                            else mpu.get_tensor_model_parallel_world_size()
                        )
                        if info.attrs.get("tensor_model_parallel") and info.attrs.get("parallel_mode") != "duplicated"
                        else 1
                    )
                    for info in bucket
                )
                for bucket in buckets
            ]
            largest = max(bucket_sizes, default=size)
            bridge = hasattr(iterator, "finish_group")
            demand = source_demand(
                size,
                self.store.max_weight_bytes if self.rank == 0 else 0,
                largest,
                adaptive=False,
                bridge=bridge,
                gathered_source_bytes=sum(bucket_sizes) if self.rank == 0 else 0,
                encode_workers=self.encode_workers if self.rank == 0 else 1,
            )
            from relax.distributed.weight_sync.source_budget import gpu_capacity

            gpu = gpu_capacity(torch.cuda, demand["gpu_bytes"]) if torch.cuda.is_available() else None
            record = source_preflight_record(
                self.scratch,
                demand,
                limits=self.source_limits,
                gpu_free=gpu["available_bytes"] if gpu else None,
                store=self.store if self.rank == 0 else None,
                logical_bytes=self.store.max_weight_bytes if self.rank == 0 else 0,
                metadata_bytes=metadata_bytes,
                source_cache=self.source_cache_root,
            )
            record["gpu_capacity"] = gpu
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
        records = self._agree({"record": record, "error": error})
        if any(row["error"] for row in records):
            raise RuntimeError(f"Source preflight failed: {[row['error'] for row in records]}")
        from relax.distributed.weight_sync.source_budget import check_source_preflight

        preflight = check_source_preflight([row["record"] for row in records])
        preflight["update_id"] = str(update_id)
        save_diagnostic(
            self.scratch / f"capacity-{self.metrics.operation_id}-rank-{self.rank}.json", canonical(preflight)
        )
        if preflight["warnings"] and self.rank == 0:
            logger.warning(
                "Estimated conversion memory exceeds currently available memory: %s. "
                "Continuing publication; configured limits and actual allocation errors remain effective.",
                preflight["warnings"],
            )
        if preflight["shortages"]:
            raise OSError(f"Source capacity unavailable: {preflight['shortages']}. Actions: {preflight['actions']}")
        try:
            with traffic.capture(self.metrics), self.metrics.stage("source_snapshot", path=self.scratch):
                size = sum(value.numel() * value.element_size() for value in weights.values())
                if size > self.store.max_weight_bytes:
                    raise ValueError("Source snapshot exceeds configured weight limit")
                frozen = {name: value.detach().to(device="cpu", copy=True) for name, value in weights.items()}
                temporary = self.export_workspace.temporary()
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
        failures = self._agree(error)
        if any(failures):
            if temporary:
                self.export_workspace.remove(temporary)
            raise RuntimeError(f"Source snapshot failed: {failures}")
        table, local_error = {}, None
        try:
            with traffic.capture(self.metrics), self.metrics.stage("source_conversion", path=self.scratch):
                traffic.emit("G", 0, "source_conversion_entered")
                checked = getattr(self.iterator, "get_hf_weight_chunks_checked", None)
                buckets = checked(frozen, self._check) if checked else self.iterator.get_hf_weight_chunks(frozen)
                for bucket in buckets:
                    host, copy_error = None, None
                    try:
                        if self.rank == 0:
                            host = [(name, tensor.detach().contiguous().cpu()) for name, tensor in bucket]
                    except Exception as caught:
                        copy_error = f"{type(caught).__name__}: {caught}"
                    self._check(copy_error)

                    def write_export():
                        exported = snapshot_tensors(host, temporary)
                        if set(table) & set(exported):
                            raise ValueError("Converter produced duplicate logical tensors")
                        table.update(exported)

                    self._file_io(write_export if self.rank == 0 else None, stage="logical export write")
            failures = self._agree(local_error)
            if any(failures):
                raise RuntimeError(f"Logical export failed: {failures}")
            with traffic.capture(self.metrics), self.metrics.stage("candidate_persistence", path=self.scratch):
                handle = self._owner_io(lambda: asdict(self.publisher.submit(str(update_id), table, source=source)))
            self.pending = PublicationHandle(**handle)
            return self.pending
        except BaseException:
            logger.exception("Weight export failed for update %s on rank %s", update_id, self.rank)
            raise
        finally:
            del frozen
            # An iterator can fail before its peers reach the same collective.
            # File workers have drained; this rank owns only its export scratch.
            # Cleanup must preserve the original error and need no peer progress.
            try:
                self.export_workspace.remove(temporary)
            except OSError:
                logger.warning("Export scratch cleanup pending: %s", temporary, exc_info=True)

    def finalize_pending(self, handle: PublicationHandle) -> PublishedVersion:
        identities = self._agree(asdict(handle))
        if handle != self.pending or any(value != identities[0] for value in identities):
            raise ValueError("Source ranks disagree on publication handle")
        with traffic.capture(self.metrics), self.metrics.stage("publication_wait", path=self.store.root):
            started = time.monotonic()
            next_notice = started + 60
            while True:
                result = None
                if self.rank == 0:
                    try:
                        result = (
                            {"published": asdict(self.publisher.wait_committed(handle)), "error": None}
                            if self.publisher.ready(handle)
                            else {"waiting": True}
                        )
                    except Exception as caught:
                        result = {"error": f"{type(caught).__name__}: {caught}"}
                result = self._broadcast(result)
                if not result.get("waiting"):
                    break
                now = time.monotonic()
                if self.rank == 0 and now >= next_notice:
                    logger.warning(
                        "Publication %s (version %s, update %s) is still encoding or writing after %.1fs; "
                        "the retained candidate is pending STORE_COMMITTED. "
                        "Inspect publication-worker progress and storage latency; confirmation continues.",
                        handle.candidate_id,
                        handle.version,
                        handle.update_id,
                        now - started,
                    )
                    next_notice = now + 60
                time.sleep(0.5)
            if result["error"]:
                raise RuntimeError(f"Publication {handle.candidate_id}: {result['error']}")
            published = PublishedVersion(**result["published"])
            self._file_io(
                lambda: atomic_write(
                    self.scratch / f"source-base-rank-{self.rank}.json", canonical(asdict(published))
                ),
                stage="source-base persistence",
            )
            self._owner_io(lambda: self.publisher.confirm(handle, published))
        self.pending = None
        # SOURCE_CONFIRMED is already durable on every participant. Diagnostic
        # I/O must not turn this completed publication into a rank-local error.
        try:
            self.metrics.save(self.scratch / f"metrics-v{published.version}-rank-{self.rank}.json")
        except OSError:
            logger.warning(
                "Version %s is published; diagnostic output failed on rank %s. Check scratch capacity and permissions.",
                published.version,
                self.rank,
                exc_info=True,
            )
        return published

    def publish(self, update_id: str | None = None, *, progress: dict | None = None) -> PublishedVersion:
        if self.pending:
            self.finalize_pending(self.pending)
        with traffic.capture(SyncMetrics(update_id or "capture-pending")):
            try:
                return self._publish(update_id, progress=progress)
            except Exception:
                if self.metrics:
                    self._save_failed_metrics(self.metrics.update_id)
                raise

    def _publish(self, update_id: str | None, *, progress: dict | None) -> PublishedVersion:
        version = self._owner_io(lambda: self.publisher.next_version())
        source = None
        if progress is not None:
            source = {
                "run_id": self.stream,
                "source_epoch": self.source_epoch,
                "capture_seq": version,
                "snapshot_kind": "update"
                if progress["optimizer_attempts"]
                else ("init" if version == 0 else "recovery"),
                **progress,
            }
        if update_id is None:
            update_id = f"{self.stream}:{self.source_epoch}:{version}"
        return self.finalize_pending(self.submit(update_id, source=source))

    def close(self) -> None:
        try:
            if self.pending:
                self.finalize_pending(self.pending)
        finally:
            # Confirmation belongs to finalize_pending on every participant.
            # Cleanup after a failed collective must retain the pending record.
            if self.publisher:
                self.publisher.close(finalize=False)
            self.export_workspace.close()
