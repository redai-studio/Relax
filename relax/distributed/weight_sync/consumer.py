# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Prepare complete local checkpoints independently of engine installation."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from relax.utils.logging_utils import get_logger

from .checkpoint import validate_checkpoint, write_checkpoint
from .diagnostics import save_diagnostic
from .metrics import MemorySampler, SyncMetrics
from .store import VersionStore, atomic_write, canonical, load_json, sync_directory, valid_id, writer_lock
from .traffic import capture
from .transport import ContentError, RemoteStoreError, SharedFSTransport, TCPTransport, fetch


logger = get_logger(__name__)


@dataclass(frozen=True)
class PreparedVersion:
    checkpoint: Path
    manifest: dict
    owner: str
    metrics: dict


class VersionConsumer:
    def __init__(
        self,
        transport: SharedFSTransport | TCPTransport,
        cache: VersionStore,
        checkpoint_root: str | Path,
        config: str | Path | None = None,
        *,
        checkpoint_budget: int,
        delivery_stage: str = "X",
    ) -> None:
        if checkpoint_budget <= 0:
            raise ValueError("checkpoint_budget must be positive")
        if delivery_stage not in ("X", "T"):
            raise ValueError("delivery_stage must identify the first target (X) or another target (T)")
        self.delivery_stage = delivery_stage
        self.transport, self.cache = transport, cache
        self.root, self.config = Path(checkpoint_root), Path(config) if config else None
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_budget = checkpoint_budget

    def prepare(self, stream: str, version: int, owner: str, *, full: bool = False) -> PreparedVersion:
        valid_id(owner)
        with writer_lock(self.root / ".preparation", timeout=120):
            self._remove_staging()
            ledger = getattr(self.transport, "ledger", None)
            before = ledger.report() if ledger else None
            received, sent = self.transport.bytes_received, getattr(self.transport, "bytes_sent", 0)
            metrics = SyncMetrics(f"{stream}:{version}")
            try:
                return self._prepare(stream, version, owner, full=full, metrics=metrics)
            except BaseException as error:
                self._record_transfers(metrics, before, received, sent)
                report = {
                    **metrics.report(),
                    "failed": True,
                    "stream": stream,
                    "version": version,
                    "error": f"{type(error).__name__}: {error}",
                }
                try:
                    save_diagnostic(self.root / "failed-metrics" / f"{metrics.operation_id}.json", canonical(report))
                except OSError:
                    logger.warning("Consumer diagnostic output failed", exc_info=True)
                raise

    def _record_transfers(self, metrics, before, received, sent):
        ledger = getattr(self.transport, "ledger", None)
        after = ledger.report() if ledger else None
        if after is not None:
            for stage in ("X", "C"):
                count = sum(
                    after["bytes"][direction][stage] - before["bytes"][direction][stage]
                    for direction in ("sent", "received")
                )
                metrics.transferred(
                    self.delivery_stage if stage == "X" else stage,
                    count,
                    link=type(self.transport).__name__,
                    scope="socket_bytes",
                )
        else:
            metrics.transferred(
                self.delivery_stage, self.transport.bytes_received - received, link=type(self.transport).__name__
            )
            metrics.transferred("C", getattr(self.transport, "bytes_sent", 0) - sent, link="control")
            metrics.unmeasured("C", "SharedFS metadata and repair control require filesystem/endpoint observations")
        return after

    def _remove_staging(self) -> None:
        # This root is exclusive to VersionConsumer. The preparation lock
        # excludes every active writer while crash leftovers are removed.
        for staging in self.root.glob(".checkpoint-*"):
            if staging.is_dir():
                shutil.rmtree(staging)

    def _prepare(self, stream: str, version: int, owner: str, *, full: bool, metrics: SyncMetrics) -> PreparedVersion:
        lease = self.root / ".leases" / f"{owner}.json"
        if not lease.exists():
            atomic_write(lease, canonical({"owner": owner, "stream": stream, "version": version, "checkpoint": None}))
        received, sent = self.transport.bytes_received, getattr(self.transport, "bytes_sent", 0)
        ledger = getattr(self.transport, "ledger", None)
        ledger_before = ledger.report() if ledger else None
        with MemorySampler() as memory, capture(metrics):
            with metrics.stage("fetch_and_materialize", path=self.cache.root):
                try:
                    manifest = fetch(self.transport, self.cache, stream, version, full=full, retention_owner=owner)
                except (ContentError, FileNotFoundError, RemoteStoreError) as error:
                    repairable = not isinstance(error, RemoteStoreError) or error.error_type == "FileNotFoundError"
                    if full or not repairable:
                        raise
                    manifest = fetch(self.transport, self.cache, stream, version, full=True, retention_owner=owner)
            metrics.update_id = manifest.get("update_id") or metrics.update_id
            try:
                destination = self.root / f"{stream}-{version}-{manifest['root']}-{manifest['artifact_id'][:16]}"
                verified = False
                # FULL and DELTA deliveries can reconstruct the same logical
                # checkpoint. Validate and share its bytes while each delivery
                # keeps its own manifest and owner lease.
                for candidate in sorted(self.root.glob(f"{stream}-{version}-{manifest['root']}-*")):
                    if candidate.is_symlink() or not candidate.is_dir():
                        continue
                    try:
                        with metrics.stage("checkpoint_reuse_validation", path=candidate):
                            validate_checkpoint(candidate, manifest)
                    except (OSError, ValueError):
                        continue
                    destination = candidate
                    verified = True
                    break
                total = sum(c["size"] for t in manifest["tensors"] for c in t["chunks"])
                used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
                if not destination.exists() and used + total + 16 * 1024**2 > self.checkpoint_budget:
                    raise OSError(
                        "Checkpoint budget exhausted; release unused prepared checkpoints or increase capacity"
                    )
                record = {
                    "checkpoint": destination.name,
                    "root": manifest["root"],
                    "owner": owner,
                    "manifest": manifest,
                }
                # Persist the destination before rename can make it visible.
                # Restart can then reclaim it even if prepare never returned.
                atomic_write(lease, canonical(record))
                with metrics.stage("checkpoint_write", path=self.root):
                    if not destination.exists():
                        # write_checkpoint validates every byte in staging
                        # before its atomic rename. Do not immediately hash
                        # the same model a second time under this same lock.
                        write_checkpoint(self.cache, stream, version, destination, config_source=self.config)
                    elif not verified:
                        # An existing but invalid destination must still fail;
                        # a path's existence is never proof of its contents.
                        validate_checkpoint(destination, manifest)
            except BaseException:
                self.cache.unpin(owner)
                raise
        ledger_after = self._record_transfers(metrics, ledger_before, received, sent)
        report = {
            **metrics.report(),
            "counter_scope": ledger_after["measurement_level"] if ledger_after else "application",
            "measurement_level": ledger_after["measurement_level"] if ledger_after else "application_events",
            "host_memory": memory.peak,
            "memory_sampling": memory.sampling,
            "stream": stream,
            "version": version,
            "root": manifest["root"],
            "artifact_id": manifest["artifact_id"],
            "delivery_stage": self.delivery_stage,
            "socket_ledger_before": ledger_before,
            "socket_ledger_after": ledger_after,
        }
        try:
            save_diagnostic(self.root / "metrics" / f"{metrics.operation_id}.json", canonical(report))
            save_diagnostic(destination / "prepare-metrics.json", canonical(report))
        except OSError:
            logger.warning(
                "Checkpoint %s is prepared; diagnostic output failed. Check the checkpoint directory.",
                destination.name,
                exc_info=True,
            )
        return PreparedVersion(destination, manifest, owner, report)

    def release(self, prepared: PreparedVersion, *, remove_checkpoint: bool = False) -> None:
        with writer_lock(self.root / ".preparation", timeout=120):
            path = prepared.checkpoint.resolve()
            if path.parent != self.root.resolve():
                raise ValueError("Prepared checkpoint is outside this consumer's directory")
            lease = self.root / ".leases" / f"{valid_id(prepared.owner)}.json"
            if lease.exists():
                state = load_json(lease.read_bytes())
                if state["checkpoint"] != path.name or state["root"] != prepared.manifest["root"]:
                    raise ValueError("Prepared checkpoint lease belongs to another target")
            referenced = any(
                p != lease and load_json(p.read_bytes())["checkpoint"] == path.name
                for p in (self.root / ".leases").glob("*.json")
            )
            if remove_checkpoint and not referenced and path.exists():
                # The caller has finished all worker reads for this lease.
                shutil.rmtree(path)
            self.cache.unpin(prepared.owner)
            lease.unlink(missing_ok=True)
            sync_directory(lease.parent)
            # Reclaim after a reader releases its checkpoint. Preparing or
            # retrying a target leaves partial downloads available for reuse.
            # Other readers keep their pins, including separate consumer roots.
            try:
                self.cache.collect(retain_anchors=2)
            except (OSError, ValueError):
                logger.warning("Checkpoint released; consumer cache collection failed", exc_info=True)

    def reconcile(self, owners: list[str], retained: set[str]) -> None:
        """Discard interrupted preparations owned by one coordinator only."""
        with writer_lock(self.root / ".preparation", timeout=120):
            self._remove_staging()
            for owner in owners:
                if owner in retained:
                    continue
                lease = self.root / ".leases" / f"{valid_id(owner)}.json"
                if lease.exists():
                    record = load_json(lease.read_bytes())
                    checkpoint = record.get("checkpoint")
                    if checkpoint:
                        target = (self.root / checkpoint).resolve()
                        referenced = any(
                            p != lease and load_json(p.read_bytes()).get("checkpoint") == checkpoint
                            for p in (self.root / ".leases").glob("*.json")
                        )
                        if target.parent != self.root.resolve():
                            raise ValueError("Interrupted checkpoint lies outside its owner directory")
                        if not referenced and target.is_dir():
                            shutil.rmtree(target)
                self.cache.unpin(owner)
                lease.unlink(missing_ok=True)
                sync_directory(lease.parent)
