# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shared-filesystem content delivery with bounded parallel reads."""

import threading
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING

from relax.utils.logging_utils import get_logger

from .codec import MAX_CHUNK_BYTES
from .deadline import checkpoint, time_budget
from .parallel_io import ordered_map
from .store import (
    MAX_MANIFEST_BYTES,
    VersionStore,
    atomic_write,
    canonical,
    digest,
    durable_directory_batch,
    load_json,
    sync_directory,
    validate_manifest,
    writer_lock,
)


if TYPE_CHECKING:
    pass


logger = get_logger(__name__)
_fetch_target = ContextVar("weight_fetch_target", default=None)


class RemoteStoreError(ValueError):
    """The source returned a bounded operation error rather than a network
    loss."""

    def __init__(self, message: str, *, error_type: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type


class ContentError(ValueError):
    """Delivered content failed verification; request same-target repair."""


class SharedFSTransport:
    def __init__(self, source: VersionStore, *, repair=None) -> None:
        self.source = source
        self.repair = repair
        self.bytes_received = 0
        self._counter_lock = threading.Lock()

    def _received(self, count: int) -> None:
        with self._counter_lock:
            self.bytes_received += count

    def manifest(self, stream: str, version: int, *, full: bool = False) -> bytes:
        if full and self.repair is not None:
            value = self.repair.manifest(stream, version, full=True)
            self._received(len(value))
            return value
        value = canonical(self._full_manifest(stream, version) if full else self.source.manifest(stream, version))
        self._received(len(value))
        return value

    def _full_manifest(self, stream: str, version: int) -> dict:
        from .metrics import SyncMetrics
        from .traffic import capture, current

        parent = current()
        if parent is None:
            return self.source.ensure_full(stream, version)
        target = self.source.manifest(stream, version)
        metrics = SyncMetrics(target.get("update_id") or f"{stream}:{version}")
        failed = True
        try:
            with capture(metrics), metrics.stage("source_full_repair", path=self.source.root):
                manifest = self.source.ensure_full(stream, version)
            failed = False
            return manifest
        finally:
            parent.related(
                {
                    **metrics.report(),
                    "stream": stream,
                    "version": version,
                    "failed": failed,
                    "storage_role": "source",
                    "parent_operation_id": parent.operation_id,
                }
            )

    def pin(self, stream: str, version: int, owner: str) -> None:
        self.source.pin(stream, version, owner)

    def unpin(self, owner: str) -> None:
        self.source.unpin(owner)

    def versions(self, stream: str, *, after: int = -1, limit: int = 100, newest_first: bool = False) -> list[dict]:
        return self.source.versions(stream, after=after, limit=limit, newest_first=newest_first)

    def register(self, consumer: str, stream: str, start_version: int = 0, *, mode: str = "all_versions") -> dict:
        return self.source.register(consumer, stream, start_version, mode=mode)

    def acknowledge(self, consumer: str, receipt: dict) -> dict:
        return self.source.acknowledge(consumer, receipt)

    def consumer_status(self, consumer: str) -> dict:
        return self.source.consumer_status(consumer)

    def unregister(self, consumer: str) -> None:
        self.source.unregister(consumer)

    def object(self, key: str) -> bytes:
        path = self.source.object_path(key)
        if path.stat().st_size > MAX_CHUNK_BYTES:
            raise ValueError("Object exceeds frame limit")
        value = path.read_bytes()
        self._received(len(value))
        return value


def fetch(
    transport: SharedFSTransport,
    target: VersionStore,
    stream: str,
    version: int,
    *,
    reuse_objects: bool = True,
    full: bool = False,
    retention_owner: str | None = None,
) -> dict:
    """Fetch a version and its dependencies; validated durable chunks are
    resumed.

    An anchor can initialize an empty receiver. This method materializes bytes;
    installation and policy for serving skipped versions belong to the caller.
    """
    with writer_lock(target.root / ".fetch", timeout=120):
        reconcile_fetches(transport, target)
        owner = "fetch-" + uuid.uuid4().hex
        journal = target.root / "fetch-operations" / (owner + ".json")
        atomic_write(journal, canonical({"owner": owner, "stream": stream, "version": version, "phase": "ACQUIRING"}))
        return _fetch_operation(
            transport,
            target,
            stream,
            version,
            owner,
            journal,
            reuse_objects=reuse_objects,
            full=full,
            retention_owner=retention_owner,
        )


def reconcile_fetches(transport, target: VersionStore) -> None:
    """Replay releases while holding the cache's exclusive fetch lock."""
    for journal in (target.root / "fetch-operations").glob("*.json"):
        record = load_json(journal.read_bytes())
        transport.unpin(record["owner"])
        journal.unlink()
        sync_directory(journal.parent)


def _fetch_operation(transport, target, stream, version, owner, journal, **options):
    try:
        transport.pin(stream, version, owner)
        return _fetch_pinned(
            transport,
            target,
            stream,
            version,
            **options,
        )
    finally:
        try:
            # Cleanup has a small independent bound even after the operation
            # expires, so completed transfers can release source retention.
            with time_budget(5, inherit=False):
                transport.unpin(owner)
                journal.unlink()
                sync_directory(journal.parent)
        except (OSError, ConnectionError, RemoteStoreError) as error:
            logger.warning(
                "Retention release failed for %s: %s. Retry unpin on the source store after connectivity recovers.",
                owner,
                error,
            )


def _fetch_pinned(
    transport: SharedFSTransport,
    target: VersionStore,
    stream: str,
    version: int,
    *,
    reuse_objects: bool,
    full: bool,
    retention_owner: str | None,
) -> dict:
    with writer_lock(target.root / ".content-writer", timeout=120):
        target.refresh_budget()
        # Retain the independent FULL delivery once this target has used it.
        # A repeated default request must not recreate a dependency on old bases.
        if target.manifest_path(stream, version).exists():
            full = full or target.manifest(stream, version, prefer_full=True)["kind"] == "full"
        pending = []
        requested = version
        while True:
            checkpoint("dependency manifest fetch")
            raw = (
                transport.manifest(stream, requested, full=True)
                if full and requested == version
                else transport.manifest(stream, requested)
            )
            if len(raw) > MAX_MANIFEST_BYTES:
                raise ValueError("Manifest too large")
            try:
                m = validate_manifest(load_json(raw), target.max_weight_bytes)
            except ValueError as error:
                raise ContentError(str(error)) from error
            if m["stream"] != stream or m["version"] != requested:
                raise ValueError("Remote identity mismatch")
            if pending and pending[-1]["base_root"] != m["root"]:
                raise ContentError("Remote chain mismatch")
            pending.append(m)
            if m["kind"] == "full":
                break
            if reuse_objects and target.has_verified_version(stream, m["base_version"], m["base_root"]):
                break
            if len(pending) >= 4096:
                raise ValueError("Catch-up depth exceeded")
            requested = m["base_version"]
        for m in reversed(pending):
            objects = {}
            for tensor in m["tensors"]:
                for chunk in tensor["chunks"]:
                    key = chunk["object"]
                    if objects.setdefault(key, chunk["encoded_size"]) != chunk["encoded_size"]:
                        raise ContentError("One immutable object has conflicting sizes")

            def fetch_object(item):
                key, size = item
                checkpoint("object fetch")
                path = target.object_path(key)
                if (
                    reuse_objects
                    and path.exists()
                    and path.stat().st_size == size
                    and digest(path.read_bytes()) == key
                ):
                    return
                # Complete the deadline-aware capacity scan before receiving
                # bytes. A complete verified download must remain resumable
                # even when the enclosing operation expires during transfer.
                target._reserve(0)
                value = transport.object(key)
                if len(value) != size or digest(value) != key:
                    raise ContentError("Object content differs from manifest")
                target.put_object(key, value)

            # Overlap independent SharedFS metadata and content reads.
            # Workers drain before retention or writer locks are released.
            workers = 4
            token = _fetch_target.set({"stream": stream, "version": m["version"]})
            try:
                with durable_directory_batch(target.root / "objects"):
                    for _ in ordered_map(fetch_object, objects.items(), workers):
                        pass
            finally:
                _fetch_target.reset(token)
            target.commit(m)
        try:
            target._materialize(stream, version)
        except ValueError as error:
            raise ContentError(str(error)) from error
        if retention_owner:
            target.pin(stream, version, retention_owner)
        return pending[0]
