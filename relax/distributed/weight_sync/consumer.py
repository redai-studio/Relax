# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent, pinned consumption and streaming safetensors reconstruction."""

import os
import shutil
import struct
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Protocol

from .format import (
    MAX_MANIFEST_BYTES,
    WeightSyncError,
    canonical_json,
    digest,
    parse_json,
    require_digest,
    validate_catalog,
    validate_manifest,
)
from .store import BlockReader, CapacityError, SharedFSStore, atomic_write


class Transport(Protocol):
    def record(self, stream: str, version: int) -> dict: ...
    def manifest(self, artifact_id: str) -> dict: ...
    def object(self, object_id: str) -> bytes: ...
    def pin(self, reference: str, stream: str, version: int) -> None: ...
    def release(self, reference: str) -> None: ...
    def ensure_full(self, stream: str, version: int, compression: bool = True) -> str: ...


@dataclass(frozen=True)
class PreparedCheckpoint:
    stream: str
    version: int
    root: str
    artifact_id: str
    path: str
    reference: str


def verify_checkpoint(path: str | Path, catalog: dict) -> None:
    """Read actual safetensors bytes without tensor/device materialization."""
    path = Path(path)
    validate_catalog(catalog)
    index = parse_json((path / "model.safetensors.index.json").read_bytes())
    expected_names = {tensor["name"] for tensor in catalog["tensors"]}
    if set(index.get("weight_map", {})) != expected_names:
        raise WeightSyncError("Checkpoint parameter coverage mismatch")
    for tensor in catalog["tensors"]:
        filename = index["weight_map"][tensor["name"]]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise WeightSyncError("Invalid checkpoint shard name")
        with (path / filename).open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise WeightSyncError("Truncated safetensors header")
            header_size = struct.unpack("<Q", prefix)[0]
            header = parse_json(source.read(min(header_size, MAX_MANIFEST_BYTES + 1)))
            info = header[tensor["name"]]
            if (info["dtype"], info["shape"]) != (tensor["dtype"], tensor["shape"]):
                raise WeightSyncError("Checkpoint tensor metadata mismatch")
            start, end = info["data_offsets"]
            if end - start != sum(block["length"] for block in tensor["blocks"]):
                raise WeightSyncError("Checkpoint tensor length mismatch")
            source.seek(8 + header_size + start)
            for block in tensor["blocks"]:
                if digest(source.read(block["length"])) != block["digest"]:
                    raise WeightSyncError(f"Checkpoint bytes differ: {tensor['name']} offset={block['offset']}")


class Consumer:
    def __init__(
        self,
        transport: Transport,
        cache: SharedFSStore,
        checkpoint_root: str | Path,
        *,
        disk_budget_bytes: int = 128 * 1024**3,
        timeout_s: float = 600,
        max_chain: int = 100,
        shard_bytes: int = 1024**3,
    ) -> None:
        if timeout_s <= 0 or disk_budget_bytes <= 0 or max_chain <= 0 or shard_bytes <= 0:
            raise WeightSyncError("Consumer budgets and deadlines must be positive")
        self.transport = transport
        self.cache = cache
        if isinstance(transport, SharedFSStore) and transport.root == cache.root:
            raise WeightSyncError("Transport and node cache must use different store directories")
        self.checkpoint_root = Path(checkpoint_root).resolve()
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.disk_budget_bytes = disk_budget_bytes
        self.timeout_s = timeout_s
        self.max_chain = max_chain
        self.shard_bytes = shard_bytes
        self.last_metrics: dict = {}

    def prepare(self, stream: str, version: int, *, full: bool = False) -> PreparedCheckpoint:
        old_deadline = getattr(self.transport, "deadline", None)
        if hasattr(self.transport, "deadline"):
            self.transport.deadline = monotonic() + self.timeout_s
        try:
            return self._prepare(stream, version, full=full)
        finally:
            if hasattr(self.transport, "deadline"):
                self.transport.deadline = old_deadline

    def _prepare(self, stream: str, version: int, *, full: bool) -> PreparedCheckpoint:
        started = monotonic()
        deadline = started + self.timeout_s
        reference = f"prepare:{uuid.uuid4().hex}"
        self.last_metrics = {"received_bytes": 0, "reused_objects": 0, "reference": reference}
        self.transport.pin(reference, stream, version)
        # The reference intentionally survives failures, including process exit.
        # Callers can explicitly release it after recovery/abandonment.
        record = self.transport.record(stream, version)
        artifact_id = self.transport.ensure_full(stream, version) if full else record["artifact_id"]
        self.last_metrics["resolve_s"] = monotonic() - started
        fetch_started = monotonic()
        try:
            manifest = self._fetch_chain(artifact_id, deadline, reference)
        except WeightSyncError as error:
            if isinstance(error, CapacityError):
                raise
            if full:
                raise
            self.last_metrics["fallback_reason"] = str(error)[:1024]
            artifact_id = self.transport.ensure_full(stream, version)
            manifest = self._fetch_chain(artifact_id, deadline, reference)
        self.last_metrics["fetch_and_fallback_s"] = monotonic() - fetch_started
        if (manifest["stream"], manifest["version"], manifest["root"]) != (stream, version, record["root"]):
            raise WeightSyncError("Prepared artifact does not match the requested target")
        rebuild_started = monotonic()
        target = self.checkpoint_root / require_digest(record["root"])
        # A node-local lock prevents multiple workers from rebuilding one target.
        import fcntl

        with (self.checkpoint_root / ".reconstruct.lock").open("ab") as lock:
            while True:
                self._deadline(deadline)
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    from time import sleep

                    sleep(min(0.01, max(0, deadline - monotonic())))
            if target.exists():
                try:
                    verify_checkpoint(target, manifest["catalog"])
                except (WeightSyncError, OSError, KeyError, TypeError):
                    # Validated immutable objects can rebuild a corrupted local checkpoint.
                    shutil.rmtree(target)
            if not target.exists():
                self._reconstruct(manifest, target, deadline)
            verify_checkpoint(target, manifest["catalog"])
        self.last_metrics["rebuild_and_verify_s"] = monotonic() - rebuild_started
        self.last_metrics["prepare_s"] = monotonic() - started
        self._deadline(deadline)
        return PreparedCheckpoint(stream, version, record["root"], artifact_id, str(target), reference)

    def _deadline(self, deadline: float) -> None:
        if monotonic() >= deadline:
            raise TimeoutError("Weight preparation deadline exceeded; retained content may be reused on retry")

    def _fetch_chain(self, artifact_id: str, deadline: float, reference: str) -> dict:
        # Keep candidate objects protected from local GC until the entire dependency
        # closure and its persistent pin become visible in the same transaction.
        with self.cache.transaction(deadline) as state:
            return self._fetch_chain_locked(artifact_id, deadline, reference, state)

    def _fetch_chain_locked(self, artifact_id: str, deadline: float, reference: str, state: dict) -> dict:
        chain = []
        current_id = artifact_id
        while True:
            self._deadline(deadline)
            manifest = self.transport.manifest(current_id)
            validate_manifest(manifest, current_id)
            chain.append((current_id, manifest))
            for encoded in manifest["blocks"]:
                self._deadline(deadline)
                if encoded["object"] is None:
                    continue
                object_id = encoded["object"]
                try:
                    payload = self.cache.object(object_id)
                    self.last_metrics["reused_objects"] += 1
                except WeightSyncError:
                    payload = self.transport.object(object_id)
                    if len(payload) != encoded["length"] or digest(payload) != object_id:
                        raise WeightSyncError("Transport returned incorrect object")
                    self.cache._put("objects", payload)
                    self.last_metrics["received_bytes"] += len(payload)
            if manifest["kind"] == "FULL":
                break
            if len(chain) >= self.max_chain:
                raise WeightSyncError("Dependency budget exceeded")
            base_record = self.transport.record(manifest["stream"], manifest["base_version"])
            if base_record["root"] != manifest["base_root"]:
                raise WeightSyncError("Transport returned the wrong base version")
            current_id = base_record["artifact_id"]
        self._deadline(deadline)
        for current_id, manifest in reversed(chain):
            self.cache._put("artifacts", canonical_json(manifest))
            records = state["streams"].setdefault(manifest["stream"], {})
            existing = records.get(str(manifest["version"]))
            if existing is not None and existing["root"] != manifest["root"]:
                raise WeightSyncError("Cached version identity conflict")
            record = {
                "stream": manifest["stream"],
                "version": manifest["version"],
                "root": manifest["root"],
                "artifact_id": current_id,
                "artifacts": {manifest["kind"]: current_id},
            }
            records[str(manifest["version"])] = record
        target = chain[0][1]
        state["references"][reference] = {"stream": target["stream"], "version": target["version"]}
        self.cache._save(state)
        return chain[0][1]

    def release(self, prepared: PreparedCheckpoint) -> None:
        self.transport.release(prepared.reference)
        self.cache.release(prepared.reference)

    def _reconstruct(self, manifest: dict, destination: Path, deadline: float) -> None:
        catalog = manifest["catalog"]
        logical_bytes = sum(block["length"] for block in validate_manifest(manifest))
        used = sum(path.stat().st_size for path in self.checkpoint_root.rglob("*") if path.is_file())
        shards, current, size = [], [], 0
        for tensor in catalog["tensors"]:
            tensor_size = sum(block["length"] for block in tensor["blocks"])
            if current and size + tensor_size > self.shard_bytes:
                shards.append(current)
                current, size = [], 0
            current.append(tensor)
            size += tensor_size
        if current:
            shards.append(current)
        weight_map, plans = {}, []
        for shard_index, tensors in enumerate(shards, 1):
            filename = f"model-{shard_index:05d}-of-{len(shards):05d}.safetensors"
            header, offset = {}, 0
            for tensor in tensors:
                size = sum(block["length"] for block in tensor["blocks"])
                header[tensor["name"]] = {
                    "dtype": tensor["dtype"],
                    "shape": tensor["shape"],
                    "data_offsets": [offset, offset + size],
                }
                weight_map[tensor["name"]] = filename
                offset += size
            header_bytes = canonical_json(header)
            header_bytes += b" " * (-len(header_bytes) % 8)
            plans.append((filename, header_bytes, tensors))
        metadata = {
            "model.safetensors.index.json": canonical_json(
                {"metadata": {"total_size": logical_bytes}, "weight_map": weight_map}
            ),
            "weight-sync.json": canonical_json({"root": manifest["root"], "catalog": catalog}),
        }
        config = catalog["schema_context"].get("model_config")
        if config is not None:
            metadata["config.json"] = canonical_json(config)
        required = logical_bytes + sum(8 + len(header) for _, header, _ in plans) + sum(map(len, metadata.values()))
        if used + required > self.disk_budget_bytes:
            raise CapacityError(f"WAITING_CAPACITY: checkpoint_used={used}, target_bytes={required}")
        staging = Path(tempfile.mkdtemp(prefix=".reconstruct-", dir=self.checkpoint_root))
        try:
            reader = BlockReader(self.cache, manifest, self.max_chain + 1)
            block_index = 0
            for filename, header_bytes, tensors in plans:
                with (staging / filename).open("wb") as output:
                    output.write(struct.pack("<Q", len(header_bytes)))
                    output.write(header_bytes)
                    for tensor in tensors:
                        for _ in tensor["blocks"]:
                            self._deadline(deadline)
                            output.write(reader.read(block_index))
                            block_index += 1
                    output.flush()
                    os.fsync(output.fileno())
            for filename, data in metadata.items():
                atomic_write(staging / filename, data)
            verify_checkpoint(staging, catalog)
            os.replace(staging, destination)
            directory_fd = os.open(self.checkpoint_root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
