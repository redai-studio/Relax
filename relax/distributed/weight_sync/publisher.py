# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Immutable CPU capture and a single pending, durable publication."""

import copy
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic
from typing import Any, Iterable

from .format import (
    DTYPE_SIZES,
    MAX_BLOCK_BYTES,
    WeightSyncError,
    canonical_json,
    digest,
    encode_block,
    integer,
    validate_manifest,
)
from .store import CapacityError, SharedFSStore


TORCH_DTYPES = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.int32": "I32",
    "torch.int64": "I64",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
}


@dataclass(frozen=True)
class PublishedVersion:
    stream: str
    version: int
    root: str
    artifact_id: str
    update_id: str


@dataclass(frozen=True)
class PublicationHandle:
    update_id: str
    candidate_id: str


class Publisher:
    """Capture on the caller thread; encode/store on one CPU worker.

    No worker ever accesses live model storage or a distributed process group.
    Retrying the last update_id after an ambiguous commit returns its original
    immutable version. Only finalize_pending advances the caller's confirmed
    base.
    """

    def __init__(
        self,
        store: SharedFSStore,
        stream: str,
        schema_context: dict,
        *,
        chunk_bytes: int = MAX_BLOCK_BYTES,
        compression: bool = True,
        anchor_interval: int = 20,
        host_budget_bytes: int = 16 * 1024**3,
        full_only: bool = False,
    ) -> None:
        if not isinstance(stream, str) or not stream:
            raise WeightSyncError("A nonempty stream is required")
        if type(compression) is not bool or type(full_only) is not bool:
            raise WeightSyncError("compression and full_only must be booleans")
        self.store = store
        self.stream = stream
        self.schema_context = copy.deepcopy(schema_context)
        canonical_json(self.schema_context)
        self.chunk_bytes = integer(chunk_bytes, 8, MAX_BLOCK_BYTES)
        self.anchor_interval = integer(anchor_interval, 1)
        self.host_budget_bytes = integer(host_budget_bytes, 1)
        self.compression = compression
        self.full_only = full_only
        self.confirmed: PublishedVersion | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight-publication")
        self._pending: tuple[PublicationHandle, Future] | None = None
        self.last_metrics: dict[str, Any] = {}

    def submit(self, update_id: str, named_tensors: Iterable[tuple[str, Any]]) -> PublicationHandle:
        if self._pending is not None:
            raise CapacityError("WAITING_QUEUE: finalize the pending publication before capturing another snapshot")
        if not isinstance(update_id, str) or not update_id:
            raise WeightSyncError("A stable nonempty update_id is required")
        snapshot = {}
        size = 0
        started = monotonic()
        for name, tensor in named_tensors:
            if not isinstance(name, str) or not name or name in snapshot:
                raise WeightSyncError(f"Empty or duplicate tensor name: {name}")
            if str(tensor.dtype) not in TORCH_DTYPES or tensor.layout.__str__() != "torch.strided":
                raise WeightSyncError(f"Unsupported tensor dtype/layout: {name}")
            size += tensor.numel() * tensor.element_size()
            # Include codec scratch (byte compare, index candidates and compression).
            if size + 16 * self.chunk_bytes > self.host_budget_bytes:
                raise CapacityError(f"WAITING_CAPACITY: snapshot={size}, codec_reserve={16 * self.chunk_bytes}")
            snapshot[name] = tensor.detach().to(device="cpu", copy=True).contiguous()
        if not snapshot:
            raise WeightSyncError("Cannot publish an empty model")
        handle = PublicationHandle(update_id, uuid.uuid4().hex)
        self.last_metrics = {"snapshot_bytes": size, "capture_s": monotonic() - started}
        self._pending = handle, self._executor.submit(self._publish, update_id, snapshot)
        return handle

    def finalize_pending(self, handle: PublicationHandle, timeout: float | None = None) -> PublishedVersion:
        if self._pending is None or self._pending[0] != handle:
            raise WeightSyncError("Unknown publication handle")
        future = self._pending[1]
        try:
            published = future.result(timeout=timeout)
        except TimeoutError:
            # The worker can still commit. Keep its slot and identity for a later finalize.
            if future.done():
                self._pending = None
            raise
        except Exception:
            self._pending = None
            raise
        self.confirmed = published
        self._pending = None
        return published

    def close(self) -> None:
        try:
            if self._pending is not None:
                self.finalize_pending(self._pending[0])
        finally:
            self._executor.shutdown(wait=True)

    def _publish(self, update_id: str, snapshot: dict) -> PublishedVersion:
        import torch

        started = monotonic()
        with self.store.transaction() as state:
            versions = state["streams"].setdefault(self.stream, {})
            latest = versions[max(versions, key=int)] if versions else None
            if latest is not None and latest["update_id"] == update_id:
                catalog = self.store._artifact_for(state, self.stream, latest["version"])["catalog"]
                captured = []
                for name in sorted(snapshot, key=lambda value: value.encode()):
                    tensor = snapshot[name]
                    dtype = TORCH_DTYPES[str(tensor.dtype)]
                    view = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
                    stride = self.chunk_bytes // DTYPE_SIZES[dtype] * DTYPE_SIZES[dtype]
                    blocks = [
                        {
                            "offset": offset,
                            "length": len(view[offset : offset + stride]),
                            "digest": digest(view[offset : offset + stride]),
                        }
                        for offset in range(0, len(view), stride)
                    ]
                    captured.append({"name": name, "dtype": dtype, "shape": list(tensor.shape), "blocks": blocks})
                if catalog != {
                    "format": 1,
                    "byte_order": "little",
                    "schema_context": self.schema_context,
                    "chunk_bytes": self.chunk_bytes,
                    "tensors": captured,
                }:
                    raise WeightSyncError("Retry of update_id contains different target weights or schema")
                return PublishedVersion(**{key: latest[key] for key in PublishedVersion.__dataclass_fields__})
            if any(record["update_id"] == update_id for record in versions.values()):
                raise WeightSyncError("update_id already belongs to an older version")
            version = latest["version"] + 1 if latest is not None else 0
            try:
                base = self.store._artifact_for(state, self.stream, latest["version"]) if latest else None
            except WeightSyncError:
                base = None
            schema = [(name, TORCH_DTYPES[str(t.dtype)], list(t.shape)) for name, t in sorted(snapshot.items())]
            same_schema = base is not None and (
                base["catalog"]["schema_context"] == self.schema_context
                and base["catalog"]["chunk_bytes"] == self.chunk_bytes
                and [(t["name"], t["dtype"], t["shape"]) for t in base["catalog"]["tensors"]] == schema
            )
            full = self.full_only or version % self.anchor_interval == 0 or not same_schema
            base_blocks = validate_manifest(base) if base is not None else []
            if not full:
                try:
                    for block in base_blocks:
                        self.store.canonical_block(block["digest"])
                except WeightSyncError:
                    # The current immutable snapshot can always reseed the stream.
                    full = True
            tensors, descriptors = [], []
            metrics = {"logical_bytes": 0, "payload_bytes": 0, "new_object_bytes": 0, "encodings": {}}
            block_index = 0
            for name in sorted(snapshot, key=lambda value: value.encode()):
                tensor = snapshot[name]
                dtype = TORCH_DTYPES[str(tensor.dtype)]
                width = DTYPE_SIZES[dtype]
                view = memoryview(tensor.reshape(-1).view(dtype=torch.uint8).numpy()).cast("B")
                stride = self.chunk_bytes // width * width
                blocks = []
                for offset in range(0, len(view), stride):
                    target = bytes(view[offset : offset + stride])
                    before = None if full else self.store.canonical_block(base_blocks[block_index]["digest"])
                    encoded, payload = encode_block(target, before, width, self.compression)
                    self.store._put("canonical", target)
                    blocks.append({"offset": offset, "length": len(target), "digest": digest(target)})
                    descriptors.append(encoded)
                    metrics["logical_bytes"] += len(target)
                    metrics["payload_bytes"] += len(payload)
                    metrics["encodings"][encoded["encoding"]] = metrics["encodings"].get(encoded["encoding"], 0) + 1
                    if payload:
                        exists = (self.store.root / "objects" / encoded["object"]).exists()
                        self.store._put("objects", payload)
                        if not exists:
                            metrics["new_object_bytes"] += len(payload)
                    block_index += 1
                tensors.append({"name": name, "dtype": dtype, "shape": list(tensor.shape), "blocks": blocks})
            catalog = {
                "format": 1,
                "byte_order": "little",
                "schema_context": self.schema_context,
                "chunk_bytes": self.chunk_bytes,
                "tensors": tensors,
            }
            manifest = {
                "format": 1,
                "stream": self.stream,
                "version": version,
                "root": digest(canonical_json(catalog)),
                "kind": "FULL" if full else "DELTA",
                "base_version": None if full else base["version"],
                "base_root": None if full else base["root"],
                "catalog": catalog,
                "blocks": descriptors,
            }
            validate_manifest(manifest)
            self.store._put("catalogs", canonical_json(catalog))
            artifact_bytes = canonical_json(manifest)
            self.store._put("canonical_manifests", artifact_bytes)
            artifact_id = self.store._put("artifacts", artifact_bytes)
            published = PublishedVersion(self.stream, version, manifest["root"], artifact_id, update_id)
            record = dict(published.__dict__) | {"artifacts": {manifest["kind"]: artifact_id}}
            versions[str(version)] = record
            metrics["manifest_bytes"] = len(artifact_bytes)
            record["metrics"] = metrics
            self.store._save(state)
            self.last_metrics.update(metrics | {"publish_s": monotonic() - started})
            return published
