# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""POSIX content store with durable, atomic version/reference transactions.

One flock serializes publication, reference changes and GC. Readers pin a
version before opening its immutable content. A crash before index replacement
can leave unreachable objects; collect() safely removes them.
"""

import copy
import fcntl
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep
from typing import Iterator

from .format import (
    MAX_BLOCK_BYTES,
    MAX_MANIFEST_BYTES,
    WeightSyncError,
    canonical_json,
    decode_block,
    digest,
    integer,
    parse_json,
    require_digest,
    validate_manifest,
)


class CapacityError(WeightSyncError):
    """Publication cannot fit without releasing retained content."""


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class SharedFSStore:
    def __init__(
        self,
        root: str | Path,
        disk_budget_bytes: int | None = None,
        *,
        canonical_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.canonical_root = Path(canonical_root).resolve() if canonical_root is not None else self.root / "canonical"
        self.canonical_root.mkdir(parents=True, exist_ok=True)
        for name in ("blocks", "catalogs", "manifests"):
            (self.canonical_root / name).mkdir(exist_ok=True)
        self.deadline: float | None = None
        self._usage = threading.local()
        self.disk_budget_bytes = integer(disk_budget_bytes, 1) if disk_budget_bytes is not None else None
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("objects", "artifacts"):
            (self.root / name).mkdir(exist_ok=True)
        with self.transaction() as state:
            if not (self.root / "index.json").exists():
                self._save(state)

    def _load(self) -> dict:
        path = self.root / "index.json"
        if not path.exists():
            return {"format": 1, "streams": {}, "references": {}, "consumers": {}, "seals": {}}
        state = parse_json(path.read_bytes(), limit=64 * MAX_MANIFEST_BYTES)
        if state.get("format") != 1:
            raise WeightSyncError("Unsupported store index")
        return state

    def _save(self, state: dict, *, reclaim: bool = False) -> None:
        data = canonical_json(state)
        if not reclaim:
            self._check_capacity(len(data))
        path = self.root / "index.json"
        before = path.stat().st_size if path.exists() else 0
        atomic_write(path, data)
        self._account(len(data) - before)

    @contextmanager
    def transaction(self, deadline: float | None = None) -> Iterator[dict]:
        deadline = deadline if deadline is not None else self.deadline
        with (self.root / ".writer.lock").open("ab") as lock:
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if deadline is not None and monotonic() >= deadline:
                        raise TimeoutError("Store transaction deadline exceeded") from None
                    sleep(0.01)
            try:
                self._usage.used = None
                yield self._load()
            finally:
                self._usage.used = None
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def inventory(self) -> dict:
        return copy.deepcopy(self._load())

    def _record(self, state: dict, stream: str, version: int) -> dict:
        integer(version)
        try:
            return state["streams"][stream][str(version)]
        except KeyError as error:
            available = sorted(int(v) for v in state["streams"].get(stream, {}))
            raise WeightSyncError(f"VERSION_UNAVAILABLE: {stream}/{version}; available={available}") from error

    def record(self, stream: str, version: int) -> dict:
        return copy.deepcopy(self._record(self._load(), stream, version))

    def head(self, stream: str) -> dict | None:
        versions = self._load()["streams"].get(stream, {})
        return copy.deepcopy(versions[max(versions, key=int)]) if versions else None

    def manifest(self, artifact_id: str) -> dict:
        path = self.root / "artifacts" / require_digest(artifact_id)
        try:
            with path.open("rb") as source:
                data = source.read(MAX_MANIFEST_BYTES + 1)
        except FileNotFoundError as error:
            raise WeightSyncError(f"Artifact unavailable: {artifact_id}") from error
        manifest = parse_json(data)
        validate_manifest(manifest, artifact_id)
        return manifest

    def object(self, object_id: str) -> bytes:
        return self._read_content(self.root / "objects", object_id)

    def canonical_block(self, block_id: str) -> bytes:
        return self._read_content(self.canonical_root / "blocks", block_id)

    def canonical_catalog(self, root: str) -> dict:
        path = self.canonical_root / "catalogs" / require_digest(root)
        with path.open("rb") as source:
            data = source.read(MAX_MANIFEST_BYTES + 1)
        if digest(data) != root:
            raise WeightSyncError("Canonical catalog checksum mismatch")
        from .format import validate_catalog

        catalog = parse_json(data)
        validate_catalog(catalog)
        return catalog

    def _read_content(self, directory: Path, object_id: str) -> bytes:
        path = directory / require_digest(object_id)
        try:
            with path.open("rb") as source:
                payload = source.read(MAX_BLOCK_BYTES + 1)
        except FileNotFoundError as error:
            raise WeightSyncError(f"Object unavailable: {object_id}") from error
        if len(payload) > MAX_BLOCK_BYTES or digest(payload) != object_id:
            raise WeightSyncError(f"Object checksum/size mismatch: {object_id}")
        return payload

    def _put(self, directory: str, data: bytes) -> str:
        object_id = digest(data)
        canonical_directories = {"canonical": "blocks", "catalogs": "catalogs", "canonical_manifests": "manifests"}
        parent = (
            self.canonical_root / canonical_directories[directory]
            if directory in canonical_directories
            else self.root / directory
        )
        path = parent / object_id
        if path.exists():
            # A corrupted existing object must never be silently reused.
            if path.read_bytes() == data:
                return object_id
        self._check_capacity(len(data))
        before = path.stat().st_size if path.exists() else 0
        atomic_write(path, data)
        self._account(len(data) - before)
        return object_id

    def _account(self, delta: int) -> None:
        if getattr(self._usage, "used", None) is not None:
            self._usage.used += delta

    def _check_capacity(self, additional: int) -> None:
        if self.deadline is not None and monotonic() >= self.deadline:
            raise TimeoutError("Store operation deadline exceeded")
        if self.disk_budget_bytes is None:
            return
        used = getattr(self._usage, "used", None)
        if used is None:
            used = self._usage.used = self.disk_usage()
        if used + additional > self.disk_budget_bytes:
            raise CapacityError(
                f"WAITING_CAPACITY: used={used}, required={additional}, budget={self.disk_budget_bytes}; "
                "release references/consumers or increase the budget and retry"
            )

    def disk_usage(self) -> int:
        size = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
        if not self.canonical_root.is_relative_to(self.root):
            size += sum(path.stat().st_size for path in self.canonical_root.rglob("*") if path.is_file())
        return size

    def _artifact_for(self, state: dict, stream: str, version: int, kind: str | None = None) -> dict:
        record = self._record(state, stream, version)
        artifact_id = record["artifact_id"] if kind is None else record["artifacts"].get(kind)
        if artifact_id is None:
            raise WeightSyncError(f"{kind} artifact unavailable for {stream}/{version}")
        manifest = self.manifest(artifact_id)
        if (manifest["stream"], manifest["version"], manifest["root"]) != (stream, version, record["root"]):
            raise WeightSyncError("Version/artifact binding mismatch")
        return manifest

    def resolve(self, stream: str, version: int, kind: str | None = None) -> dict:
        return self._artifact_for(self._load(), stream, version, kind)

    def read_block(self, manifest: dict, index: int, max_chain: int = 1024) -> bytes:
        return BlockReader(self, manifest, max_chain).read(index)

    def pin(self, reference: str, stream: str, version: int, *, deadline: float | None = None) -> None:
        if not reference:
            raise WeightSyncError("Empty reference identity")
        with self.transaction(deadline) as state:
            self._record(state, stream, version)
            target = {"stream": stream, "version": version}
            existing = state["references"].get(reference)
            if existing is not None and existing != target:
                raise WeightSyncError("Reference already binds a different target")
            state["references"][reference] = target
            self._save(state)

    def release(self, reference: str, *, deadline: float | None = None) -> None:
        with self.transaction(deadline) as state:
            state["references"].pop(reference, None)
            self._save(state, reclaim=True)

    def register_consumer(self, consumer_id: str, stream: str, start_version: int = 0) -> dict:
        integer(start_version)
        with self.transaction() as state:
            head = max((int(v) for v in state["streams"].get(stream, {})), default=-1)
            for version in range(start_version, head + 1):
                self._record(state, stream, version)
            existing = state["consumers"].get(consumer_id)
            if existing is not None:
                if existing["stream"] != stream or existing["start_version"] != start_version:
                    raise WeightSyncError("Consumer identity already registered with another range")
                return copy.deepcopy(existing)
            cursor = {"stream": stream, "start_version": start_version, "last_active_version": start_version - 1}
            state["consumers"][consumer_id] = cursor
            self._save(state)
            return cursor

    def acknowledge(self, consumer_id: str, version: int, root: str, installation_id: str) -> dict:
        with self.transaction() as state:
            cursor = state["consumers"][consumer_id]
            record = self._record(state, cursor["stream"], version)
            if record["root"] != require_digest(root) or not installation_id:
                raise WeightSyncError("Invalid ACTIVE receipt")
            receipt = {"version": version, "root": root, "installation_id": installation_id}
            if cursor.get("receipt") == receipt:
                return copy.deepcopy(cursor)
            if version != cursor["last_active_version"] + 1:
                raise WeightSyncError("Only the next consecutive ACTIVE version may advance the cursor")
            cursor["last_active_version"] = version
            cursor["receipt"] = receipt
            state["references"][f"serving:{consumer_id}"] = {"stream": cursor["stream"], "version": version}
            self._save(state)
            return copy.deepcopy(cursor)

    def unregister_consumer(self, consumer_id: str) -> None:
        with self.transaction() as state:
            state["consumers"].pop(consumer_id, None)
            self._save(state, reclaim=True)

    def ensure_full(
        self, stream: str, version: int, compression: bool = True, *, deadline: float | None = None
    ) -> str:
        from .format import encode_block

        if type(compression) is not bool:
            raise WeightSyncError("compression must be a boolean")
        with self.transaction(deadline) as state:
            record = self._record(state, stream, version)
            existing_id = record["artifacts"].get("FULL")
            if existing_id is not None:
                try:
                    existing = self.manifest(existing_id)
                    if (existing["stream"], existing["version"], existing["root"], existing["kind"]) != (
                        stream,
                        version,
                        record["root"],
                        "FULL",
                    ):
                        raise WeightSyncError("FULL identity mismatch")
                    reader = BlockReader(self, existing, 1)
                    for index in range(len(existing["blocks"])):
                        self._check_deadline(deadline)
                        reader.read(index)
                    return existing_id
                except (OSError, WeightSyncError):
                    pass
            try:
                catalog = self.canonical_catalog(record["root"])
            except FileNotFoundError as error:
                raise WeightSyncError(
                    "Same-target FULL requires the store owner's retained canonical material"
                ) from error
            if existing_id is not None:
                # Use the owner-local manifest to restore damaged delivery metadata too.
                data = (self.canonical_root / "manifests" / existing_id).read_bytes()
                full = parse_json(data)
                validate_manifest(full, existing_id)
                for block, encoded in zip(validate_manifest(full), full["blocks"], strict=True):
                    self._check_deadline(deadline)
                    target = self.canonical_block(block["digest"])
                    candidate, payload = encode_block(target, None, block["width"], encoded["encoding"] == "RAW_ZLIB")
                    if candidate != encoded:
                        raise WeightSyncError("Cannot reproduce the immutable FULL encoding with this codec runtime")
                    self._put("objects", payload)
                self._put("artifacts", data)
                return existing_id
            from .format import validate_catalog

            descriptors = []
            for block in validate_catalog(catalog):
                self._check_deadline(deadline)
                # Recovery must not depend on the damaged delivery/base chain.
                target = self.canonical_block(block["digest"])
                encoded, payload = encode_block(target, None, block["width"], compression)
                self._put("objects", payload)
                descriptors.append(encoded)
            full = {
                "format": 1,
                "stream": stream,
                "version": version,
                "root": record["root"],
                "catalog": catalog,
                "kind": "FULL",
                "base_version": None,
                "base_root": None,
                "blocks": descriptors,
            }
            validate_manifest(full)
            data = canonical_json(full)
            self._put("canonical_manifests", data)
            artifact_id = self._put("artifacts", data)
            record["artifacts"]["FULL"] = artifact_id
            self._save(state)
            return artifact_id

    @staticmethod
    def _check_deadline(deadline: float | None) -> None:
        if deadline is not None and monotonic() >= deadline:
            raise TimeoutError("FULL materialization deadline exceeded")

    def _gc_manifest(self, artifact_id: str) -> dict:
        try:
            return self.manifest(artifact_id)
        except WeightSyncError:
            # GC runs under the writer lock. Restore immutable delivery metadata
            # from the independent owner copy before following its references.
            path = self.canonical_root / "manifests" / require_digest(artifact_id)
            with path.open("rb") as source:
                data = source.read(MAX_MANIFEST_BYTES + 1)
            manifest = parse_json(data)
            validate_manifest(manifest, artifact_id)
            try:
                self._put("artifacts", data)
            except CapacityError:
                # Reclamation can traverse verified owner metadata even when
                # the temporary delivery repair cannot fit before GC completes.
                pass
            return manifest

    def seal(self, seal_id: str, stream: str, versions: list[int]) -> None:
        # Materialize FULL variants first. A persistent reference is acquired for each
        # target before reconstruction; an interrupted seal remains conservatively pinned.
        for version in versions:
            self.pin(f"seal:{seal_id}:{version}", stream, version)
            self.ensure_full(stream, version)
        with self.transaction() as state:
            for version in versions:
                manifest = self._artifact_for(state, stream, version, "FULL")
                for index in range(len(manifest["blocks"])):
                    self.read_block(manifest, index)
            value = {"stream": stream, "versions": sorted(set(versions))}
            if seal_id in state["seals"] and state["seals"][seal_id] != value:
                raise WeightSyncError("Seal identity already binds another target set")
            state["seals"][seal_id] = value
            self._save(state)

    def collect(self) -> dict[str, int]:
        with self.transaction() as state:
            retained = {(ref["stream"], ref["version"]) for ref in state["references"].values()}
            for stream, versions in state["streams"].items():
                ordered = sorted(int(v) for v in versions)
                anchors = [v for v in ordered if self._gc_manifest(versions[str(v)]["artifact_id"])["kind"] == "FULL"]
                cutoff = anchors[-2] if len(anchors) >= 2 else (ordered[0] if ordered else 0)
                retained.update((stream, v) for v in ordered if v >= cutoff)
            for cursor in state["consumers"].values():
                retained.update(
                    (cursor["stream"], int(v))
                    for v in state["streams"].get(cursor["stream"], {})
                    if int(v) > cursor["last_active_version"]
                )
            artifacts, objects, canonical, catalogs = set(), set(), set(), set()
            queue = list(retained)
            visited = set()
            while queue:
                stream, version = queue.pop()
                if (stream, version) in visited:
                    continue
                visited.add((stream, version))
                record = self._record(state, stream, version)
                catalogs.add(record["root"])
                for artifact_id in record["artifacts"].values():
                    artifacts.add(artifact_id)
                    manifest = self._gc_manifest(artifact_id)
                    canonical.update(block["digest"] for block in validate_manifest(manifest))
                    objects.update(b["object"] for b in manifest["blocks"] if b["object"] is not None)
                    if manifest["kind"] == "DELTA":
                        queue.append((stream, manifest["base_version"]))
            for stream, versions in state["streams"].items():
                state["streams"][stream] = {v: r for v, r in versions.items() if (stream, int(v)) in visited}
            # Delete visibility first; a crash leaves only unreachable content behind.
            self._save(state, reclaim=True)
            removed = 0
            for directory, live in (
                (self.root / "artifacts", artifacts),
                (self.root / "objects", objects),
                (self.canonical_root / "blocks", canonical),
                (self.canonical_root / "catalogs", catalogs),
                (self.canonical_root / "manifests", artifacts),
            ):
                for path in directory.iterdir():
                    if path.name not in live:
                        removed += path.stat().st_size
                        path.unlink()
            return {"removed_bytes": removed, "retained_versions": len(visited), "retained_objects": len(objects)}


class BlockReader:
    """Reuse validated metadata across blocks instead of rehashing the model
    per block."""

    def __init__(self, store: SharedFSStore, manifest: dict, max_chain: int) -> None:
        self.store = store
        self.target = manifest
        self.max_chain = max_chain
        self.state = store._load()
        self._metadata: dict[int, tuple[dict, list[dict]]] = {}
        self._schema: bytes | None = None
        self._register(manifest)

    def _register(self, manifest: dict) -> None:
        blocks = validate_manifest(manifest)
        catalog = manifest["catalog"]
        schema = canonical_json(
            {
                "context": catalog["schema_context"],
                "chunk_bytes": catalog["chunk_bytes"],
                "tensors": [(t["name"], t["shape"], t["dtype"]) for t in catalog["tensors"]],
            }
        )
        if self._schema is not None and schema != self._schema:
            raise WeightSyncError("Delta/base schema mismatch")
        self._schema = schema
        self._metadata[manifest["version"]] = manifest, blocks

    def read(self, index: int) -> bytes:
        current = self.target
        chain = []
        while True:
            blocks = self._metadata[current["version"]][1]
            if not 0 <= index < len(blocks):
                raise WeightSyncError("Base block coordinate mismatch")
            block, encoded = blocks[index], current["blocks"][index]
            chain.append((block, encoded))
            if encoded["encoding"] in {"RAW", "RAW_ZLIB"}:
                break
            if len(chain) >= self.max_chain:
                raise WeightSyncError("Dependency chain exceeds reconstruction budget; request same-target FULL")
            if current["base_version"] not in self._metadata:
                base = self.store._artifact_for(self.state, current["stream"], current["base_version"])
                self._register(base)
            base, base_blocks = self._metadata[current["base_version"]]
            if base["root"] != current["base_root"]:
                raise WeightSyncError("Base identity mismatch")
            if index >= len(base_blocks) or any(
                base_blocks[index][key] != block[key] for key in ("name", "offset", "length", "width")
            ):
                raise WeightSyncError("Base block coordinate mismatch")
            current = base
        output = None
        for block, encoded in reversed(chain):
            payload = b"" if encoded["encoding"] == "COPY" else self.store.object(encoded["object"])
            output = decode_block(encoded, payload, output, block["length"], block["width"], block["digest"])
        return output
