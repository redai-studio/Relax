# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Single-writer immutable store. Publication is the final manifest rename.

Requires a filesystem providing atomic rename and, for power-loss durability,
POSIX directory fsync. Windows is supported for functional tests only.
"""

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
import zlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from .codec import MAX_CHUNK_BYTES, decode, encode
from .deadline import checkpoint as check_deadline
from .deadline import operation_remaining
from .parallel_io import ordered_map


DTYPES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
        "utf-8"
    )


def load_json(data: bytes | str) -> object:
    def unique(pairs: list) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON number: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid_constant)


def logical_root(m: dict) -> str:
    directory = {
        "schema_context": m["schema_context"],
        "tensors": [
            {
                "name": t["name"],
                "dtype": t["dtype"],
                "shape": t["shape"],
                "chunks": [{k: c[k] for k in ("offset", "size", "output")} for c in t["chunks"]],
            }
            for t in m["tensors"]
        ],
    }
    return digest(canonical(directory))


def identify(m: dict) -> dict:
    m["root"] = logical_root(m)
    m["artifact_id"] = digest(canonical({k: v for k, v in m.items() if k != "artifact_id"}))
    return m


def valid_id(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value) is None:
        raise ValueError("Invalid stream identity")
    return value


def valid_hash(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Invalid content identity")
    return value


def sync_directory(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def stored_bytes(directory: Path) -> int:
    """Count each immutable hard-linked file once for disk admission.

    Reuse DirEntry metadata during traversal: large immutable stores otherwise
    pay for two path-based stat calls for every retained object. File symlinks
    count their target inode; directory symlinks are not traversed.
    """
    if not directory.is_dir():
        return 0

    def batches():
        pending, batch = [str(directory)], []
        while pending:
            check_deadline("storage capacity scan")
            try:
                entries = os.scandir(pending.pop())
            except FileNotFoundError:
                continue
            with entries as scanned:
                for entry in scanned:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(entry.path)
                        else:
                            batch.append(entry)
                            if len(batch) == 64:
                                yield batch
                                batch = []
                    except FileNotFoundError:
                        continue
        if batch:
            yield batch

    def metadata(batch):
        records = []
        for entry in batch:
            check_deadline("storage capacity metadata")
            try:
                if entry.is_file():
                    status = entry.stat()
                    key = (status.st_dev, status.st_ino) if status.st_ino else entry.path
                    records.append((key, status.st_size))
            except FileNotFoundError:
                # Atomic writers can rename temporary entries during admission.
                continue
        return records

    seen, total = set(), 0
    chunks = iter(batches())
    # Avoid a pool for the small stores used by control setup and unit tests.
    first = next(chunks, None)
    if first is None:
        return 0
    for records in ([metadata(first)], ordered_map(metadata, chunks, 8)):
        for batch in records:
            for key, size in batch:
                if key not in seen:
                    total += size
                    seen.add(key)
    return total


def ensure_directory(path: Path) -> None:
    if path.is_dir():
        return
    ensure_directory(path.parent)
    path.mkdir(exist_ok=True)
    sync_directory(path.parent)


_deferred_directory_sync = ContextVar("weight_store_deferred_directory_sync", default=None)
_held_content_locks = ContextVar("weight_store_content_locks", default=frozenset())


class _ControlCapacityRequired(Exception):
    """Retry admission after acquiring the content lock in its normal order."""


@contextmanager
def durable_directory_batch(directory: Path) -> Iterator[None]:
    """Persist immutable names together before their manifest can be committed.

    Every file is still fsynced before rename. No manifest may be written in
    this scope. The directory is synced even when a retry reuses all files left
    by an interrupted batch. Context propagation keeps concurrent control
    operations outside this scope on their normal per-write durability path.
    """
    ensure_directory(directory)
    token = _deferred_directory_sync.set(directory)
    try:
        yield
        sync_directory(directory)
    finally:
        _deferred_directory_sync.reset(token)


def atomic_write(path: Path, data: bytes) -> None:
    from .traffic import file_write

    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            written = out.write(data)
            file_write(path, written)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        if _deferred_directory_sync.get() != path.parent:
            sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def writer_lock(path: Path, *, timeout: float = 0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        deadline = time.monotonic() + (operation_remaining(timeout, "store lock") if timeout else 0)
        while True:
            try:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    import msvcrt

                    handle.seek(0)
                    if handle.read(1) == b"":
                        handle.write(b"0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as error:
                if error.errno not in (11, 13, 35, 36):
                    raise
                if not timeout:
                    raise
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("Store is busy; retry the operation within a new budget") from error
                time.sleep(min(0.025, left))
        token = None
        if path.name == ".content-writer":
            identity = (os.getpid(), str(path.resolve()))
            token = _held_content_locks.set(_held_content_locks.get() | {identity})
        try:
            yield
        finally:
            if token is not None:
                _held_content_locks.reset(token)
            if os.name == "posix":
                fcntl.flock(handle, fcntl.LOCK_UN)
            else:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def validate_manifest(m: dict, max_weight_bytes: int) -> dict:
    if type(m.get("format")) is not int or m["format"] != 1 or m.get("byteorder") != "little":
        raise ValueError("Unsupported manifest format")
    valid_id(m["stream"])
    version = m["version"]
    if type(version) is not int or version < 0:
        raise ValueError("Invalid version")
    if type(m.get("compression", False)) is not bool:
        raise ValueError("Invalid compression policy")
    if m["kind"] not in ("full", "delta"):
        raise ValueError("Invalid version kind")
    if m["kind"] == "full":
        if m["base_version"] is not None or m["base_root"] is not None:
            raise ValueError("Full version has a base")
    elif version == 0 or type(m["base_version"]) is not int or m["base_version"] != version - 1:
        raise ValueError("Delta must reference preceding version")
    else:
        valid_hash(m["base_root"])
    if logical_root(m) != valid_hash(m["root"]):
        raise ValueError("Logical root mismatch")
    if digest(canonical({k: v for k, v in m.items() if k != "artifact_id"})) != valid_hash(m["artifact_id"]):
        raise ValueError("Artifact identity mismatch")
    context = m.get("schema_context")
    if context is not None:
        if context["model_config_sha256"] is not None:
            valid_hash(context["model_config_sha256"])
            if digest(canonical(context.get("model_config"))) != context["model_config_sha256"]:
                raise ValueError("Model configuration digest mismatch")
        if not isinstance(context["converter_revision"], str) or not 0 < len(context["converter_revision"]) <= 256:
            raise ValueError("Invalid converter identity")
    if not isinstance(m["tensors"], list) or len(m["tensors"]) > 100000:
        raise ValueError("Invalid tensor table")
    names, total = [], 0
    for tensor in m["tensors"]:
        name, dtype, shape = tensor["name"], tensor["dtype"], tensor["shape"]
        if not isinstance(name, str) or not name or len(name) > 1024 or name == "__metadata__":
            raise ValueError("Invalid tensor name")
        if dtype not in DTYPES or not isinstance(shape, list) or len(shape) > 16:
            raise ValueError("Invalid tensor schema")
        if any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("Invalid shape")
        size = math.prod(shape) * DTYPES[dtype]
        total += size
        if total > max_weight_bytes:
            raise ValueError("Weight budget exceeded")
        position = 0
        for chunk in tensor["chunks"]:
            n = chunk["size"]
            if type(n) is not int or not 0 < n <= MAX_CHUNK_BYTES or n % DTYPES[dtype]:
                raise ValueError("Invalid chunk size")
            codecs = ("copy", "raw", "sparse", "raw_zlib", "xor_zlib")
            if type(chunk["offset"]) is not int or chunk["offset"] != position or chunk["kind"] not in codecs:
                raise ValueError("Invalid chunk coverage")
            if m["kind"] == "full" and chunk["kind"] not in ("raw", "raw_zlib"):
                raise ValueError("Full version contains delta chunk")
            valid_hash(chunk["output"])
            valid_hash(chunk["object"])
            encoded_size = chunk["encoded_size"]
            if type(encoded_size) is not int or not 0 <= encoded_size <= MAX_CHUNK_BYTES:
                raise ValueError("Invalid encoded size")
            position += n
        if position != size:
            raise ValueError("Incomplete tensor coverage")
        names.append(name)
    if names != sorted(set(names), key=lambda name: name.encode("utf-8")):
        raise ValueError("Tensor names must be unique and sorted")
    return m


class VersionStore:
    def __init__(
        self,
        root: str | Path,
        *,
        disk_budget: int,
        max_weight_bytes: int,
        canonical_root: str | Path | None = None,
        canonical_budget: int | None = None,
        canonical_blocks: bool | None = None,
    ) -> None:
        self.root = Path(root)
        self.disk_budget = disk_budget
        self.max_weight_bytes = max_weight_bytes
        self._budget_lock = threading.RLock()
        self._content_locks = tuple(threading.Lock() for _ in range(64))
        self._remaining_bytes: int | None = None
        self._remaining_canonical_bytes: int | None = None
        if disk_budget <= 0 or max_weight_bytes <= 0:
            raise ValueError("Explicit positive budgets are required")
        ensure_directory(self.root.resolve())

        # Save placement separately from wire manifests. Consumers receive no
        # publisher filesystem paths, and a restarted owner uses the same data.
        layout_path = self.root / "canonical-location.json"
        layout = load_json(layout_path.read_bytes()) if layout_path.exists() else None
        requested = str(Path(canonical_root).resolve()) if canonical_root is not None else None
        if layout is not None:
            if requested is not None and requested != layout["root"]:
                raise ValueError("canonical_root differs from the retained store; use its recorded placement")
            if canonical_budget is not None and canonical_budget <= 0:
                raise ValueError("canonical_budget must be positive")
            canonical_budget = canonical_budget if canonical_budget is not None else layout["budget"]
        elif requested is not None:
            placement, package_root = Path(requested), self.root.resolve()
            if placement == package_root or placement in package_root.parents or package_root in placement.parents:
                raise ValueError("Independent canonical_root must be outside the package store")
            if type(canonical_budget) is not int or canonical_budget <= 0:
                raise ValueError("Independent canonical_root requires a positive canonical_budget")
            if any((self.root / "versions").glob("*/*.json")):
                raise ValueError("Set canonical_root before the first publication; existing content stays in place")
            # A configured volume may hold several stores. Each retains its
            # own namespace, which is persisted and survives owner restarts.
            layout = {"root": requested, "namespace": digest(str(package_root).encode()), "budget": canonical_budget}
            if len(canonical(layout)) > min(disk_budget, canonical_budget):
                raise OSError("Store and canonical budgets must include placement metadata")
            directory = placement / layout["namespace"]
            ensure_directory(directory)
            atomic_write(directory / "placement.json", canonical(layout))
            atomic_write(layout_path, canonical(layout))
        elif canonical_budget is not None:
            raise ValueError("canonical_budget requires canonical_root")
        self.canonical_directory = (
            Path(layout["root"]) / valid_hash(layout["namespace"]) if layout else self.root / "materialized"
        )
        self.canonical_budget = canonical_budget if layout else None
        self._canonical_layout = layout

        format_path = self.root / "canonical-format.json"
        retained_format = load_json(format_path.read_bytes()) if format_path.exists() else None
        self._canonical_blocks_requested = canonical_blocks
        self._canonical_blocks = retained_format == "blocks-v1" if canonical_blocks is None else canonical_blocks
        if retained_format is not None and retained_format != ("blocks-v1" if self.canonical_blocks else "tensors-v1"):
            raise ValueError("Canonical format differs from the retained store")
        if self.canonical_blocks and retained_format is None:
            atomic_write(format_path, canonical("blocks-v1"))

    @property
    def canonical_blocks(self) -> bool:
        # A transport may open an empty store before its publisher starts.
        # Adopt the publisher's persisted format when the first version arrives.
        if not self._canonical_blocks:
            path = self.root / "canonical-format.json"
            if path.is_file() and load_json(path.read_bytes()) == "blocks-v1":
                if self._canonical_blocks_requested is False:
                    raise ValueError("Canonical format changed; reopen with the recorded format")
                self._canonical_blocks = True
        return self._canonical_blocks

    def _check_canonical_placement(self) -> None:
        # A transport can open an empty store before its publisher selects
        # canonical placement. Adopt the persisted location before FULL repair
        # or collection; the location is immutable after first publication.
        if self._canonical_layout is None:
            path = self.root / "canonical-location.json"
            if path.is_file():
                layout = load_json(path.read_bytes())
                self.canonical_directory = Path(layout["root"]) / valid_hash(layout["namespace"])
                self.canonical_budget = layout["budget"]
                self._remaining_canonical_bytes = None
                self._canonical_layout = layout
        if self._canonical_layout is not None:
            marker = self.canonical_directory / "placement.json"
            if not marker.is_file() or load_json(marker.read_bytes()) != self._canonical_layout:
                raise OSError(
                    "Canonical volume is unavailable or belongs to another store; restore the recorded mount "
                    "or request FULL repair from the publication owner through TCP"
                )

    def _reserve_canonical(self, size: int) -> None:
        with self._budget_lock:
            self._check_canonical_placement()
            if self.canonical_budget is None:
                self._reserve(size)
                return
            if self._remaining_canonical_bytes is None:
                used = stored_bytes(self.canonical_directory)
                self._remaining_canonical_bytes = self.canonical_budget - used
            if size > self._remaining_canonical_bytes:
                raise OSError(
                    "Canonical disk budget exhausted; retain required versions and increase capacity or collect unused versions"
                )
            self._remaining_canonical_bytes -= size

    def _reserve(self, size: int) -> None:
        with self._budget_lock:
            if self._remaining_bytes is None:
                used = stored_bytes(self.root)
                self._remaining_bytes = self.disk_budget - used
            if size > self._remaining_bytes:
                raise OSError("Weight store disk budget exhausted")
            # Conservative within one writer operation: replacing a file does not
            # refund its old size until the next scan. Temporary-file peak fits too.
            self._remaining_bytes -= size

    def refresh_budget(self) -> None:
        """Start a new allocation epoch while holding the store writer lock."""
        self._remaining_bytes = None
        self._remaining_canonical_bytes = None

    def _control_write(self, action):
        """Refill under content -> metadata lock order; spending needs only
        metadata."""
        held = (os.getpid(), str((self.root / ".content-writer").resolve())) in _held_content_locks.get()
        with writer_lock(self.root / ".writer", timeout=30):
            try:
                return action(held)
            except _ControlCapacityRequired:
                pass
        with writer_lock(self.root / ".content-writer", timeout=120), writer_lock(self.root / ".writer", timeout=30):
            return action(True)

    def _reserve_control(self, size: int, *, refill: bool) -> None:
        """Spend already admitted metadata capacity under the writer lock.

        The reservation is an ordinary counted file, so old bulk writers also
        account for it. Its durable shrink precedes the metadata write: a crash
        can lose credit, never reuse it. Refill performs normal admission. No
        cached estimate of the growing object tree is used on the fast path.
        """
        path = self.root / ".control-reservation-v1"
        header = canonical({"format": "control-reservation-v1", "disk_budget": self.disk_budget}) + b"\n"
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        with self._budget_lock:
            check_deadline("control metadata reservation")
            try:
                fd = os.open(path, flags)
            except FileNotFoundError:
                fd = None
            if fd is not None:
                with os.fdopen(fd, "r+b", buffering=0) as reservation:
                    info = os.fstat(reservation.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                        raise OSError("Control reservation must be a private regular file")
                    credit = info.st_size - len(header)
                    if reservation.read(len(header)) == header and size <= credit <= 1024 * 1024:
                        reservation.truncate(info.st_size - size)
                        os.fsync(reservation.fileno())
                        check_deadline("control metadata reservation")
                        return

            if not refill:
                raise _ControlCapacityRequired()
            # Different budgets, missing/partial reservations and exhausted
            # credit all require an exact scan before allocating new bytes.
            self._remaining_bytes = None
            self._reserve(size)
            # Keep the reserve small relative to both quota and free capacity.
            credit = min(1024 * 1024, self.disk_budget // 1024, self._remaining_bytes // 64)
            if credit < size or credit + len(header) > self._remaining_bytes:
                return
            self._reserve(len(header) + credit)
            fd = os.open(path, flags | os.O_CREAT, 0o600)
            with os.fdopen(fd, "r+b", buffering=0) as reservation:
                info = os.fstat(reservation.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise OSError("Control reservation must be a private regular file")
                reservation.write(header)
                reservation.truncate(len(header) + credit)
                os.fsync(reservation.fileno())
            sync_directory(self.root)
            check_deadline("control metadata reservation")

    def pin(self, stream: str, version: int, owner: str) -> None:
        """Protect a version and its dependency chain while it is in use."""
        self._control_write(lambda refill: self._pin_locked(stream, version, owner, refill=refill))

    def _pin_locked(self, stream: str, version: int, owner: str, *, refill: bool) -> None:
        """Called under the writer lock, including the fetch commit
        transaction."""
        manifest = self.manifest(stream, version)
        if (self.root / "released-fetches" / (valid_id(owner) + ".json")).exists():
            raise ValueError("Fetch retention operation was already released")
        path = self.root / "pins" / (valid_id(owner) + ".json")
        content = canonical({"stream": stream, "version": version, "root": manifest["root"]})
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("Retention owner already protects another version; release it first")
            return
        self._reserve_control(len(content), refill=refill)
        atomic_write(path, content)

    def unpin(self, owner: str) -> None:
        def mutate(refill):
            path = self.root / "pins" / (valid_id(owner) + ".json")
            if owner.startswith("fetch-"):
                # A delayed acquire must not recreate a pin after its release.
                self._reserve_control(2, refill=refill)
                atomic_write(self.root / "released-fetches" / (owner + ".json"), b"{}")
            path.unlink(missing_ok=True)
            if path.parent.exists():
                sync_directory(path.parent)

        self._control_write(mutate)

    def register(self, consumer: str, stream: str, start_version: int = 0, *, mode: str = "all_versions") -> dict:
        """Retain unconsumed history for all_versions, or the active target for
        latest."""
        if mode not in ("all_versions", "latest"):
            raise ValueError("Consumption mode must be all_versions or latest")
        valid_id(consumer)
        self.manifest_path(stream, start_version)

        def mutate(refill):
            path = self.root / "consumers" / (consumer + ".json")
            if path.exists():
                state = load_json(path.read_bytes())
                if (
                    state["stream"] != stream
                    or state["start_version"] != start_version
                    or state.get("mode", "all_versions") != mode
                ):
                    raise ValueError("Consumer identity already has a different subscription")
                return state
            versions = sorted(int(p.stem) for p in (self.root / "versions" / stream).glob("*.json"))
            missing = [v for v in range(start_version, versions[-1] + 1) if v not in versions] if versions else []
            if missing and mode == "all_versions":
                raise ValueError(f"Requested history was collected: {missing}")
            state = {
                "consumer": consumer,
                "stream": stream,
                "start_version": start_version,
                "mode": mode,
                "cursor": start_version - 1,
                "active": None,
            }
            encoded = canonical(state)
            self._reserve_control(len(encoded), refill=refill)
            atomic_write(path, encoded)
            return state

        return self._control_write(mutate)

    def acknowledge(self, consumer: str, receipt: dict) -> dict:
        def mutate(refill):
            path = self.root / "consumers" / (valid_id(consumer) + ".json")
            state = load_json(path.read_bytes())
            target = receipt["target"]
            if receipt.get("phase") != "ACTIVE" or receipt.get("serving", target) != target:
                raise ValueError("Only the requested ACTIVE target can advance consumption")
            if target["stream"] != state["stream"] or type(target["version"]) is not int:
                raise ValueError("Receipt belongs to another subscription")
            if state["active"] == target:
                return state
            if state.get("mode", "all_versions") == "all_versions" and target["version"] != state["cursor"] + 1:
                raise ValueError("all_versions acknowledgement must be contiguous")
            if target["version"] <= state["cursor"]:
                raise ValueError("Consumption cannot move backwards or change an acknowledged target")
            if self.manifest(target["stream"], target["version"])["root"] != target["root"]:
                raise ValueError("Receipt root differs from published content")
            state.update(cursor=target["version"], active=target)
            data = canonical(state)
            self._reserve_control(len(data), refill=refill)
            atomic_write(path, data)
            return state

        return self._control_write(mutate)

    def unregister(self, consumer: str) -> None:
        with writer_lock(self.root / ".writer", timeout=30):
            path = self.root / "consumers" / (valid_id(consumer) + ".json")
            path.unlink(missing_ok=True)
            if path.parent.exists():
                sync_directory(path.parent)

    def consumer_status(self, consumer: str) -> dict:
        return load_json((self.root / "consumers" / (valid_id(consumer) + ".json")).read_bytes())

    def versions(self, stream: str, *, after: int = -1, limit: int = 100, newest_first: bool = False) -> list[dict]:
        valid_id(stream)
        if (
            type(after) is not int
            or after < -1
            or type(limit) is not int
            or not 1 <= limit <= 1000
            or type(newest_first) is not bool
        ):
            raise ValueError("Invalid version listing range")
        with writer_lock(self.root / ".writer", timeout=30):
            numbers = sorted(
                (int(path.stem) for path in (self.root / "versions" / stream).glob("*.json")), reverse=newest_first
            )
            return [
                {key: manifest[key] for key in ("stream", "version", "root", "artifact_id", "update_id", "kind")}
                for manifest in (
                    self.manifest(stream, version)
                    for version in [number for number in numbers if number > after][:limit]
                )
            ]

    def collect(self, *, retain_anchors: int = 2) -> dict:
        """Keep recent anchors, their following versions, and explicitly pinned
        chains.

        Active readers must pin before opening files. This is explicit
        maintenance, not an implicit fallback that discards data when a write
        exceeds quota.
        """
        if retain_anchors < 2:
            raise ValueError("At least two anchors must be retained")
        with writer_lock(self.root / ".content-writer", timeout=120), writer_lock(self.root / ".writer", timeout=30):
            self._check_canonical_placement()
            self.refresh_budget()
            self._replay_deletions()
            self._sweep_unpublished()
            manifests = {}
            for path in (self.root / "versions").glob("*/*.json"):
                m = self.manifest(path.parent.name, int(path.stem))
                manifests[m["stream"], m["version"]] = m
            keep = set()
            for stream in {s for s, _ in manifests}:
                versions = sorted(v for s, v in manifests if s == stream)
                anchors = [v for v in versions if manifests[stream, v]["kind"] == "full"]
                if not anchors:
                    raise ValueError("Cannot collect a stream without an anchor")
                cutoff = anchors[max(0, len(anchors) - retain_anchors)]
                keep.update((stream, v) for v in versions if v >= cutoff)
            for path in (self.root / "pins").glob("*.json"):
                pin = json.loads(path.read_bytes())
                key = (pin["stream"], pin["version"])
                if key not in manifests or manifests[key]["root"] != pin["root"]:
                    raise ValueError("Invalid pin; refusing collection")
                keep.add(key)
            for path in (self.root / "consumers").glob("*.json"):
                state = load_json(path.read_bytes())
                if state.get("mode", "all_versions") == "all_versions":
                    keep.update((s, v) for s, v in manifests if s == state["stream"] and v > state["cursor"])
                if state["active"]:
                    keep.add((state["stream"], state["cursor"]))
            for path in (self.root / "seals").glob("*.json"):
                from .seal import read_seal

                sealed = read_seal(self, path.stem)
                keep.update((sealed["stream"], entry["version"]) for entry in sealed["closure"])
            for key in list(keep):
                m = manifests[key]
                while m["kind"] == "delta":
                    key = (m["stream"], m["base_version"])
                    m = manifests[key]
                    keep.add(key)
            removed_versions = 0
            for key, m in manifests.items():
                if key in keep:
                    continue
                tombstone = self.root / "deletions" / f"{key[0]}-{key[1]}.json"
                atomic_write(tombstone, canonical(m))
                self._delete_version(tombstone, m)
                removed_versions += 1
            objects = {c["object"] for key in keep for t in manifests[key]["tensors"] for c in t["chunks"]}
            for key in keep:
                if self.full_path(*key).exists():
                    full = self.manifest(*key, prefer_full=True)
                    objects.update(c["object"] for t in full["tensors"] for c in t["chunks"])
            freed = 0
            for path in (self.root / "objects").glob("*"):
                if path.is_file() and path.name not in objects and re.fullmatch(r"[0-9a-f]{64}", path.name):
                    freed += path.stat().st_size
                    path.unlink()
            if (self.root / "objects").exists():
                sync_directory(self.root / "objects")
            raw_freed = 0
            live_raw = {c["output"] for key in keep for t in manifests[key]["tensors"] for c in t["chunks"]}
            block_directory = self.canonical_directory / "blocks"
            for path in block_directory.glob("*"):
                if path.is_file() and re.fullmatch(r"[0-9a-f]{64}", path.name) and path.name not in live_raw:
                    raw_freed += path.stat().st_size
                    path.unlink()
            if block_directory.exists():
                sync_directory(block_directory)
            return {"removed_versions": removed_versions, "object_bytes_freed": freed, "raw_bytes_freed": raw_freed}

    def _delete_version(self, tombstone: Path, manifest: dict) -> None:
        stream, version = manifest["stream"], manifest["version"]
        path = self.manifest_path(stream, version)
        path.unlink(missing_ok=True)
        if path.parent.exists():
            sync_directory(path.parent)
        for tensor in manifest["tensors"]:
            raw = self.tensor_path(stream, version, tensor["name"])
            raw.unlink(missing_ok=True)
            if raw.parent.exists():
                sync_directory(raw.parent)
        full = self.full_path(stream, version)
        full.unlink(missing_ok=True)
        if full.parent.exists():
            sync_directory(full.parent)
        tombstone.unlink()
        sync_directory(tombstone.parent)

    def _replay_deletions(self) -> None:
        for path in (self.root / "deletions").glob("*.json"):
            manifest = validate_manifest(load_json(path.read_bytes()), self.max_weight_bytes)
            self._delete_version(path, manifest)

    def _sweep_unpublished(self) -> None:
        # The content writer lock excludes every publisher/reconstructor. Only
        # recognized store-owned raw files and interrupted atomic writes qualify.
        for directory in self.canonical_directory.glob("*/*"):
            if not directory.is_dir() or not directory.name.isdecimal():
                continue
            committed = self.manifest_path(directory.parent.name, int(directory.name)).exists()
            for path in directory.iterdir():
                if path.is_file() and (
                    path.name.startswith(".partial-")
                    or (not committed and re.fullmatch(r"[0-9a-f]{64}\.bin", path.name))
                ):
                    path.unlink()
            sync_directory(directory)
        for directory in (self.root / "objects", self.canonical_directory / "blocks"):
            for path in directory.glob(".partial-*"):
                path.unlink()

    def manifest_path(self, stream: str, version: int) -> Path:
        if type(version) is not int or version < 0:
            raise ValueError("Invalid version")
        return self.root / "versions" / valid_id(stream) / f"{version}.json"

    def object_path(self, key: str) -> Path:
        return self.root / "objects" / valid_hash(key)

    def full_path(self, stream: str, version: int) -> Path:
        self.manifest_path(stream, version)
        return self.root / "full" / stream / f"{version}.json"

    def manifest(self, stream: str, version: int, *, prefer_full: bool = False) -> dict:
        if (self.root / "deletions" / f"{valid_id(stream)}-{version}.json").exists():
            raise FileNotFoundError("Version is being collected")
        path = (
            self.full_path(stream, version)
            if prefer_full and self.full_path(stream, version).exists()
            else self.manifest_path(stream, version)
        )
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest too large")
        m = validate_manifest(load_json(path.read_bytes()), self.max_weight_bytes)
        if m["stream"] != stream or m["version"] != version:
            raise ValueError("Manifest identity mismatch")
        return m

    def put_object(self, key: str, payload: bytes) -> None:
        lock = self._content_locks[int(valid_hash(key)[:8], 16) % len(self._content_locks)]
        with lock:
            if len(payload) > MAX_CHUNK_BYTES or digest(payload) != key:
                raise ValueError("Object integrity mismatch")
            path = self.object_path(key)
            if path.exists() and path.stat().st_size > MAX_CHUNK_BYTES:
                raise ValueError("Existing object exceeds the size bound")
            if path.exists() and digest(path.read_bytes()) == key:
                return
            self._reserve(len(payload))
            atomic_write(path, payload)

    def commit(self, m: dict) -> None:
        check_deadline("manifest commit")
        validate_manifest(m, self.max_weight_bytes)
        encoded = canonical(m)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest too large")

        def verify_object(chunk: dict) -> None:
            check_deadline("durable object verification")
            obj = self.object_path(chunk["object"])
            if obj.stat().st_size != chunk["encoded_size"] or digest(obj.read_bytes()) != chunk["object"]:
                raise ValueError("Missing or corrupt durable object")

        # Bound both concurrent reads and live block buffers. All checks finish
        # before metadata becomes visible; worker failures retain that boundary.
        chunks = (chunk for tensor in m["tensors"] for chunk in tensor["chunks"])
        for _ in ordered_map(verify_object, chunks, 8):
            pass
        with writer_lock(self.root / ".writer", timeout=30):
            self._commit_metadata(m, encoded)

    def _commit_metadata(self, m: dict, encoded: bytes) -> None:
        path = self.manifest_path(m["stream"], m["version"])
        if path.exists():
            prior = self.manifest(m["stream"], m["version"])
            if prior == m:
                return
            if prior["root"] != m["root"] or prior.get("update_id") != m.get("update_id"):
                raise ValueError("Version identity reused with different content")
            if m["kind"] != "full":
                raise ValueError("Only a FULL repair package may supplement an existing version")
            path = self.full_path(m["stream"], m["version"])
            if path.exists():
                if self.manifest(m["stream"], m["version"], prefer_full=True) != m:
                    raise ValueError("FULL artifact already exists")
                return
        if m["kind"] == "delta":
            base = self.manifest(m["stream"], m["base_version"])
            if base["root"] != m["base_root"]:
                raise ValueError("Base root mismatch")
            schema = lambda x: [(t["name"], t["dtype"], t["shape"]) for t in x["tensors"]]
            if schema(m) != schema(base):
                raise ValueError("Delta schema changed")
            if m.get("schema_context") != base.get("schema_context"):
                raise ValueError("Delta conversion context changed")
        self._reserve(len(encoded))
        atomic_write(path, encoded)

    def tensor_path(self, stream: str, version: int, name: str) -> Path:
        self.manifest_path(stream, version)
        self._check_canonical_placement()
        return self.canonical_directory / stream / str(version) / (digest(name.encode()) + ".bin")

    def raw_block_path(self, output: str) -> Path:
        self._check_canonical_placement()
        return self.canonical_directory / "blocks" / valid_hash(output)

    def _put_raw_block(self, value: bytes, output: str) -> None:
        lock = self._content_locks[int(valid_hash(output)[:8], 16) % len(self._content_locks)]
        with lock:
            if digest(value) != output:
                raise ValueError("Canonical block content mismatch")
            path = self.raw_block_path(output)
            if path.is_file() and path.stat().st_size == len(value) and digest(path.read_bytes()) == output:
                return
            self._reserve_canonical(len(value))
            atomic_write(path, value)

    def read_chunk(self, stream: str, version: int, name: str, chunk: dict) -> bytes:
        if self.canonical_blocks and self.raw_block_path(chunk["output"]).is_file():
            path = self.raw_block_path(chunk["output"])
            if path.stat().st_size != chunk["size"]:
                raise ValueError("Canonical block size differs from the manifest")
            value = path.read_bytes()
        else:
            # Retained tensor-format stores can be converted block by block.
            with self.tensor_path(stream, version, name).open("rb") as source:
                source.seek(chunk["offset"])
                value = source.read(chunk["size"])
        if len(value) != chunk["size"] or digest(value) != chunk["output"]:
            raise ValueError("Canonical content differs from the manifest")
        return value

    def iter_tensor(self, manifest: dict, tensor: dict):
        for chunk in tensor["chunks"]:
            check_deadline("canonical read")
            yield self.read_chunk(manifest["stream"], manifest["version"], tensor["name"], chunk)

    def _materialize_blocks(self, stream: str, version: int) -> dict:
        chain = []
        manifest = self.manifest(stream, version, prefer_full=True)
        current = manifest
        while not self._valid_materialized(current):
            chain.append(current)
            if current["kind"] == "full":
                break
            if len(chain) >= 4096:
                raise ValueError("Chain exceeds catch-up bound")
            base = self.manifest(stream, current["base_version"], prefer_full=True)
            if base["root"] != current["base_root"]:
                raise ValueError("Base root mismatch")
            current = base
        for current in reversed(chain):
            previous = None
            if current["kind"] == "delta":
                previous = self.manifest(stream, current["base_version"])
            previous_tensors = {t["name"]: t for t in previous["tensors"]} if previous else {}

            def reconstruct(job):
                tensor, index, chunk = job
                check_deadline("canonical block reconstruction")
                try:
                    value = self.read_chunk(stream, current["version"], tensor["name"], chunk)
                except (FileNotFoundError, ValueError):
                    pass
                else:
                    if not self.raw_block_path(chunk["output"]).is_file():
                        self._put_raw_block(value, chunk["output"])
                    return  # Existing immutable bytes were fully verified above.
                base = None
                if chunk["kind"] not in ("raw", "raw_zlib"):
                    base_chunk = previous_tensors[tensor["name"]]["chunks"][index]
                    if (base_chunk["offset"], base_chunk["size"]) != (chunk["offset"], chunk["size"]):
                        raise ValueError("Delta chunk layout differs from its base")
                    base = self.read_chunk(stream, current["base_version"], tensor["name"], base_chunk)
                path = self.object_path(chunk["object"])
                if path.stat().st_size != chunk["encoded_size"]:
                    raise ValueError("Object size differs from manifest")
                payload = path.read_bytes()
                if digest(payload) != chunk["object"]:
                    raise ValueError("Corrupt object")
                value = decode(chunk["kind"], payload, base, DTYPES[tensor["dtype"]], chunk["size"])
                self._put_raw_block(value, chunk["output"])

            jobs = (
                (tensor, index, chunk) for tensor in current["tensors"] for index, chunk in enumerate(tensor["chunks"])
            )
            # Finish and drain this version before any dependent version starts.
            for _ in ordered_map(reconstruct, jobs, 8):
                pass
        return manifest

    def _valid_materialized(self, m: dict) -> bool:
        if self.canonical_blocks:

            def verify_chunk(job):
                name, chunk = job
                check_deadline("canonical verification")
                self.read_chunk(m["stream"], m["version"], name, chunk)

            jobs = ((tensor["name"], chunk) for tensor in m["tensors"] for chunk in tensor["chunks"])
            try:
                for _ in ordered_map(verify_chunk, jobs, 8):
                    pass
                return True
            except (OSError, ValueError):
                return False
        for tensor in m["tensors"]:
            path = self.tensor_path(m["stream"], m["version"], tensor["name"])
            if not path.is_file():
                return False
            with path.open("rb") as source:
                for chunk in tensor["chunks"]:
                    check_deadline("canonical verification")
                    if digest(source.read(chunk["size"])) != chunk["output"]:
                        return False
                if source.read(1):
                    return False
        return True

    def has_verified_version(self, stream: str, version: int, root: str) -> bool:
        """Check both manifest identity and all locally materialized bytes."""
        try:
            manifest = self.manifest(stream, version)
            return manifest["root"] == root and self._valid_materialized(manifest)
        except (OSError, ValueError):
            return False

    def materialize(self, stream: str, version: int) -> dict:
        with writer_lock(self.root / ".content-writer", timeout=120):
            return self._materialize(stream, version)

    def _materialize(self, stream: str, version: int) -> dict:
        if self.canonical_blocks:
            return self._materialize_blocks(stream, version)
        chain = []
        m = self.manifest(stream, version, prefer_full=True)
        if self._valid_materialized(m):
            return m
        while True:
            check_deadline("canonical dependency chain")
            chain.append(m)
            if m["kind"] == "full":
                break
            if len(chain) >= 4096:
                raise ValueError("Chain exceeds catch-up bound")
            previous = self.manifest(stream, m["base_version"])
            if previous["root"] != m["base_root"]:
                raise ValueError("Base root mismatch")
            if self._valid_materialized(previous):
                break
            m = previous
        for m in reversed(chain):
            for tensor in m["tensors"]:
                target = self.tensor_path(stream, m["version"], tensor["name"])
                target.parent.mkdir(parents=True, exist_ok=True)
                # Revalidate caches after restart; a marker alone is not evidence.
                if target.exists():
                    with target.open("rb") as cached:
                        matches = True
                        for c in tensor["chunks"]:
                            check_deadline("cached tensor verification")
                            if digest(cached.read(c["size"])) != c["output"]:
                                matches = False
                                break
                        if matches and not cached.read(1):
                            continue
                self._reserve_canonical(sum(c["size"] for c in tensor["chunks"]))
                fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=target.parent)
                try:
                    with os.fdopen(fd, "wb") as out:
                        for chunk in tensor["chunks"]:
                            check_deadline("canonical reconstruction")
                            base = None
                            if chunk["kind"] not in ("raw", "raw_zlib"):
                                with self.tensor_path(stream, m["base_version"], tensor["name"]).open("rb") as source:
                                    source.seek(chunk["offset"])
                                    base = source.read(chunk["size"])
                            object_path = self.object_path(chunk["object"])
                            if object_path.stat().st_size != chunk["encoded_size"]:
                                raise ValueError("Object size differs from manifest")
                            payload = object_path.read_bytes()
                            if len(payload) != chunk["encoded_size"] or digest(payload) != chunk["object"]:
                                raise ValueError("Corrupt object")
                            value = decode(chunk["kind"], payload, base, DTYPES[tensor["dtype"]], chunk["size"])
                            if digest(value) != chunk["output"]:
                                raise ValueError("Reconstructed weight mismatch")
                            from .traffic import file_write

                            file_write(target, out.write(value))
                        out.flush()
                        os.fsync(out.fileno())
                    os.replace(temporary, target)
                    sync_directory(target.parent)
                finally:
                    Path(temporary).unlink(missing_ok=True)
        return self.manifest(stream, version)

    def publish(
        self,
        stream: str,
        version: int,
        tensors: dict,
        *,
        anchor_interval: int = 20,
        chunk_bytes: int = MAX_CHUNK_BYTES,
        full_only: bool = False,
        compression: bool = False,
        encode_workers: int = 4,
        model_config: dict | None = None,
        converter_revision: str = "opaque-bytes-v1",
        update_id: str | None = None,
        source: dict | None = None,
    ) -> dict:
        """Publish immutable input files: name -> (dtype, shape, file path).

        Call at a quiescent training boundary. The caller owns the input files
        until publication returns; mutating them concurrently is unsupported.
        """
        if anchor_interval < 1 or not 8 <= chunk_bytes <= MAX_CHUNK_BYTES or chunk_bytes % 8:
            raise ValueError("Invalid anchor or chunk size")
        if type(compression) is not bool:
            raise ValueError("compression must be a boolean")
        if type(encode_workers) is not int or encode_workers < 1:
            raise ValueError("encode_workers must be a positive integer")
        # HF configurations may contain integer dictionary keys and tuples.
        # Bind the wire representation, which is also used after a restart.
        model_config = load_json(canonical(model_config))
        with writer_lock(self.root / ".content-writer", timeout=120):
            self._check_canonical_placement()
            self.refresh_budget()
            base = None
            if version and not full_only and version % anchor_interval:
                base = self._materialize(stream, version - 1)
                context = {
                    "model_config_sha256": digest(canonical(model_config)) if model_config is not None else None,
                    "model_config": model_config,
                    "converter_revision": converter_revision,
                }
                schema = [
                    (name, dtype, list(shape))
                    for name, (dtype, shape, _) in sorted(tensors.items(), key=lambda item: item[0].encode("utf-8"))
                ]
                previous_schema = [(t["name"], t["dtype"], t["shape"]) for t in base["tensors"]]
                if context != base["schema_context"] or schema != previous_schema:
                    base = None
            m = {
                "format": 1,
                "byteorder": "little",
                "stream": valid_id(stream),
                "version": version,
                "update_id": str(version) if update_id is None else str(update_id),
                "kind": "delta" if base else "full",
                "base_version": version - 1 if base else None,
                "base_root": base["root"] if base else None,
                "tensors": [],
                "schema_context": {
                    "model_config_sha256": digest(canonical(model_config)) if model_config is not None else None,
                    "model_config": model_config,
                    "converter_revision": converter_revision,
                },
            }
            if compression:
                m["compression"] = True
            if source is not None:
                m["source"] = load_json(canonical(source))
            total = 0
            previous_tensors = {t["name"]: t for t in base["tensors"]} if base else {}
            jobs = []
            for name, (dtype, shape, path) in sorted(tensors.items(), key=lambda item: item[0].encode("utf-8")):
                size = math.prod(shape) * DTYPES[dtype]
                total += size
                if total > self.max_weight_bytes or Path(path).stat().st_size != size:
                    raise ValueError("Input size does not match schema or budget")
                tensor = {"name": name, "dtype": dtype, "shape": list(shape), "chunks": []}
                previous_chunks = {c["offset"]: c for c in previous_tensors[name]["chunks"]} if base else {}
                for offset in range(0, size, chunk_bytes):
                    length = min(chunk_bytes, size - offset)
                    previous_chunk = previous_chunks.get(offset)
                    if base and (previous_chunk is None or previous_chunk["size"] != length):
                        raise ValueError("Chunk layout changed; publish a FULL version")
                    jobs.append((tensor, path, offset, length, previous_chunk))
                m["tensors"].append(tensor)

            def encode_chunk(job):
                tensor, path, offset, length, previous_chunk = job
                check_deadline("publication block read")
                with Path(path).open("rb") as source:
                    source.seek(offset)
                    current = source.read(length)
                if len(current) != length:
                    raise ValueError("Source changed while publication was pending")
                prior = self.read_chunk(stream, version - 1, tensor["name"], previous_chunk) if base else None
                kind, payload = encode(current, prior, DTYPES[tensor["dtype"]], compression=compression)
                key = digest(payload)
                self.put_object(key, payload)
                return (
                    tensor,
                    {
                        "offset": offset,
                        "size": length,
                        "kind": kind,
                        "object": key,
                        "encoded_size": len(payload),
                        "output": digest(current),
                    },
                )

            # Immutable objects may complete out of order. Per-content locks
            # deduplicate writes, budget reservations are serialized, and one
            # owner assembles the ordered manifest after all writes complete.
            with durable_directory_batch(self.root / "objects"):
                for tensor, chunk in ordered_map(encode_chunk, jobs, encode_workers):
                    tensor["chunks"].append(chunk)
            identify(m)
            validate_manifest(m, self.max_weight_bytes)
            if self.manifest_path(stream, version).exists():
                prior = self.manifest(stream, version)
                if prior != m:
                    raise ValueError("Version identity reused with different publication")
                return prior
            # Persist the exact logical target before making its manifest visible.
            # FULL repair is independent of the source process and older anchors.
            previous = {}
            if version and self.manifest_path(stream, version - 1).exists():
                previous = {t["name"]: t for t in self.manifest(stream, version - 1)["tensors"]}
            if self.canonical_blocks:

                def persist_block(job):
                    tensor, path, offset, length, _ = job
                    with Path(path).open("rb") as source:
                        source.seek(offset)
                        value = source.read(length)
                    chunk = tensor["chunks"][offset // chunk_bytes]
                    if len(value) != length or digest(value) != chunk["output"]:
                        raise ValueError("Source changed while publication was pending")
                    self._put_raw_block(value, chunk["output"])

                with durable_directory_batch(self.canonical_directory / "blocks"):
                    for _ in ordered_map(persist_block, jobs, encode_workers):
                        pass
                if any(
                    Path(path).stat().st_size != math.prod(shape) * DTYPES[dtype]
                    for dtype, shape, path in tensors.values()
                ):
                    raise ValueError("Source grew while publication was pending")
            for tensor in m["tensors"]:
                if self.canonical_blocks:
                    continue
                target = self.tensor_path(stream, version, tensor["name"])
                ensure_directory(target.parent)
                prior = previous.get(tensor["name"])
                if prior and [(c["offset"], c["size"], c["output"]) for c in prior["chunks"]] == [
                    (c["offset"], c["size"], c["output"]) for c in tensor["chunks"]
                ]:
                    old = self.tensor_path(stream, version - 1, tensor["name"])
                    if old.is_file():
                        with old.open("rb") as source:
                            verified = all(digest(source.read(c["size"])) == c["output"] for c in tensor["chunks"])
                            verified = verified and not source.read(1)
                        if verified:
                            temporary = target.parent / (".partial-link-" + digest(tensor["name"].encode()))
                            temporary.unlink(missing_ok=True)
                            os.link(old, temporary)
                            os.replace(temporary, target)
                            sync_directory(target.parent)
                            continue
                self._reserve_canonical(sum(c["size"] for c in tensor["chunks"]))
                fd, temporary = tempfile.mkstemp(prefix=".partial-", dir=target.parent)
                try:
                    with os.fdopen(fd, "wb") as out, Path(tensors[tensor["name"]][2]).open("rb") as source:
                        for chunk in tensor["chunks"]:
                            value = source.read(chunk["size"])
                            if digest(value) != chunk["output"]:
                                raise ValueError("Source changed while publication was pending")
                            from .traffic import file_write

                            file_write(target, out.write(value))
                        if source.read(1):
                            raise ValueError("Source grew while publication was pending")
                        out.flush()
                        os.fsync(out.fileno())
                    os.replace(temporary, target)
                    sync_directory(target.parent)
                finally:
                    Path(temporary).unlink(missing_ok=True)
            self.commit(m)
            return m

    def ensure_full(self, stream: str, version: int) -> dict:
        """Build a same-target FULL package using retained canonical bytes."""
        with writer_lock(self.root / ".content-writer", timeout=120):
            self.refresh_budget()
            original = self.manifest(stream, version, prefer_full=True)
            if original["kind"] == "full":

                def repair_object(job):
                    tensor, chunk = job
                    check_deadline("FULL object verification/repair")
                    path = self.object_path(chunk["object"])
                    if (
                        path.is_file()
                        and path.stat().st_size == chunk["encoded_size"]
                        and digest(path.read_bytes()) == chunk["object"]
                    ):
                        return
                    try:
                        value = self.read_chunk(stream, version, tensor["name"], chunk)
                    except (OSError, ValueError) as error:
                        raise ValueError("Canonical target cannot repair this FULL object") from error
                    if digest(value) != chunk["output"]:
                        raise ValueError("Canonical target cannot repair this FULL object")
                    payload = zlib.compress(value, level=1) if chunk["kind"] == "raw_zlib" else value
                    if digest(payload) != chunk["object"]:
                        raise ValueError("Repaired FULL encoding differs from the immutable artifact")
                    self.put_object(chunk["object"], payload)

                jobs = ((tensor, chunk) for tensor in original["tensors"] for chunk in tensor["chunks"])
                for _ in ordered_map(repair_object, jobs, 4):
                    pass
                return original
            self._materialize(stream, version)
            full = load_json(canonical(original))
            full.update(kind="full", base_version=None, base_root=None)

            def encode_full(job):
                tensor, chunk = job
                check_deadline("FULL encoding")
                value = self.read_chunk(stream, version, tensor["name"], chunk)
                kind, payload = encode(
                    value, None, DTYPES[tensor["dtype"]], compression=original.get("compression", False)
                )
                key = digest(payload)
                self.put_object(key, payload)
                return chunk, kind, key, len(payload)

            jobs = ((tensor, chunk) for tensor in full["tensors"] for chunk in tensor["chunks"])
            for chunk, kind, key, size in ordered_map(encode_full, jobs, 4):
                chunk.update(kind=kind, object=key, encoded_size=size)
            identify(full)
            if full["root"] != original["root"]:
                raise ValueError("FULL repair changed logical content")
            self.commit(full)
            return full
