# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model-independent, bounded storage for deterministic frozen conditions.

Only adapters decide what is frozen and construct the content identity. This
module never encodes images, samples randomness, or owns model parameters.
Returned trees are private copies: callers cannot append to a shared KV cache.

Two stores share one interface. :class:`ConditionFeatureCache` is a
process-local LRU. :class:`SharedConditionStore` keeps one file per entry in a
node-local directory (``/dev/shm`` by default) shared by every data-parallel
rank of one run. The SFT sampler reshuffles samples across ranks every epoch,
and sharded encoders may only be skipped when *all* ranks hit, so a
process-local cache almost never yields a collective hit on more than one GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import socket
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class FrozenConditionSpec:
    encoder_identity: str
    preprocessing_version: str
    dtype: str
    feature_schema: str
    adapter_version: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in asdict(self).values()):
            raise ValueError("Every frozen-condition identity component must be a nonempty string")


def _digest_value(digest: Any, value: Any) -> None:
    """Length-framed, type-sensitive hashing; never transfers a CUDA tensor."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Condition keys must be built from CPU content, not CUDA tensors")
        if value.layout != torch.strided:
            raise TypeError("Condition identity only supports dense tensors")
        _digest_value(digest, ("tensor", str(value.dtype), tuple(value.shape)))
        value = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    if isinstance(value, bytes):
        digest.update(b"b" + str(len(value)).encode() + b":" + value)
    elif isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Condition tree mapping keys must be strings")
        digest.update(b"d" + str(len(value)).encode() + b":")
        for key in sorted(value):
            _digest_value(digest, key)
            _digest_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update((b"t" if isinstance(value, tuple) else b"l") + str(len(value)).encode() + b":")
        for item in value:
            _digest_value(digest, item)
    elif value is None or type(value) in (str, int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite condition identity")
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        digest.update(type(value).__name__.encode() + str(len(encoded)).encode() + b":" + encoded)
    else:
        raise TypeError(f"Unsupported condition identity type: {type(value).__name__}")


def condition_cache_key(spec: FrozenConditionSpec, content: Any) -> str:
    digest = hashlib.sha256()
    _digest_value(digest, asdict(spec))
    _digest_value(digest, content)
    return digest.hexdigest()


def _walk_tensors(tree: Any):
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        if not all(isinstance(key, str) for key in tree):
            raise TypeError("Condition tree mapping keys must be strings")
        for value in tree.values():
            yield from _walk_tensors(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from _walk_tensors(value)
    elif tree is not None and type(tree) not in (str, int, float, bool):
        raise TypeError(f"Unsupported condition payload type: {type(tree).__name__}")


def _copy_tree(tree: Any, device: torch.device | None) -> Any:
    if isinstance(tree, torch.Tensor):
        # copy=True protects both same-device and cross-device consumers. Never
        # cast: cache readback must preserve the encoder's exact dtype and bits.
        # Version-tracked storage also works when a caller fills the cache in
        # inference_mode; inference tensors themselves have no version counter.
        with torch.inference_mode(False):
            return tree.detach().to(device=device or tree.device, copy=True).contiguous()
    if isinstance(tree, dict):
        return {key: _copy_tree(value, device) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_copy_tree(value, device) for value in tree)
    if isinstance(tree, list):
        return [_copy_tree(value, device) for value in tree]
    return tree


def _tree_bytes(tree: Any) -> int:
    # Charge tensor bytes, Python tree/key/scalar overhead and Tensor headers.
    # max_entries separately bounds allocator/OrderedDict overhead. This is a
    # payload accounting limit, not a claim about CUDA allocator reservation.
    size = sys.getsizeof(tree)
    if isinstance(tree, torch.Tensor):
        return size + tree.numel() * tree.element_size()
    if isinstance(tree, dict):
        return size + sum(sys.getsizeof(key) + _tree_bytes(value) for key, value in tree.items())
    if isinstance(tree, (tuple, list)):
        return size + sum(_tree_bytes(value) for value in tree)
    return size


def _structure(tree: Any) -> Any:
    if isinstance(tree, torch.Tensor):
        return ("tensor", tuple(tree.shape), tree.dtype, tree.device, tree._version)
    if isinstance(tree, dict):
        return ("dict", tuple((key, _structure(value)) for key, value in tree.items()))
    if isinstance(tree, (tuple, list)):
        return (type(tree).__name__, tuple(_structure(value) for value in tree))
    return (type(tree).__name__, tree)


@dataclass
class _Entry:
    payload: Any
    size: int
    structure: Any


class ConditionFeatureCache:
    """Single-owner byte/entry-bounded LRU. No disk or global mutable state.

    CPU and CUDA storage use the same interface. An adapter may compose two
    instances for tiers; the first implementation deliberately avoids hidden
    background transfers. get() always returns new tensors, including on GPU.
    """

    def __init__(self, max_bytes: int, storage_device: str | torch.device = "cpu", max_entries: int = 4096) -> None:
        if type(max_bytes) is not int or max_bytes < 0 or type(max_entries) is not int or max_entries <= 0:
            raise ValueError("Cache byte limit must be nonnegative and entry limit positive")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.storage_device = torch.device(storage_device)
        if self.storage_device.type not in ("cpu", "cuda"):
            raise ValueError("Condition cache storage must be CPU or CUDA")
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes = 0
        self._closed = False
        self._counters = dict(hits=0, misses=0, puts=0, evictions=0, oversized=0, corruptions=0)
        self._counters.update(read_tensor_bytes=0, write_tensor_bytes=0, d2h_bytes=0, h2d_bytes=0)
        self._read_seconds = self._write_seconds = 0.0

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Condition cache is closed")

    def get(self, key: str, device: str | torch.device | None = None) -> Any | None:
        self._require_open()
        started = time.monotonic()
        entry = self._entries.get(key)
        if entry is None:
            self._counters["misses"] += 1
            return None
        try:
            if _structure(entry.payload) != entry.structure:
                raise ValueError("Stored condition tree was modified")
            target = torch.device(device) if device is not None else None
            result = _copy_tree(entry.payload, target)
        except (TypeError, ValueError, AttributeError):
            self.discard(key)
            self._counters["corruptions"] += 1
            self._counters["misses"] += 1
            return None
        self._entries.move_to_end(key)
        self._counters["hits"] += 1
        for tensor in _walk_tensors(entry.payload):
            size = tensor.numel() * tensor.element_size()
            self._counters["read_tensor_bytes"] += size
            if tensor.device.type == "cpu" and target is not None and target.type == "cuda":
                self._counters["h2d_bytes"] += size
            if tensor.device.type == "cuda" and target is not None and target.type == "cpu":
                self._counters["d2h_bytes"] += size
        self._read_seconds += time.monotonic() - started
        return result

    def contains(self, key: str) -> bool:
        """Probe without a copy, for collective agreement before any
        transfer."""
        self._require_open()
        entry = self._entries.get(key)
        if entry is None:
            return False
        try:
            valid = _structure(entry.payload) == entry.structure
        except (TypeError, ValueError, AttributeError):
            valid = False
        if not valid:
            self.discard(key)
            self._counters["corruptions"] += 1
        return valid

    def put(self, key: str, payload: Any) -> bool:
        self._require_open()
        if not isinstance(key, str) or not key:
            raise ValueError("Cache keys must be nonempty strings")
        tensors = list(_walk_tensors(payload))
        if not tensors:
            raise ValueError("A frozen-condition payload must contain at least one tensor")
        if any(t.requires_grad or t.grad_fn is not None or t.layout != torch.strided for t in tensors):
            raise ValueError("Cache only accepts dense, detached frozen features")
        size = _tree_bytes(payload) + sys.getsizeof(key)
        self.discard(key)
        if size > self.max_bytes:
            self._counters["oversized"] += 1
            return False
        started = time.monotonic()
        while self._entries and (self._bytes + size > self.max_bytes or len(self._entries) >= self.max_entries):
            _, old = self._entries.popitem(last=False)
            self._bytes -= old.size
            self._counters["evictions"] += 1
        stored = _copy_tree(payload, self.storage_device)
        self._entries[key] = _Entry(stored, size, _structure(stored))
        self._bytes += size
        self._counters["puts"] += 1
        for tensor in tensors:
            count = tensor.numel() * tensor.element_size()
            self._counters["write_tensor_bytes"] += count
            if tensor.device.type == "cuda" and self.storage_device.type == "cpu":
                self._counters["d2h_bytes"] += count
            if tensor.device.type == "cpu" and self.storage_device.type == "cuda":
                self._counters["h2d_bytes"] += count
        self._write_seconds += time.monotonic() - started
        return True

    def discard(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes -= entry.size

    def clear(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def close(self) -> None:
        self.clear()
        self._closed = True

    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def stats(self) -> dict[str, Any]:
        return {
            **self._counters,
            "entries": len(self._entries),
            "bytes": self._bytes,
            "max_bytes": self.max_bytes,
            "max_entries": self.max_entries,
            "storage_device": str(self.storage_device),
            "closed": self._closed,
            "read_seconds": self._read_seconds,
            "write_seconds": self._write_seconds,
        }


def all_ranks_hit(local_hit: bool, process_group: Any = None) -> bool:
    """All ranks take the same branch before invoking sharded encoders.

    Prefer the existing Gloo group to avoid a CUDA scalar round trip. The
    explicit NCCL fallback also works in standalone FSDP runners. This tiny
    control-plane synchronization is required for collective call ordering.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return bool(local_hit)
    group = process_group if process_group is not None else dist.group.WORLD
    if dist.get_world_size(group) == 1:
        return bool(local_hit)
    device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl" else "cpu"
    flag = torch.tensor([int(local_hit)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


# ---------------------------------------------------------------------------
# Node-shared store
# ---------------------------------------------------------------------------

_MAGIC = b"RCC1"
_TRAILER = b"RCCE"
_ALIGN = 64
_OWNERS = "owners"
_LOCK = ".lock"
_ROOT_LOCK = ".lifecycle.lock"
_SUFFIX = ".rcc"


def _align(value: int) -> int:
    return (value + _ALIGN - 1) // _ALIGN * _ALIGN


def _encode_tree(tree: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(tree, torch.Tensor):
        tensors.append(tree)
        return {"t": len(tensors) - 1}
    if tree is None:
        return {"n": 0}
    if isinstance(tree, dict):
        return {"d": [[key, _encode_tree(value, tensors)] for key, value in tree.items()]}
    if isinstance(tree, tuple):
        return {"u": [_encode_tree(value, tensors) for value in tree]}
    if isinstance(tree, list):
        return {"l": [_encode_tree(value, tensors) for value in tree]}
    if type(tree) in (str, int, float, bool):
        return {"s": tree}
    raise TypeError(f"Unsupported condition payload type: {type(tree).__name__}")


def _decode_tree(node: Any, tensors: list[torch.Tensor]) -> Any:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError("Malformed condition tree")
    ((kind, value),) = node.items()
    if kind == "t":
        return tensors[value]
    if kind == "n":
        return None
    if kind == "d":
        return {str(key): _decode_tree(item, tensors) for key, item in value}
    if kind == "u":
        return tuple(_decode_tree(item, tensors) for item in value)
    if kind == "l":
        return [_decode_tree(item, tensors) for item in value]
    if kind == "s":
        return value
    raise ValueError(f"Unknown condition tree node {kind!r}")


def _map_tree(tree: Any, fn: Any) -> Any:
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    if isinstance(tree, dict):
        return {key: _map_tree(value, fn) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_map_tree(value, fn) for value in tree)
    if isinstance(tree, list):
        return [_map_tree(value, fn) for value in tree]
    return tree


def _dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown stored dtype {name!r}")
    return dtype


def _raw_bytes(tensor: torch.Tensor) -> memoryview:
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return memoryview(b"")
    return memoryview(flat.view(torch.uint8).numpy())


def _nbytes(tensors: Any) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _NodeLock:
    """Cross-process exclusive file lock on POSIX; thread-only elsewhere."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._thread_lock = threading.Lock()
        self._file: Any = None

    def __enter__(self) -> _NodeLock:
        self._thread_lock.acquire()
        try:
            import fcntl
        except ImportError:
            return self
        try:
            self._file = open(self._path, "a+b")
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        except BaseException:
            if self._file is not None:
                self._file.close()
                self._file = None
            self._thread_lock.release()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._file is not None:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
                self._file.close()
                self._file = None
        finally:
            self._thread_lock.release()


@dataclass
class _PendingWrite:
    key: str
    tree: Any
    tensors: list[torch.Tensor]
    event: Any
    size: int
    generation: int


class SharedConditionStore:
    """Node-shared, byte/entry-bounded LRU of frozen condition trees.

    Every data-parallel rank of one run on a node opens the same run
    directory. Each entry is written once; any rank of the node reads it back,
    so a sample reassigned to another rank in a later epoch still hits.

    * ``contains(key)`` loads and validates the entry into a private host
      buffer (pinned when CUDA is available) before reporting a hit. An
      eviction after the collective hit agreement cannot invalidate the hit;
      ``get`` then only issues the asynchronous host-to-device copy.
    * ``put`` never waits for the GPU. The device-to-host copy is enqueued on
      the current stream; a background thread waits for it, serializes the
      entry to a temporary file and atomically renames it. Writes that do not
      fit the staging budget are dropped and counted, never queued unbounded.
    * ``max_bytes``/``max_entries`` bound the whole run directory on the node
      and are enforced under a cross-process lock by evicting the least
      recently used entries. Reads refresh an entry's recency.
    * Missing, truncated or mismatching files are misses; corrupted files are
      deleted. The last live owner removes the run directory on ``close``;
      directories left behind by crashed runs are reaped on the next start.
    """

    def __init__(
        self,
        max_bytes: int,
        root: str | os.PathLike,
        *,
        max_entries: int = 4096,
        staging_bytes: int = 2 * 1024**3,
        run_id: str | None = None,
    ) -> None:
        if type(max_bytes) is not int or max_bytes < 0 or type(max_entries) is not int or max_entries <= 0:
            raise ValueError("Cache byte limit must be nonnegative and entry limit positive")
        if type(staging_bytes) is not int or staging_bytes <= 0:
            raise ValueError("Condition cache staging budget must be positive")
        self.max_bytes, self.max_entries, self.staging_bytes = max_bytes, max_entries, staging_bytes
        self.storage_device = torch.device("cpu")
        self.root = Path(root)
        # This inode outlives run directories, including their per-run locks.
        self._lifecycle_lock = _NodeLock(self.root / _ROOT_LOCK)
        # One marker per store instance: "<host>-<pid>-<token>".
        self._owner = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._run_dir: Path | None = None
        self._lock: _NodeLock | None = None
        self._staged: tuple[str, Any] | None = None
        self._generation = 0
        self._pending_bytes = 0
        self._pending_lock = threading.Lock()
        self._queue: queue.Queue[_PendingWrite | None] = queue.Queue()
        self._writer: threading.Thread | None = None
        self._closed = False
        self._counters = dict(hits=0, misses=0, puts=0, evictions=0, oversized=0, corruptions=0)
        self._counters.update(read_tensor_bytes=0, write_tensor_bytes=0, d2h_bytes=0, h2d_bytes=0)
        self._counters.update(committed=0, deduplicated=0, write_dropped=0, write_errors=0)
        self._read_seconds = self._write_seconds = 0.0
        self._dir_bytes = self._dir_entries = 0
        if run_id is not None:
            self.bind(run_id=run_id)

    # -- lifecycle -------------------------------------------------------------

    def bind(self, group: Any = None, run_id: str | None = None) -> None:
        """Join the run directory shared by every rank of ``group``.

        Rank 0 of the group chooses the run id; each node uses that name in its
        own node-local root. Idempotent. Without a process group the store is
        private to this process.
        """
        if self._run_dir is not None:
            return
        if self._closed:
            raise RuntimeError("Condition cache is closed")
        if run_id is None:
            import torch.distributed as dist

            holder = [uuid.uuid4().hex]
            if dist.is_available() and dist.is_initialized():
                group = group if group is not None else dist.group.WORLD
                if dist.get_world_size(group) > 1:
                    dist.broadcast_object_list(holder, src=dist.get_global_rank(group, 0), group=group)
            run_id = holder[0]
        if not isinstance(run_id, str) or not run_id or any(c in run_id for c in "/\\.") or run_id == _OWNERS:
            raise ValueError(f"Invalid condition cache run id {run_id!r}")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lifecycle_lock:
            self._reap_stale_runs(keep=run_id)
            run_dir = self.root / run_id
            (run_dir / _OWNERS).mkdir(parents=True, exist_ok=True)
            (run_dir / _OWNERS / self._owner).touch()
            self._run_dir, self._lock = run_dir, _NodeLock(run_dir / _LOCK)
            self._writer = threading.Thread(target=self._write_loop, name="relax-condition-cache-writer", daemon=True)
            self._writer.start()

    def _reap_stale_runs(self, keep: str) -> None:
        """Caller holds the root lifecycle lock through owner registration."""
        host = socket.gethostname() + "-"
        for run_dir in list(self.root.iterdir()):
            owners = run_dir / _OWNERS
            if run_dir.name == keep or not owners.is_dir():
                continue
            try:
                markers = [marker.name for marker in owners.iterdir()]
            except FileNotFoundError:
                continue
            # Only this host's owners can be checked. A node-local root never
            # holds other hosts' runs, so a foreign marker means "keep".
            pids = [name[len(host) :].split("-", 1)[0] if name.startswith(host) else "" for name in markers]
            stale = all(pid.isdigit() and not _pid_alive(int(pid)) for pid in pids)
            if stale:
                shutil.rmtree(run_dir, ignore_errors=True)
                logger.info("Reaped stale frozen-condition cache directory %s", run_dir)

    def _require_open(self) -> Path:
        if self._closed:
            raise RuntimeError("Condition cache is closed")
        if self._run_dir is None:
            self.bind()
        return self._run_dir

    def close(self) -> None:
        """Release staging memory and pending writes; the last owner on the
        node removes the run directory."""
        if self._closed:
            return
        self._closed = True
        self._staged = None
        self._generation += 1  # queued writes become stale and are dropped
        if self._writer is not None:
            self._queue.put(None)
            self._writer.join(timeout=60)
        run_dir = self._run_dir
        if run_dir is None:
            return
        # Never acquire the lifecycle lock while holding a run lock. Writer
        # shutdown is requested above; writers take only run locks.
        with self._lifecycle_lock:
            if not run_dir.is_dir():
                return
            with self._lock:
                owners = run_dir / _OWNERS
                (owners / self._owner).unlink(missing_ok=True)
                remaining = list(owners.iterdir()) if owners.is_dir() else []
                if not remaining:
                    for path in list(run_dir.iterdir()):
                        if path.name == _LOCK:
                            continue
                        if path.is_dir():
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            path.unlink(missing_ok=True)
            if not remaining:
                shutil.rmtree(run_dir, ignore_errors=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return

    # -- read path -------------------------------------------------------------

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not key or any(c in key for c in "/\\") or key.startswith("."):
            raise ValueError("Cache keys must be nonempty file-safe strings")
        return self._require_open() / f"{key}{_SUFFIX}"

    def _load(self, key: str, path: Path) -> Any | None:
        pin = torch.cuda.is_available()
        try:
            with open(path, "rb") as source:
                size = os.fstat(source.fileno()).st_size
                head = source.read(len(_MAGIC) + 8)
                if len(head) != len(_MAGIC) + 8 or head[: len(_MAGIC)] != _MAGIC:
                    raise ValueError("bad magic")
                header_len = int.from_bytes(head[len(_MAGIC) :], "little")
                if header_len <= 0 or header_len > size:
                    raise ValueError("bad header length")
                header = json.loads(source.read(header_len).decode("utf-8"))
                data_offset = _align(len(_MAGIC) + 8 + header_len)
                payload = int(header["payload_bytes"])
                if header.get("key") != key or size != data_offset + payload + len(_TRAILER):
                    raise ValueError("key or length mismatch")
                source.seek(data_offset)
                buffer = torch.empty(payload, dtype=torch.uint8, pin_memory=pin)
                if payload and source.readinto(memoryview(buffer.numpy())) != payload:
                    raise ValueError("truncated payload")
                if source.read() != _TRAILER:
                    raise ValueError("bad trailer")
            tensors = []
            for meta in header["tensors"]:
                offset, nbytes = int(meta["offset"]), int(meta["nbytes"])
                dtype, shape = _dtype(meta["dtype"]), [int(v) for v in meta["shape"]]
                if offset < 0 or offset % _ALIGN or offset + nbytes > payload:
                    raise ValueError("tensor outside payload")
                if math.prod(shape) * torch.empty((), dtype=dtype).element_size() != nbytes:
                    raise ValueError("tensor size mismatch")
                tensors.append(buffer[offset : offset + nbytes].view(dtype).reshape(shape))
            tree = _decode_tree(header["tree"], tensors)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, UnicodeDecodeError) as exc:
            logger.warning("Discarding corrupted frozen-condition entry %s: %s", path.name, exc)
            self._counters["corruptions"] += 1
            path.unlink(missing_ok=True)
            return None
        try:
            os.utime(path)  # LRU recency shared by every rank on the node
        except OSError:
            pass
        return tree

    def contains(self, key: str) -> bool:
        """Stage the entry in host memory; a staged entry survives eviction."""
        path = self._path(key)
        if self._staged is not None and self._staged[0] == key:
            return True
        self._staged = None
        started = time.monotonic()
        tree = self._load(key, path)
        self._read_seconds += time.monotonic() - started
        if tree is None:
            return False
        self._staged = (key, tree)
        return True

    def get(self, key: str, device: str | torch.device | None = None) -> Any | None:
        if not self.contains(key):
            self._counters["misses"] += 1
            return None
        started = time.monotonic()
        tree = self._staged[1]
        self._staged = None
        target = torch.device(device) if device is not None else torch.device("cpu")
        size = _nbytes(_walk_tensors(tree))
        if target.type == "cpu":
            result = _copy_tree(tree, target)  # private copy, staging buffer is dropped
        else:
            # The caching host allocator keeps each pinned staging buffer
            # alive until its asynchronous copy has completed.
            result = _map_tree(tree, lambda tensor: tensor.to(device=target, non_blocking=True))
            self._counters["h2d_bytes"] += size
        self._counters["hits"] += 1
        self._counters["read_tensor_bytes"] += size
        self._read_seconds += time.monotonic() - started
        return result

    # -- write path ------------------------------------------------------------

    def put(self, key: str, payload: Any) -> bool:
        path = self._path(key)
        self._staged = None
        tensors = list(_walk_tensors(payload))
        if not tensors:
            raise ValueError("A frozen-condition payload must contain at least one tensor")
        if any(t.requires_grad or t.grad_fn is not None or t.layout != torch.strided for t in tensors):
            raise ValueError("Cache only accepts dense, detached frozen features")
        if path.exists():
            self._counters["deduplicated"] += 1
            return True
        size = sum(_align(t.numel() * t.element_size()) for t in tensors)
        if size > self.max_bytes:
            self._counters["oversized"] += 1
            return False
        with self._pending_lock:
            if self._pending_bytes + size > self.staging_bytes:
                self._counters["write_dropped"] += 1
                return False
            self._pending_bytes += size
        started = time.monotonic()
        try:
            host: list[torch.Tensor] = []
            event = None
            for tensor in tensors:
                if tensor.device.type == "cuda":
                    staging = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                    staging.copy_(tensor.detach(), non_blocking=True)
                    host.append(staging)
                    self._counters["d2h_bytes"] += tensor.numel() * tensor.element_size()
                else:
                    host.append(tensor.detach().contiguous().clone())
            if any(t.device.type == "cuda" for t in tensors):
                event = torch.cuda.Event()
                event.record()
            tree = _encode_tree(payload, [])
        except BaseException:
            with self._pending_lock:
                self._pending_bytes -= size
            raise
        self._queue.put(_PendingWrite(key, tree, host, event, size, self._generation))
        self._counters["puts"] += 1
        self._counters["write_tensor_bytes"] += _nbytes(tensors)
        self._write_seconds += time.monotonic() - started
        return True

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                if item.generation == self._generation and not self._closed:
                    started = time.monotonic()
                    if item.event is not None:
                        item.event.synchronize()  # background thread only
                    self._commit(item)
                    self._write_seconds += time.monotonic() - started
            except Exception as exc:  # a cache write must never fail training
                self._counters["write_errors"] += 1
                logger.warning("Frozen-condition cache write failed: %s", exc)
            finally:
                item.tensors = []
                with self._pending_lock:
                    self._pending_bytes -= item.size

    def _commit(self, item: _PendingWrite) -> None:
        run_dir = self._run_dir
        metas, offset = [], 0
        for tensor in item.tensors:
            nbytes = tensor.numel() * tensor.element_size()
            dtype = str(tensor.dtype).removeprefix("torch.")
            metas.append({"dtype": dtype, "shape": list(tensor.shape), "offset": offset, "nbytes": nbytes})
            offset += _align(nbytes)
        header = json.dumps(
            {"key": item.key, "tree": item.tree, "tensors": metas, "payload_bytes": offset}, separators=(",", ":")
        ).encode("utf-8")
        tmp = run_dir / f".tmp-{self._owner}-{uuid.uuid4().hex}"
        try:
            with open(tmp, "wb") as sink:
                sink.write(_MAGIC + len(header).to_bytes(8, "little") + header)
                sink.write(b"\0" * (_align(sink.tell()) - sink.tell()))
                for tensor, meta in zip(item.tensors, metas):
                    sink.write(_raw_bytes(tensor))
                    sink.write(b"\0" * (_align(meta["nbytes"]) - meta["nbytes"]))
                sink.write(_TRAILER)
            final = run_dir / f"{item.key}{_SUFFIX}"
            with self._lock:
                if item.generation != self._generation or self._closed:
                    return
                if final.exists():
                    self._counters["deduplicated"] += 1
                    return
                entries = self._scan()
                used = sum(size for _, size, _ in entries)
                file_size = tmp.stat().st_size
                while entries and (used + file_size > self.max_bytes or len(entries) >= self.max_entries):
                    _, size, victim = entries.pop(0)
                    victim.unlink(missing_ok=True)
                    used -= size
                    self._counters["evictions"] += 1
                if used + file_size > self.max_bytes:
                    self._counters["oversized"] += 1
                    return
                os.replace(tmp, final)
                self._dir_bytes, self._dir_entries = used + file_size, len(entries) + 1
                self._counters["committed"] += 1
        finally:
            tmp.unlink(missing_ok=True)

    def _scan(self) -> list[tuple[int, int, Path]]:
        """Committed entries, least recently used first."""
        entries = []
        for entry in os.scandir(self._run_dir):
            if entry.name.endswith(_SUFFIX) and not entry.name.startswith("."):
                try:
                    stat = entry.stat()
                except FileNotFoundError:
                    continue
                entries.append((stat.st_mtime_ns, stat.st_size, Path(entry.path)))
        entries.sort(key=lambda item: (item[0], item[2].name))
        return entries

    # -- maintenance -----------------------------------------------------------

    def discard(self, key: str) -> None:
        if self._staged is not None and self._staged[0] == key:
            self._staged = None
        self._path(key).unlink(missing_ok=True)

    def clear(self) -> None:
        """Drop this run's entries on the node and this rank's queued writes.

        Invalidation is collective (every rank observes the same frozen-weight
        or eligibility change), so each rank also drops its own late writes.
        """
        self._require_open()
        self._staged = None
        with self._lock:
            self._generation += 1
            for _, _, path in self._scan():
                path.unlink(missing_ok=True)
            self._dir_bytes = self._dir_entries = 0

    def keys(self) -> tuple[str, ...]:
        self._require_open()
        return tuple(path.name[: -len(_SUFFIX)] for _, _, path in self._scan())

    def flush(self, timeout: float = 60.0) -> bool:
        """Wait until this rank's accepted writes are committed or dropped."""
        deadline = time.monotonic() + timeout
        while self._pending_bytes and time.monotonic() < deadline:
            time.sleep(0.005)
        return not self._pending_bytes

    def stats(self) -> dict[str, Any]:
        if self._run_dir is not None and not self._closed:
            entries = self._scan()
            self._dir_bytes, self._dir_entries = sum(size for _, size, _ in entries), len(entries)
        return {
            **self._counters,
            "entries": self._dir_entries,
            "bytes": self._dir_bytes,
            "max_bytes": self.max_bytes,
            "max_entries": self.max_entries,
            "pending_bytes": self._pending_bytes,
            "staging_bytes": self.staging_bytes,
            "storage_device": "shm",
            "run_dir": str(self._run_dir) if self._run_dir is not None else None,
            "closed": self._closed,
            "read_seconds": self._read_seconds,
            "write_seconds": self._write_seconds,
        }


def build_condition_cache(max_bytes: int, storage: str | None = None, root: str | None = None) -> Any:
    """Cache selected by ``RELAX_CONDITION_CACHE_*``; ``None`` when disabled.

    ``shm`` (default) shares entries between the ranks of one node. It is the
    storage that keeps collective hits once the sampler moves a sample to
    another rank. ``cpu``/``cuda`` keep a private per-process LRU, which only
    hits reliably in single-rank runs.
    """
    from relax.utils.env import Envs

    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("RELAX_CONDITION_CACHE_BYTES cannot be negative")
    if not max_bytes:
        return None
    storage = storage or Envs.RELAX_CONDITION_CACHE_DEVICE
    entries = Envs.RELAX_CONDITION_CACHE_MAX_ENTRIES
    if storage == "shm":
        return SharedConditionStore(
            max_bytes,
            root or Envs.RELAX_CONDITION_CACHE_DIR,
            max_entries=entries,
            staging_bytes=Envs.RELAX_CONDITION_CACHE_STAGING_BYTES,
        )
    return ConditionFeatureCache(max_bytes, storage_device=storage, max_entries=entries)


def bind_condition_cache(cache: Any, group: Any) -> None:
    """Join the node-shared run directory; a no-op for private caches."""
    bind = getattr(cache, "bind", None)
    if bind is not None:
        bind(group)


def model_files_identity(model_path: str | os.PathLike, small_file_bytes: int = 64 * 1024**2) -> str:
    """Identity of a local model directory without rereading the weights.

    Small files (configs, tokenizers, processors) are hashed by content and
    large weight shards by relative path, size and mtime. Entries never outlive
    one run, in which the frozen weights are loaded once, so this only has to
    tell model directories apart. ``RELAX_CONDITION_CACHE_IDENTITY`` binds an
    externally verified content SHA instead.
    """
    root = Path(model_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Frozen encoder identity requires a local model directory: {model_path}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts)
    if not files:
        raise FileNotFoundError(f"No model files under {model_path}")
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + str(stat.st_size).encode() + b"\0")
        if stat.st_size <= small_file_bytes:
            digest.update(b"c" + path.read_bytes())
        else:
            digest.update(b"m" + str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def encoder_identity(model_path: str | os.PathLike) -> str:
    """Verified ``RELAX_CONDITION_CACHE_IDENTITY`` if set, else the cheap
    directory identity."""
    from relax.utils.env import Envs

    identity = Envs.RELAX_CONDITION_CACHE_IDENTITY
    if identity:
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity.lower()):
            raise ValueError("RELAX_CONDITION_CACHE_IDENTITY must be a verified asset-manifest SHA256")
        return identity.lower()
    return model_files_identity(model_path)
