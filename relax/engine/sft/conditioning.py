# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Model-independent frozen conditioning reuse and cache ownership.

The adapter calls resolve inside its normal FSDP root forward. A unanimous
decision on the existing Gloo group keeps sharded forward collectives aligned.
Storage is node-local and run-scoped; it never contains noise or optimizer
data.
"""

from __future__ import annotations

import os
import socket
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from relax.models.generative_sft import ConditioningSpec
from relax.utils.conditioning_cache import ConditioningCache, build_key, fingerprint_file
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _flag(name: str) -> bool:
    value = os.environ.get(name, "0").lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value in {"1", "true"}


def conditioning_cache_enabled() -> bool:
    return _flag("RELAX_CONDITION_CACHE")


def model_fingerprint(model_path: str | Path) -> str:
    """Fingerprint actual checkpoint/tokenizer contents once, outside
    training."""
    root = Path(model_path)
    selected = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and (
            path.suffix in {".safetensors", ".json"}
            or path.name in {"merges.txt", "vocab.txt", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model"}
        )
    )
    if not any(path.suffix == ".safetensors" for path in selected):
        raise ValueError("No safetensors checkpoint found for conditioning fingerprint")
    return build_key({path.relative_to(root).as_posix(): fingerprint_file(path) for path in selected})


def unanimous_ready(ready: bool, group: Any) -> bool:
    """This collective must be called by every rank before a cached skip."""
    if not dist.is_initialized():
        return ready
    flag = torch.tensor(int(ready), dtype=torch.int32, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _tensor_bytes(features: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in features.values())


class ConditioningRuntime:
    """Resolve frozen conditions with bounded storage and in-flight copies."""

    def __init__(
        self,
        *,
        model: Any,
        model_path: str,
        device: torch.device,
        group: Any,
        rank: int,
        namespace: str,
        local_batch_size: int,
    ) -> None:
        self.enabled = conditioning_cache_enabled()
        self.device, self.group = device, group
        self._closed = False
        self._pending_copies: list[tuple[Any, Any, int]] = []
        self._pending_copy_limit = max(1, local_batch_size)
        self._step: defaultdict[str, int] = defaultdict(int)
        self._model = model
        self._conditioning_snapshot: dict[str, Any] | None = None
        self._conditioning_semantic_snapshot: str | None = None
        self._cache_guard_disabled = bool(getattr(model, "conditioning_reuse_disabled", False))
        self._cache_guard_reason = getattr(model, "conditioning_reuse_disabled_reason", None)
        self.cache: ConditioningCache | None = None
        self._prefetch: Any = None
        self.prefetch_depth = int(os.environ.get("RELAX_CONDITION_PREFETCH_DEPTH", "0"))
        prefetch_bytes = int(os.environ.get("RELAX_CONDITION_PREFETCH_MAX_BYTES", str(512 * 1024**2)))
        if self.prefetch_depth < 0 or (self.prefetch_depth and prefetch_bytes <= 0):
            raise ValueError("Conditioning prefetch requires nonnegative depth and positive byte capacity")
        self.fingerprint = ""
        if self.enabled:
            # Fingerprinting failures are broadcast rather than stranding peers.
            fingerprint: list[Any] = [None]
            if rank == 0:
                try:
                    fingerprint[0] = {"sha256": model_fingerprint(model_path)}
                except Exception as error:
                    fingerprint[0] = {"error": f"{type(error).__name__}: {error}"}
            if dist.is_initialized():
                dist.broadcast_object_list(fingerprint, src=0, group=group)
            if "error" in fingerprint[0]:
                raise ValueError(fingerprint[0]["error"])
            self.fingerprint = fingerprint[0]["sha256"]
            maximum = int(os.environ.get("RELAX_CONDITION_CACHE_MAX_BYTES", str(8 * 1024**3)))
            if maximum <= 0:
                raise ValueError("Enabled conditioning cache requires a positive byte limit")
            directory = os.environ.get("RELAX_CONDITION_CACHE_DIR") or "/dev/shm/relax-conditioning"
            self.cache = ConditioningCache(maximum, directory, f"{namespace}:{socket.gethostname()}")
        try:
            if self.enabled and self.prefetch_depth:
                from relax.engine.sft.conditioning_prefetch import ConditioningPrefetch

                self._prefetch = ConditioningPrefetch(self.cache, depth=self.prefetch_depth, max_bytes=prefetch_bytes)
            self._model.conditioning_cache = self
        except BaseException:
            # Initialization has not returned, so no prefetch has been submitted.
            for resource in (self._prefetch, self.cache):
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as cleanup_error:
                        logger.warning("Conditioning initialization cleanup failed: %s", cleanup_error)
            self._closed = True
            raise

    def begin_step(self) -> None:
        self._step.clear()

    def before_forward(self) -> None:
        """Fail closed on a conditioning-master change outside FSDP forward.

        The adapter optionally declares ``frozen``, named tensor metadata and
        CPU-only ``semantic`` configuration. Every rank enters unanimous_ready:
        one poisoned shard owner makes the whole group compute.
        """
        if self._cache_guard_disabled:
            return
        inspect_state = getattr(self._model, "conditioning_state", None)
        if inspect_state is None:
            return
        reason = None
        try:
            state = inspect_state()
            parameters = state["parameters"]
            # Canonical JSON validates the CPU-only declaration and snapshots
            # nested mutable dictionaries without retaining their references.
            semantic = build_key({"semantic": state["semantic"]}) if "semantic" in state else None
            if not state["frozen"]:
                name = next((name for name, value in parameters.items() if value["requires_grad"]), "conditioning")
                reason = f"{name}.requires_grad is true"
            elif self._conditioning_snapshot is not None and semantic != self._conditioning_semantic_snapshot:
                reason = "conditioning semantic configuration changed"
            elif self._conditioning_snapshot is not None:
                for name in sorted(set(parameters) | set(self._conditioning_snapshot)):
                    if name not in parameters or name not in self._conditioning_snapshot:
                        reason = f"{name}.presence changed"
                        break
                    before, current = self._conditioning_snapshot[name], parameters[name]
                    for field in ("identity", "version", "local_version", "requires_grad", "dtype", "shape"):
                        if before[field] != current[field]:
                            reason = f"{name}.{field} changed"
                            break
                    if reason is None and before.get("stride") != current.get("stride"):
                        reason = f"{name}.stride changed"
                    if reason is not None:
                        break
            if reason is None:
                self._conditioning_snapshot = parameters
                self._conditioning_semantic_snapshot = semantic
        except Exception as error:
            reason = f"conditioning state inspection failed: {type(error).__name__}: {error}"
        if reason is not None:
            self._cache_guard_disabled = True
            self._cache_guard_reason = reason
            self._model.conditioning_reuse_disabled = True
            self._model.conditioning_reuse_disabled_reason = reason
            logger.warning(
                "Disabling conditioning reuse for this model instance: %s; start a new task to reenable", reason
            )

    @property
    def prefetch_enabled(self) -> bool:
        return self._prefetch is not None and not self._closed

    def _conditioning_key(self, spec: ConditioningSpec) -> str:
        return build_key({"weights": self.fingerprint, "schema": spec.name, "dependencies": spec.dependencies})

    def prefetch_many(self, specs: list[ConditioningSpec]) -> int:
        """Replace the bounded future window, without model work or
        collectives.

        The caller prepares inputs on the original RNG stream/order. Resolve
        rebuilds the actual key and still checks the frozen-parameter guard;
        this method never treats a sample ID as a cache identity.
        """
        if not self.prefetch_enabled:
            return 0
        if self._cache_guard_disabled:
            self._prefetch.clear()
            return 0
        keys = [self._conditioning_key(spec) for spec in specs if spec.cacheable]
        return self._prefetch.set_window(keys)

    def resolve(
        self,
        spec: ConditioningSpec,
        compute: Callable[[], dict[str, torch.Tensor]],
        restore: Callable[[dict[str, torch.Tensor]], dict[str, Any]],
    ) -> dict[str, Any]:
        self._release_copies()
        features = restored = None
        local_hit = False
        key = None
        reusable = self.enabled and spec.cacheable and not self._cache_guard_disabled
        if self._prefetch is not None and not reusable:
            self._prefetch.clear()
        if reusable:
            try:
                key = self._conditioning_key(spec)
                lease = None
                if self._prefetch is not None:
                    lease = self._prefetch.take(key)
                    self._step["prefetch_hit"] += int(lease is not None)
                if lease is None:
                    lease = self.cache.lookup(key)
                if lease is not None:
                    features = self._copy_features(lease)
                    if features is None:
                        key = None  # Admission bypass: retain the entry without rewriting it.
                    else:
                        restored = restore(features)  # validate before the unanimous decision
                        local_hit = True
            except Exception as error:
                features = restored = None
                self._step["cache_errors"] += 1
                if key is not None and self.cache is not None:
                    self.cache.invalidate(key)
                logger.warning("Conditioning read failed; using online encoding: %s", error)
        self._step["local_hit"] += int(local_hit)
        ready = local_hit
        if self.enabled:
            self._step["coordinate_calls"] += 1
            ready = unanimous_ready(ready, self.group)
        if ready:
            self._step["effective_hit"] += 1
            self._step["cache_read_bytes"] += _tensor_bytes(features)
            return restored
        self._step["forced_recompute"] += int(local_hit)
        self._step["miss"] += int(not local_hit)
        features = restored = None
        features = compute()
        self._step["encoding_calls"] += 1
        self._step["condition_payload_bytes"] += _tensor_bytes(features)
        if reusable and key is not None and not local_hit:
            try:
                if self.cache.put(key, features):
                    self._step["cache_write_bytes"] += _tensor_bytes(features)
            except Exception as error:
                self._step["cache_errors"] += 1
                logger.warning("Conditioning write failed; retaining online result: %s", error)
        return restore(features)

    def _copy_features(self, lease: Any) -> dict[str, torch.Tensor] | None:
        if self.device.type != "cuda":
            with lease:
                return lease.copy_to(self.device)[0]
        try:
            nbytes = _tensor_bytes(lease.data)
            if (
                len(self._pending_copies) >= self._pending_copy_limit
                or any(event is None for event, _, _ in self._pending_copies)
                or sum(size for _, _, size in self._pending_copies) + nbytes > self.cache.max_bytes
            ):
                lease.close()
                return None
            stream = torch.cuda.current_stream(self.device)
            event = torch.cuda.Event()
        except BaseException:
            lease.close()  # No copy has been submitted yet.
            raise
        index = len(self._pending_copies)
        # Own the sources before copy_to can enqueue even a partial transfer.
        # An unrecorded event must never authorize releasing those sources.
        self._pending_copies.append((None, lease, nbytes))
        try:
            features, copied_bytes = lease.copy_to(self.device)
            self._step["condition_h2d_bytes"] += copied_bytes
            return features
        finally:
            event.record(stream)
            self._pending_copies[index] = (event, lease, nbytes)

    def _release_copies(self, *, synchronized: bool = False) -> None:
        while self._pending_copies:
            event, lease, nbytes = self._pending_copies[0]
            if synchronized:
                lease.close()
            else:
                try:
                    if event is None or not event.query():
                        break
                    lease.close()
                except Exception as error:
                    self._pending_copies[0] = (None, lease, nbytes)
                    logger.warning("Conditioning copy cleanup deferred until step end: %s", error)
                    break
            self._pending_copies.pop(0)

    def end_step(self) -> None:
        """Finish copies before returning their leases at the step boundary."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._release_copies(synchronized=True)

    def stats(self) -> dict[str, Any]:
        """Current-step counters and ownership state; no I/O or device
        queries."""
        return {
            **self._step,
            "cache_guard_disabled": self._cache_guard_disabled,
            "cache_guard_reason": self._cache_guard_reason,
            "pending_copy_count": len(self._pending_copies),
            "pending_copy_bytes": sum(size for _, _, size in self._pending_copies),
        }

    def close(self) -> None:
        if self._closed:
            return
        # Keep the runtime retryable if draining is interrupted. In particular,
        # the cache cannot close while its read-ahead worker still owns a read.
        # Do not run later cleanup in a finally that could mask this exception.
        if self._prefetch is not None:
            self._prefetch.close()
        if self._pending_copies:
            torch.cuda.synchronize(self.device)
            self._release_copies(synchronized=True)
        if self.cache is not None:
            self.cache.close()
        self._model.conditioning_cache = None
        self._closed = True
