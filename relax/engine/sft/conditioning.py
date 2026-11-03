# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Model-independent conditioning reuse and opt-in SFT measurement.

The adapter calls resolve inside its normal FSDP root forward. A unanimous
decision on the existing Gloo group keeps sharded forward collectives aligned.
Only backend-validated collective-free conditioning can decide rank-locally;
that execution policy is agreed once and fixed for the runtime lifetime.
Storage is node-local and run-scoped; it never contains noise or optimizer
data.
"""

from __future__ import annotations

import json
import os
import socket
import time
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


def conditioning_measurement_enabled() -> bool:
    return bool(
        _flag("RELAX_CONDITION_CACHE")
        or _flag("RELAX_CONDITION_PROFILE")
        or os.environ.get("RELAX_CONDITION_CACHE_RUN_DIR")
    )


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


def agree_execution_policy(model: Any, group: Any) -> str:
    """Agree once before any rank can take a different collective path.

    The backend grants this capability after checking replica ownership. A
    cache miss, changed input or poisoned weight guard never changes the
    execution policy: those conditions may differ between ranks.
    """
    try:
        policy = getattr(model, "_conditioning_execution_policy", "unanimous")
    except Exception:
        policy = None
    local = policy if isinstance(policy, str) and policy in ("unanimous", "rank_local") else None
    policies = [local]
    if dist.is_initialized():
        policies = [None] * dist.get_world_size(group)
        dist.all_gather_object(policies, local, group=group)
    if any(value not in ("unanimous", "rank_local") for value in policies):
        raise ValueError("Invalid backend conditioning execution policy")
    if any(value != policies[0] for value in policies):
        raise ValueError("Conditioning execution policy differs between training ranks")
    return policies[0]


def _tensor_bytes(features: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in features.values())


def _rss_bytes() -> int | None:
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


class ConditioningRuntime:
    """Optional resolver, cache ownership and per-rank raw JSONL telemetry."""

    def __init__(
        self,
        *,
        model: Any,
        model_path: str,
        device: torch.device,
        group: Any,
        rank: int,
        world_size: int,
        namespace: str,
        config: dict[str, Any],
        metadata_sha256: str,
        global_batch_size: int,
        start_step: int,
        num_steps: int,
        trainer_started: float | None = None,
    ) -> None:
        self.enabled = _flag("RELAX_CONDITION_CACHE")
        self.force_online = False  # numerical reference pass still populates the cache
        self.device, self.group, self.rank = device, group, rank
        self.world_size = world_size
        self._conditioning_started = time.monotonic()
        self._started = self._conditioning_started if trainer_started is None else trainer_started
        self._execution_policy = agree_execution_policy(model, group)
        self._closed = False
        self._close_event_written = False
        self._events: list[tuple[Any, Any]] = []
        self._step: defaultdict[str, int | float] = defaultdict(int)
        self._last_end: float | None = None
        self._cycle_started = 0.0
        self._cycle_unix = 0.0
        self._start_step = start_step
        self._num_steps = num_steps
        self._global_batch_size = global_batch_size
        self._samples: list[int] = []
        self._seen_samples: set[int] = set()
        self._repeated_sample = False
        self._model = model
        self._conditioning_snapshot: dict[str, Any] | None = None
        self._conditioning_semantic_snapshot: str | None = None
        self._cache_guard_disabled = bool(getattr(model, "conditioning_reuse_disabled", False))
        self._cache_guard_reason = getattr(model, "conditioning_reuse_disabled_reason", None)
        self._writer: Any = None
        self.cache: ConditioningCache | None = None
        self._prefetch: Any = None
        self.prefetch_depth = int(os.environ.get("RELAX_CONDITION_PREFETCH_DEPTH", "0"))
        prefetch_bytes = int(os.environ.get("RELAX_CONDITION_PREFETCH_MAX_BYTES", str(512 * 1024**2)))
        if self.prefetch_depth < 0 or (self.prefetch_depth and prefetch_bytes <= 0):
            raise ValueError("Conditioning prefetch requires nonnegative depth and positive byte capacity")
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
        if self.enabled:
            maximum = int(os.environ.get("RELAX_CONDITION_CACHE_MAX_BYTES", str(8 * 1024**3)))
            if maximum <= 0:
                raise ValueError("Enabled conditioning cache requires a positive byte limit")
            directory = os.environ.get("RELAX_CONDITION_CACHE_DIR") or "/dev/shm/relax-conditioning"
            self.cache = ConditioningCache(maximum, directory, f"{namespace}:{socket.gethostname()}")
        directory = os.environ.get("RELAX_CONDITION_CACHE_RUN_DIR")
        try:
            if self.enabled and self.prefetch_depth:
                from relax.engine.sft.conditioning_prefetch import ConditioningPrefetch

                self._prefetch = ConditioningPrefetch(self.cache, depth=self.prefetch_depth, max_bytes=prefetch_bytes)
            if directory:
                run_directory = Path(directory)
                run_directory.mkdir(parents=True, exist_ok=True)
                self._writer = (run_directory / f"events-rank{rank}.jsonl").open("x", buffering=1)
            self._write(
                {
                    "stage": "init",
                    "run_id": namespace,
                    "mode": "cached" if self.enabled else "online",
                    "rank": rank,
                    "world_size": world_size,
                    "global_batch_size": global_batch_size,
                    "start_step": start_step,
                    "expected_steps": num_steps - start_step,
                    "config": config,
                    "metadata_sha256": metadata_sha256,
                    "model_fingerprint": self.fingerprint,
                    "created_unix": time.time(),
                    "time_init_seconds": time.monotonic() - self._started,
                    "conditioning_init_seconds": time.monotonic() - self._conditioning_started,
                    "total_seconds_scope": "trainer lifetime; model and Ray startup excluded (see external job.json)",
                    "cache_enabled": self.enabled,
                    "conditioning_execution_policy": self._execution_policy,
                    "prefetch_depth": self.prefetch_depth if self.enabled else 0,
                    "prefetch_max_bytes": prefetch_bytes if self._prefetch is not None else 0,
                    "prefetch_scope": "validated CPU leases; device copies remain on the training thread",
                }
            )
            self._model.conditioning_cache = self
        except BaseException:
            # No prefetch can be submitted until initialization returns. Try
            # every cleanup, retaining the original initialization exception.
            for resource in (self._prefetch, self.cache, self._writer):
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as cleanup_error:
                        logger.warning("Conditioning initialization cleanup failed: %s", cleanup_error)
            self._closed = True
            raise

    def _write(self, event: dict[str, Any]) -> None:
        if self._writer is not None:
            self._writer.write(json.dumps(event, allow_nan=False) + "\n")

    def begin_step(self) -> None:
        self._step.clear()
        self._events.clear()
        self._samples.clear()
        now = time.monotonic()
        self._cycle_started = self._last_end if self._last_end is not None else now
        self._cycle_unix = time.time() - (now - self._cycle_started)

    def sample(self, sample_id: int) -> None:
        self._check_conditioning_state()
        self._samples.append(sample_id)
        self._repeated_sample = sample_id in self._seen_samples
        self._seen_samples.add(sample_id)

    def _check_conditioning_state(self) -> None:
        """Fail closed on a conditioning-master change outside FSDP forward.

        The adapter optionally declares ``frozen``, named tensor metadata and
        CPU-only ``semantic`` configuration. Sharded encoders still enter
        unanimous_ready: one poisoned
        shard owner makes the whole group compute. A collective-free replica
        instead bypasses reuse locally without changing its execution policy.
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

    def record_input_transfer(self, nbytes: int) -> None:
        self._step["online_input_h2d_bytes"] += nbytes
        if self._repeated_sample:
            self._step["repeated_h2d_bytes"] += nbytes

    def record_prepare(self, seconds: float) -> None:
        self._step["condition_prepare_seconds"] += seconds

    @property
    def prefetch_enabled(self) -> bool:
        return self._prefetch is not None and not self._closed

    def _conditioning_key(self, spec: ConditioningSpec) -> str:
        return build_key({"weights": self.fingerprint, "schema": spec.name, "dependencies": spec.dependencies})

    def prefetch(self, spec: ConditioningSpec) -> bool:
        """Schedule CPU storage work for an already prepared effective
        input."""
        if not self.prefetch_enabled or self.force_online or self._cache_guard_disabled or not spec.cacheable:
            return False
        return self._prefetch.submit(self._conditioning_key(spec))

    def prefetch_many(self, specs: list[ConditioningSpec]) -> int:
        """Replace the bounded future window, without model work or
        collectives.

        The caller prepares inputs on the original RNG stream/order. Resolve
        rebuilds the actual key and still checks the frozen-parameter guard;
        this method never treats a sample ID as a cache identity.
        """
        if not self.prefetch_enabled:
            return 0
        if self.force_online or self._cache_guard_disabled:
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
        started = time.perf_counter()
        features = restored = None
        local_hit = False
        key = None
        lookup_started = time.perf_counter()
        reusable = self.enabled and spec.cacheable and not self._cache_guard_disabled
        if self._prefetch is not None and (not reusable or self.force_online):
            self._prefetch.clear()
        if reusable:
            try:
                key = self._conditioning_key(spec)
                lease = None
                if not self.force_online:
                    if self._prefetch is not None:
                        lease = self._prefetch.take(key)
                        self._step["prefetch_hit"] += int(lease is not None)
                    if lease is None:
                        lease = self.cache.lookup(key)
                if lease is not None:
                    with lease:
                        copied_bytes = 0
                        copy_event = torch.cuda.Event() if self.device.type == "cuda" else None
                        enqueue_started = time.perf_counter()
                        try:
                            copy_to = getattr(lease, "copy_to", None)
                            if copy_to is not None:
                                features, copied_bytes = copy_to(self.device)
                            else:
                                features = {}
                                for name, tensor in lease.data.items():
                                    features[name] = tensor.to(
                                        self.device, non_blocking=True, copy=self.device.type == "cpu"
                                    )
                                    copied_bytes += tensor.numel() * tensor.element_size()
                        finally:
                            self._step["h2d_enqueue_seconds"] += time.perf_counter() - enqueue_started
                            if copy_event is not None:
                                self._step["condition_h2d_bytes"] += copied_bytes
                                wait_started = time.perf_counter()
                                try:
                                    # Drain even a partially enqueued copy before releasing CPU mappings.
                                    copy_event.record(torch.cuda.current_stream(self.device))
                                    copy_event.synchronize()
                                finally:
                                    self._step["h2d_wait_seconds"] += time.perf_counter() - wait_started
                        restore_started = time.perf_counter()
                        try:
                            restored = restore(features)  # validate before the unanimous decision
                        finally:
                            self._step["cache_restore_seconds"] += time.perf_counter() - restore_started
                        local_hit = True
            except Exception as error:
                features = restored = None
                self._step["cache_errors"] += 1
                if key is not None and self.cache is not None:
                    self.cache.invalidate(key)
                logger.warning(
                    "Conditioning read failed; using online encoding (%s): %s", self._execution_policy, error
                )
        self._step["cache_lookup_seconds"] += time.perf_counter() - lookup_started
        self._step["local_hit"] += int(local_hit)
        coordinate_started = time.perf_counter()
        ready = local_hit and not self.force_online
        if self.enabled and self._execution_policy == "unanimous":
            self._step["coordinate_calls"] += 1
            ready = unanimous_ready(ready, self.group)
        self._step["coordinate_seconds"] += time.perf_counter() - coordinate_started
        if ready:
            self._step["effective_hit"] += 1
            self._step["cache_read_bytes"] += _tensor_bytes(features)
            self._step["condition_seconds"] += time.perf_counter() - started
            return restored
        self._step["forced_recompute"] += int(local_hit)
        self._step["miss"] += int(not local_hit)
        features = restored = None
        encode_started = time.perf_counter()
        gpu_start = gpu_end = None
        if self.device.type == "cuda":
            gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            gpu_start.record()
        features = compute()
        if gpu_end is not None:
            gpu_end.record()
            self._events.append((gpu_start, gpu_end))
        else:
            self._step["encode_seconds"] += time.perf_counter() - encode_started
        self._step["encoding_calls"] += 1
        self._step["condition_payload_bytes"] += _tensor_bytes(features)
        if reusable and key is not None and (not local_hit or self.force_online):
            store_started = time.perf_counter()
            try:
                if self.cache.put(key, features):
                    self._step["cache_write_bytes"] += _tensor_bytes(features)
                    self._step["condition_d2h_bytes"] += _tensor_bytes(features)
            except Exception as error:
                self._step["cache_errors"] += 1
                logger.warning("Conditioning write failed; retaining online result: %s", error)
            self._step["cache_store_seconds"] += time.perf_counter() - store_started
        self._step["condition_seconds"] += time.perf_counter() - started
        return restore(features)

    def end_step(self, step: int, *, local_samples: int, loss: float) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._step["encode_seconds"] += sum(start.elapsed_time(end) / 1000 for start, end in self._events)
        detailed = step == self._start_step or step % 16 == 0 or step + 1 >= self._num_steps
        resources = self._resources(detailed=detailed)
        ended = time.monotonic()
        counters = {
            name: self._step[name]
            for name in (
                "condition_prepare_seconds",
                "condition_seconds",
                "encode_seconds",
                "cache_lookup_seconds",
                "cache_store_seconds",
                "h2d_enqueue_seconds",
                "h2d_wait_seconds",
                "cache_restore_seconds",
                "coordinate_seconds",
                "coordinate_calls",
                "local_hit",
                "effective_hit",
                "miss",
                "forced_recompute",
                "encoding_calls",
                "repeated_h2d_bytes",
                "online_input_h2d_bytes",
                "condition_h2d_bytes",
                "condition_d2h_bytes",
                "cache_read_bytes",
                "cache_write_bytes",
                "condition_payload_bytes",
                "cache_errors",
                "prefetch_hit",
            )
        }
        self._write(
            {
                "stage": "step",
                "step": step,
                "rank": self.rank,
                "global_samples": self._global_batch_size,
                "verification": _flag("RELAX_CONDITION_VERIFY") and step == self._start_step,
                "cache_guard_disabled": self._cache_guard_disabled,
                "cache_guard_reason": self._cache_guard_reason,
                "local_samples": local_samples,
                "sample_ids": list(self._samples),
                "started_unix": self._cycle_unix,
                "finished_unix": time.time(),
                "wall_seconds": ended - self._cycle_started,
                "loss": loss,
                **counters,
                **resources,
            }
        )
        self._last_end = ended

    def _resources(self, *, detailed: bool = True) -> dict[str, Any]:
        cache: dict[str, Any] = {}
        scope = "disabled"
        if self.cache is not None:
            light = getattr(self.cache, "stats_light", None)
            use_light = not detailed and callable(light)
            try:
                cache = light() if use_light else self.cache.stats()
                scope = cache.get("storage_scope", "index_accounting" if use_light else "filesystem")
            except Exception as error:
                scope = "unavailable"
                cache = {"storage_scope": scope, "error": f"{type(error).__name__}: {error}"}
                logger.warning("Conditioning resource snapshot unavailable: %s", error)
        current = scope in {"filesystem", "index_accounting"}
        result = {
            "cpu_rss_bytes": _rss_bytes(),
            "cache_bytes": cache.get("resident_bytes") if current else (0 if self.cache is None else None),
            "cache_stats": cache,
            "cache_storage_scope": scope,
            "cache_stats_detailed": scope == "filesystem",
            "disk_bytes": 0,
            "prefetch_stats": self._prefetch.stats() if self._prefetch is not None else {},
        }
        try:
            stat = os.statvfs("/dev/shm")
            result["shm_used_bytes"] = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
        except OSError:
            result["shm_used_bytes"] = None
        if self.device.type == "cuda":
            result.update(
                {
                    "gpu_allocated_bytes": torch.cuda.memory_allocated(self.device),
                    "gpu_reserved_bytes": torch.cuda.memory_reserved(self.device),
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                }
            )
        if self.cache is not None and not str(self.cache.directory).startswith("/dev/shm/"):
            result["disk_bytes"] = cache.get("total_bytes") if current else None
        return result

    def close(self, completed_step: int, *, status: str | None = None) -> None:
        if self._closed:
            return
        started = time.monotonic()
        # Keep the runtime retryable if draining is interrupted. In particular,
        # the cache cannot close while its read-ahead worker still owns a read.
        # Do not run later cleanup in a finally that could mask this exception.
        if self._prefetch is not None:
            self._prefetch.close()
        if self.cache is not None:
            self.cache.close()
        if not self._close_event_written:
            self._write(
                {
                    "stage": "close",
                    "rank": self.rank,
                    "completed_steps": completed_step - self._start_step,
                    "status": status or ("complete" if completed_step >= self._num_steps else "error"),
                    "total_seconds": time.monotonic() - self._started,
                    "close_seconds": time.monotonic() - started,
                    **self._resources(detailed=True),
                }
            )
            self._close_event_written = True
        if self._writer is not None:
            self._writer.close()
        self._model.conditioning_cache = None
        self._closed = True
