# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One bounded CPU read-ahead window for a train batch's consumer sources.

The consumer thread owns registration, admission and buffer allocation. The
worker only reads into those buffers. Calls to read/release are serialized by
consumer_lock; the worker never acquires that lock or calls back into cache.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

    from .consumer import ConsumerLayerReader, ConsumerTensorCache


_INDEXER_PREFETCH_BYTES = 256 << 20


@dataclass(frozen=True)
class _PrefetchWindow:
    future: Future[float]
    sources: tuple[ConsumerLayerReader, ...]
    staging: tuple[torch.Tensor, ...]
    reserved_bytes: int
    ready: tuple[Future[None], ...] = ()
    admitted: set[object] = field(default_factory=set)


class _ConsumerReadAhead:
    def __init__(self, cache: ConsumerTensorCache) -> None:
        self.cache = cache
        self.consumer_lock = RLock()
        self._sources: dict[object, ConsumerLayerReader] = {}
        self._groups: dict[tuple[int, int], dict[object, ConsumerLayerReader]] = {}
        self._executor: ThreadPoolExecutor | None = None
        self.pending: _PrefetchWindow | None = None
        self.peak_managed_bytes = 0
        self.read_calls = self.cold_reads = self.prefetch_hits = 0
        self.prefetch_windows = self.prefetched_layers = self.max_window_layers = 0
        self.cross_group_windows = self.prefetch_ready_hits = 0
        self.read_seconds = self.demand_wait_seconds = 0.0
        self.prefetch_read_seconds = 0.0
        self._last_indexer_read: tuple[tuple[int, int], int] | None = None
        self._indexer_direction = 1

    @property
    def reserved_bytes(self) -> int:
        window = self.pending
        if window is None:
            return 0
        # The worker retains all staging until its window completes. An
        # admitted buffer evicted from cache must count against the window again.
        cached = sum(self.cache.entries[key][1] for key in window.admitted if key in self.cache.entries)
        return window.reserved_bytes - cached

    def register(self, source: ConsumerLayerReader) -> None:
        self._sources[source._cache_key] = source
        self._groups.setdefault(source._cache_group, {})[source._cache_key] = source

    def unregister(self, key: object) -> None:
        source = self._sources.pop(key, None)
        if source is not None:
            group = self._groups[source._cache_group]
            group.pop(key)
            if not group:
                del self._groups[source._cache_group]
        if not self._sources and self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def read(self, source: ConsumerLayerReader) -> tuple[torch.Tensor, torch.Tensor | None]:
        started = perf_counter()
        self.read_calls += 1
        try:
            return self._read(source)
        finally:
            self.read_seconds += perf_counter() - started

    def _read(self, source: ConsumerLayerReader) -> tuple[torch.Tensor, torch.Tensor | None]:
        if source.kind == "indexer":
            previous = self._last_indexer_read
            if previous is None or previous[0] != source._cache_group:
                self._indexer_direction = 1
            elif previous[1] != source.layer:
                self._indexer_direction = 1 if source.layer > previous[1] else -1
            self._last_indexer_read = (source._cache_group, source.layer)
        # A completed background failure must interrupt warm cache hits too.
        if self.pending is not None and self.pending.future.done():
            prefetched = source in self.pending.sources and source._cache_key not in self.cache.entries
            self.drain()
            if prefetched and source._cache_key in self.cache.entries:
                self.prefetch_hits += 1
                self.prefetch_ready_hits += 1
        cached = self.cache.get(source._cache_key)
        if cached is None and self.pending is not None:
            window = self.pending
            progressive = bool(window.ready) and source in window.sources
            index = window.sources.index(source) if progressive else 0
            ready = window.ready[index].done() if progressive else window.future.done()
            started = perf_counter()
            try:
                if progressive:
                    try:
                        window.ready[index].result()
                        if window.future.done():
                            window.future.result()
                        cached = source._finish_materialize(window.staging[index])
                        self.cache.put(source._cache_key, cached)
                        window.admitted.add(source._cache_key)
                    except BaseException as primary:
                        # Drain every write before releasing buffers or leases.
                        try:
                            window.future.result()
                        except BaseException as cleanup:
                            self._discard_failed(window)
                            if cleanup is primary:
                                raise
                            raise primary from cleanup
                        self._discard_failed(window)
                        raise
                else:
                    self.drain()
                    cached = self.cache.get(source._cache_key)
            finally:
                self.demand_wait_seconds += perf_counter() - started
            self.prefetch_hits += cached is not None
            self.prefetch_ready_hits += ready and cached is not None
        if cached is None:
            self.cold_reads += 1
            started = perf_counter()
            cached = source._materialize()
            self.demand_wait_seconds += perf_counter() - started
            self.cache.put(source._cache_key, cached)
        self.schedule(source)
        return cached

    def _discard_failed(self, window: _PrefetchWindow) -> None:
        for key in window.admitted:
            self.cache.discard(key)
        self.pending = None

    def drain(self) -> None:
        window = self.pending
        if window is None:
            return
        # No reader/store lock is held. Even admitted layers must wait for
        # the entire worker before a reader lease can be released.
        try:
            self.prefetch_read_seconds += window.future.result()
            values = [
                (source, source._finish_materialize(staging))
                for source, staging in zip(window.sources, window.staging, strict=True)
                if source._cache_key not in window.admitted
            ]
        except BaseException:
            self._discard_failed(window)
            raise
        finally:
            self.pending = None
        for source, value in values:
            if not source.released:
                self.cache.put(source._cache_key, value)

    def schedule(self, source: ConsumerLayerReader) -> None:
        if self.pending is not None or not self.cache.capacity_bytes or len(self.cache.entries) == len(self._sources):
            return
        # Reserve raw outputs, possible compact copies and transport/scatter
        # scratch. The bound excludes TQ internals and the pinned allocator pool.
        reserved = source.reader.read_scratch_bytes
        remaining = self.cache.capacity_bytes - self.cache.bytes - reserved
        window_remaining = _INDEXER_PREFETCH_BYTES
        following = False
        blocked_current = False
        sources = []
        group = self._groups[source._cache_group]
        candidates = (
            reversed(group.values()) if source.kind == "indexer" and self._indexer_direction < 0 else group.values()
        )
        protected = {source._cache_key}
        for candidate in candidates:
            if candidate is source:
                following = True
                continue
            if not following or candidate.released:
                continue
            if candidate._cache_key in self.cache.entries:
                protected.add(candidate._cache_key)
                continue
            size = self.cache._source_bytes[candidate._cache_key] * (1 if source.kind == "indexer" else 2)
            if source.kind == "indexer" and size > window_remaining:
                break
            if size > remaining and source.kind == "indexer":
                evictable = [(key, value[1]) for key, value in self.cache.entries.items() if key not in protected]
                if size <= remaining + sum(value for _, value in evictable):
                    for key, value in evictable:
                        self.cache.discard(key)
                        self.cache.evictions += 1
                        remaining += value
                        if size <= remaining:
                            break
            if size > remaining:
                blocked_current = True
                break
            sources.append(candidate)
            reserved += size
            remaining -= size
            window_remaining -= size
        if not sources and not blocked_current and source.kind == "moe":
            # Once this microbatch's following layers are cached, overlap the
            # next microbatch's first reads with the remaining model compute.
            # MoE's small top-k permits batching the next group's layers under
            # the existing byte budget while this group's cached layers run.
            following_group = False
            for key, group in self._groups.items():
                if key == source._cache_group:
                    following_group = True
                    continue
                if not following_group:
                    continue
                target = next(iter(group.values()))
                reserved = target.reader.read_scratch_bytes
                remaining = self.cache.capacity_bytes - self.cache.bytes - reserved
                for candidate in group.values():
                    if candidate.released or candidate._cache_key in self.cache.entries:
                        continue
                    if candidate.kind != "moe":
                        break
                    size = 2 * self.cache._source_bytes[candidate._cache_key]
                    if size > remaining:
                        break
                    sources.append(candidate)
                    reserved += size
                    remaining -= size
                break
        if not sources:
            return
        staging = tuple(candidate._allocate_staging() for candidate in sources)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-read")

        ready: tuple[Future[None], ...] = tuple(Future() for _ in sources) if source.kind == "indexer" else ()

        def read_window() -> float:
            started = perf_counter()
            try:
                outputs = [(candidate.layer, value.numpy()) for candidate, value in zip(sources, staging, strict=True)]
                if ready:
                    sources[0].reader._read_layers_into(
                        sources[0].batch_layout.samples, outputs, on_layer=lambda index: ready[index].set_result(None)
                    )
                    if not all(value.done() for value in ready):
                        raise RuntimeError("Replay read completed without publishing every layer")
                else:
                    sources[0].reader._read_layers_into(sources[0].batch_layout.samples, outputs)
            except BaseException as exc:
                for value in ready:
                    if not value.done():
                        value.set_exception(exc)
                raise
            return perf_counter() - started

        future = self._executor.submit(read_window)
        self.pending = _PrefetchWindow(future, tuple(sources), staging, reserved, ready)
        self.prefetch_windows += 1
        self.prefetched_layers += len(sources)
        self.max_window_layers = max(self.max_window_layers, len(sources))
        self.cross_group_windows += sources[0]._cache_group != source._cache_group
        self.peak_managed_bytes = max(self.peak_managed_bytes, self.cache.bytes + reserved)
