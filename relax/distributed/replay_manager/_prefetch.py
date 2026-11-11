# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One bounded CPU read-ahead window for a train batch's consumer sources.

The consumer thread owns registration, admission and buffer allocation. The
worker only reads into those buffers. Calls to read/release are serialized by
consumer_lock; the worker never acquires that lock or calls back into cache.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from time import perf_counter
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

    from .consumer import ConsumerLayerReader, ConsumerTensorCache


@dataclass(frozen=True)
class _PrefetchWindow:
    future: Future[float]
    sources: tuple[ConsumerLayerReader, ...]
    staging: tuple[torch.Tensor, ...]
    reserved_bytes: int


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

    @property
    def reserved_bytes(self) -> int:
        return self.pending.reserved_bytes if self.pending is not None else 0

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
        cached = self.cache.get(source._cache_key)
        if cached is None and self.pending is not None:
            ready = self.pending.future.done()
            started = perf_counter()
            try:
                self.drain()
            finally:
                self.demand_wait_seconds += perf_counter() - started
            cached = self.cache.get(source._cache_key)
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

    def drain(self) -> None:
        window = self.pending
        if window is None:
            return
        # No reader/store lock is held here. Finish the whole window before
        # admitting any result, so a failure cannot publish partial tensors.
        try:
            self.prefetch_read_seconds += window.future.result()
            values = [
                source._finish_materialize(staging)
                for source, staging in zip(window.sources, window.staging, strict=True)
            ]
        finally:
            self.pending = None
        for source, value in zip(window.sources, values, strict=True):
            if not source.released:
                self.cache.put(source._cache_key, value)

    def schedule(self, source: ConsumerLayerReader) -> None:
        if self.pending is not None or not self.cache.capacity_bytes or len(self.cache.entries) == len(self._sources):
            return
        # Reserve raw outputs, possible compact copies and transport/scatter
        # scratch. The bound excludes TQ internals and the pinned allocator pool.
        reserved = source.reader.read_scratch_bytes
        remaining = self.cache.capacity_bytes - self.cache.bytes - reserved
        following = False
        blocked_current = False
        sources = []
        for candidate in self._groups[source._cache_group].values():
            if candidate is source:
                following = True
                continue
            if not following or candidate.released or candidate._cache_key in self.cache.entries:
                continue
            size = 2 * self.cache._source_bytes[candidate._cache_key]
            if size > remaining:
                blocked_current = True
                break
            sources.append(candidate)
            reserved += size
            remaining -= size
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

        def read_window() -> float:
            started = perf_counter()
            sources[0].reader._read_layers_into(
                sources[0].batch_layout.samples,
                [(candidate.layer, value.numpy()) for candidate, value in zip(sources, staging, strict=True)],
            )
            return perf_counter() - started

        future = self._executor.submit(read_window)
        self.pending = _PrefetchWindow(future, tuple(sources), staging, reserved)
        self.prefetch_windows += 1
        self.prefetched_layers += len(sources)
        self.max_window_layers = max(self.max_window_layers, len(sources))
        self.cross_group_windows += sources[0]._cache_group != source._cache_group
        self.peak_managed_bytes = max(self.peak_managed_bytes, self.cache.bytes + reserved)
