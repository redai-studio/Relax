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
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

    from .consumer import ConsumerLayerReader, ConsumerTensorCache


@dataclass(frozen=True)
class _PrefetchWindow:
    future: Future[None]
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
        cached = self.cache.get(source._cache_key)
        if cached is None and self.pending is not None:
            self.drain()
            cached = self.cache.get(source._cache_key)
        if cached is None:
            cached = source._materialize()
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
            window.future.result()
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
        if self.pending is not None or not self.cache.capacity_bytes:
            return
        # Reserve raw outputs, possible compact copies and transport/scatter
        # scratch. The bound excludes TQ internals and the pinned allocator pool.
        reserved = source.reader.read_scratch_bytes
        remaining = self.cache.capacity_bytes - self.cache.bytes - reserved
        following = False
        sources = []
        for candidate in self._groups[source._cache_group].values():
            if candidate is source:
                following = True
                continue
            if not following or candidate.released or candidate._cache_key in self.cache.entries:
                continue
            size = 2 * self.cache._source_bytes[candidate._cache_key]
            if size > remaining:
                break
            sources.append(candidate)
            reserved += size
            remaining -= size
        if not sources:
            return
        staging = tuple(candidate._allocate_staging() for candidate in sources)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-read")
        future = self._executor.submit(
            source.reader._read_layers_into,
            source.batch_layout.samples,
            [(candidate.layer, value.numpy()) for candidate, value in zip(sources, staging, strict=True)],
        )
        self.pending = _PrefetchWindow(future, tuple(sources), staging, reserved)
        self.peak_managed_bytes = max(self.peak_managed_bytes, self.cache.bytes + reserved)
