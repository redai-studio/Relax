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
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

    from .consumer import ConsumerLayerReader, ConsumerTensorCache


_INDEXER_PREFETCH_BYTES = 256 << 20


@dataclass(frozen=True)
class _PrefetchWindow:
    future: Future[None]
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
        self._last_indexer_read: tuple[tuple[int, int], int] | None = None
        self._indexer_direction = 1

    @property
    def reserved_bytes(self) -> int:
        window = self.pending
        if window is None:
            return 0
        # Worker arguments retain all raw buffers until the window completes.
        # Count an admitted buffer here again if its cache entry was evicted.
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
        if source.kind == "indexer":
            previous = self._last_indexer_read
            if previous is None or previous[0] != source._cache_group:
                self._indexer_direction = 1
            elif previous[1] != source.layer:
                self._indexer_direction = 1 if source.layer > previous[1] else -1
            self._last_indexer_read = (source._cache_group, source.layer)
        # A failed background read must also interrupt otherwise warm hits.
        if self.pending is not None and self.pending.future.done():
            self.drain()
        cached = self.cache.get(source._cache_key)
        if cached is None and self.pending is not None:
            window = self.pending
            if window.ready and source in window.sources:
                index = window.sources.index(source)
                try:
                    window.ready[index].result()
                    if window.future.done():
                        window.future.result()
                    cached = source._finish_materialize(window.staging[index])
                    self.cache.put(source._cache_key, cached)
                    window.admitted.add(source._cache_key)
                except BaseException as primary:
                    # All worker writes must finish before dropping its window
                    # or allowing a reader lease to be released.
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
        if cached is None:
            cached = source._materialize()
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
        # No reader/store lock is held here. Release still waits for every
        # worker write, including layers already admitted to the CPU cache.
        try:
            window.future.result()
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
        if self.pending is not None or not self.cache.capacity_bytes:
            return
        # Reserve raw outputs, possible compact copies and transport/scatter
        # scratch. The bound excludes TQ internals and the pinned allocator pool.
        reserved = source.reader.read_scratch_bytes
        remaining = self.cache.capacity_bytes - self.cache.bytes - reserved
        window_remaining = _INDEXER_PREFETCH_BYTES
        following = False
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
            # Indexer staging already has its final int32 representation; only
            # MoE can need a second, compact CPU copy during materialization.
            size = self.cache._source_bytes[candidate._cache_key] * (1 if source.kind == "indexer" else 2)
            # Bound cold read-ahead separately from retained materializations.
            # Check before eviction: a full window must not discard warm data.
            if source.kind == "indexer" and size > window_remaining:
                break
            if size > remaining and source.kind == "indexer":
                # Keep the current output and cached layers nearer than this
                # candidate. Older or more distant entries may yield space to
                # the next read; otherwise a full LRU disables read-ahead.
                evictable = [(key, value[1]) for key, value in self.cache.entries.items() if key not in protected]
                if size <= remaining + sum(value for _, value in evictable):
                    for key, value in evictable:
                        self.cache.discard(key)
                        self.cache.evictions += 1
                        remaining += value
                        if size <= remaining:
                            break
            if size > remaining:
                break
            sources.append(candidate)
            reserved += size
            remaining -= size
            window_remaining -= size
        if not sources:
            return
        staging = tuple(candidate._allocate_staging() for candidate in sources)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-read")
        ready: tuple[Future[None], ...] = tuple(Future() for _ in sources) if source.kind == "indexer" else ()

        def load() -> None:
            try:
                outputs = [(candidate.layer, value.numpy()) for candidate, value in zip(sources, staging, strict=True)]
                if ready:
                    source.reader._read_layers_into(
                        source.batch_layout.samples, outputs, on_layer=lambda index: ready[index].set_result(None)
                    )
                    if not all(value.done() for value in ready):
                        raise RuntimeError("Replay read completed without publishing every layer")
                else:
                    source.reader._read_layers_into(source.batch_layout.samples, outputs)
            except BaseException as exc:
                for value in ready:
                    if not value.done():
                        value.set_exception(exc)
                raise

        future = self._executor.submit(load)
        self.pending = _PrefetchWindow(future, tuple(sources), staging, reserved, ready)
        self.peak_managed_bytes = max(self.peak_managed_bytes, self.cache.bytes + reserved)
