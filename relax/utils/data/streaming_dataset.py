# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Streaming Dataset Implementation.

This module provides a memory-efficient streaming dataset implementation
that loads data on-demand instead of loading everything into memory at once.

Key features:
- Lazy loading: Data is loaded only when needed
- Buffer caching: Recently accessed samples are cached
- Shuffle support: Global shuffle with epoch-based seeding
- Filter support: Length filtering is done at access time

Usage:
    from relax.utils.data.streaming_dataset import StreamingDataset

    dataset = StreamingDataset(
        path="data.jsonl",
        tokenizer=tokenizer,
        processor=processor,
        max_length=2048,
        buffer_size=10000,
    )

    # Get a batch of samples
    samples, crossed_epoch = dataset.get_batch(32)
"""

import json
import os
import threading
import time
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Iterator, Optional

import numpy as np


try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from relax.utils.data.data_utils import (
    BaseDataset,
    check_sample_length,
    parse_generalized_path,
    resolve_path_plan,
)
from relax.utils.logging_utils import get_logger
from relax.utils.multimodal.config import MultimodalConfig
from relax.utils.types import Sample


logger = get_logger(__name__)

__all__ = [
    "StreamingDataset",
    "StreamingReader",
    "CompositeStreamingReader",
    "SampleBuffer",
    "IndexManager",
    "PrefetchBuffer",
]


class StreamingReader:
    """Streaming file reader with random access support.

    Builds an index of line offsets on first access to enable efficient random
    access to any line in the file.
    """

    def __init__(self, path: str):
        """Initialize the streaming reader.

        Args:
            path: Path to the data file (JSONL or Parquet)
        """
        self.path, self.row_slice = parse_generalized_path(path)

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset path '{self.path}' does not exist.")

        self._line_offsets: Optional[list[int]] = None
        self._total_lines: Optional[int] = None
        self._is_parquet = self.path.endswith(".parquet")
        self._parquet_data: Optional[list] = None  # For parquet, we cache in memory

        # Validate file format
        if not self.path.endswith((".jsonl", ".parquet")):
            raise ValueError(f"Unsupported file format: {self.path}. Supported formats are .jsonl and .parquet.")

    def _build_index(self) -> None:
        """Build line offset index for JSONL files.

        This scans the file once to record the byte offset of each line,
        enabling efficient random access later.
        """
        if self._line_offsets is not None:
            return

        if self._is_parquet:
            self._load_parquet()
            return

        logger.info(f"Building index for {self.path}...")
        self._line_offsets = []

        with open(self.path, "rb") as f:
            offset = 0
            for line in f:
                line_stripped = line.strip()
                if line_stripped:  # Skip empty lines
                    self._line_offsets.append(offset)
                offset += len(line)

        # Apply row slice if specified
        if self.row_slice is not None:
            start = self.row_slice.start or 0
            stop = self.row_slice.stop or len(self._line_offsets)
            self._line_offsets = self._line_offsets[start:stop]

        self._total_lines = len(self._line_offsets)
        logger.info(f"Index built: {self._total_lines} lines")

    def _load_parquet(self) -> None:
        """Load parquet file into memory."""
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        logger.info(f"Loading parquet file {self.path}...")
        pf = pq.ParquetFile(self.path)
        self._parquet_data = []

        # Read row groups individually instead of using iter_batches().
        # iter_batches() creates chunked arrays for multi-row-group files,
        # which fails with ArrowNotImplementedError on nested types
        # (e.g. list<struct<...>>, struct<...>).
        for i in range(pf.metadata.num_row_groups):
            self._parquet_data.extend(pf.read_row_group(i).to_pylist())

        # Apply row slice if specified
        if self.row_slice is not None:
            start = self.row_slice.start or 0
            stop = self.row_slice.stop or len(self._parquet_data)
            self._parquet_data = self._parquet_data[start:stop]

        self._total_lines = len(self._parquet_data)
        logger.info(f"Parquet loaded: {self._total_lines} rows")

    def __len__(self) -> int:
        """Return total number of lines/rows."""
        if self._total_lines is None:
            self._build_index()
        return self._total_lines

    def __getitem__(self, idx: int) -> dict:
        """Read a single line/row by index.

        Args:
            idx: Line index (0-based)

        Returns:
            Parsed JSON object or parquet row
        """
        if self._total_lines is None:
            self._build_index()

        if idx < 0 or idx >= self._total_lines:
            raise IndexError(f"Index {idx} out of range [0, {self._total_lines})")

        if self._is_parquet:
            return self._parquet_data[idx]

        # Read from JSONL using line offset
        offset = self._line_offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(offset)
            line = f.readline().decode("utf-8").strip()
            return json.loads(line)

    def iter_batch(self, indices: list[int]) -> Iterator[dict]:
        """Iterate over multiple indices efficiently.

        For JSONL, this sorts indices to minimize seeking.
        For Parquet, this is a simple iteration.

        Args:
            indices: List of indices to read

        Yields:
            (index, data) tuples
        """
        if self._total_lines is None:
            self._build_index()

        if self._is_parquet:
            for idx in indices:
                yield idx, self._parquet_data[idx]
            return

        # Sort indices for sequential reading (more efficient for HDD)
        sorted_pairs = sorted(enumerate(indices), key=lambda x: self._line_offsets[x[1]])
        results = [None] * len(indices)

        with open(self.path, "rb") as f:
            for original_pos, idx in sorted_pairs:
                offset = self._line_offsets[idx]
                f.seek(offset)
                line = f.readline().decode("utf-8").strip()
                results[original_pos] = (idx, json.loads(line))

        yield from results


class CompositeStreamingReader:
    """Compose multiple single-file StreamingReader instances into one logical
    reader with optional slicing over the concatenated sample stream."""

    def __init__(self, paths: list[str], row_slice: Optional[slice] = None):
        if not paths:
            raise ValueError("paths must not be empty")

        self.paths = paths
        self.row_slice = row_slice
        self.readers = [StreamingReader(path) for path in paths]
        self._cumulative_lengths: list[int] = []

        total = 0
        for reader in self.readers:
            total += len(reader)
            self._cumulative_lengths.append(total)

        self._base_total_lines = total
        self._slice_spec = None if row_slice is None else row_slice.indices(total)
        if self._slice_spec is None:
            self._total_lines = total
        else:
            start, stop, step = self._slice_spec
            self._total_lines = len(range(start, stop, step))

    def __len__(self) -> int:
        return self._total_lines

    def _external_to_global_index(self, idx: int) -> int:
        if idx < 0 or idx >= self._total_lines:
            raise IndexError(f"Index {idx} out of range [0, {self._total_lines})")

        if self._slice_spec is None:
            return idx

        start, _, step = self._slice_spec
        return start + idx * step

    def _locate_reader(self, global_idx: int) -> tuple[int, int]:
        reader_idx = bisect_right(self._cumulative_lengths, global_idx)
        prev_total = 0 if reader_idx == 0 else self._cumulative_lengths[reader_idx - 1]
        return reader_idx, global_idx - prev_total

    def __getitem__(self, idx: int) -> dict:
        global_idx = self._external_to_global_index(idx)
        reader_idx, local_idx = self._locate_reader(global_idx)
        return self.readers[reader_idx][local_idx]

    def iter_batch(self, indices: list[int]) -> Iterator[tuple[int, dict]]:
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for original_pos, idx in enumerate(indices):
            global_idx = self._external_to_global_index(idx)
            reader_idx, local_idx = self._locate_reader(global_idx)
            groups.setdefault(reader_idx, []).append((original_pos, idx, local_idx))

        results: list[Optional[tuple[int, dict]]] = [None] * len(indices)
        for reader_idx, entries in groups.items():
            local_indices = [local_idx for _, _, local_idx in entries]
            fetched = list(self.readers[reader_idx].iter_batch(local_indices))
            for (original_pos, external_idx, _), (_, data) in zip(entries, fetched, strict=True):
                results[original_pos] = (external_idx, data)

        for result in results:
            yield result


class SampleBuffer:
    """LRU cache for processed samples.

    Caches recently accessed samples to avoid re-processing them. Uses
    OrderedDict for O(1) access and LRU eviction.
    """

    def __init__(self, max_size: int = 10000):
        """Initialize the buffer.

        Args:
            max_size: Maximum number of samples to cache
        """
        self.cache: OrderedDict[int, Sample] = OrderedDict()
        self.max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, idx: int) -> Optional[Sample]:
        """Get a cached sample.

        Args:
            idx: Sample index

        Returns:
            Cached sample or None if not in cache
        """
        if idx in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(idx)
            self._hits += 1
            return self.cache[idx]
        self._misses += 1
        return None

    def put(self, idx: int, sample: Sample) -> None:
        """Cache a sample.

        Args:
            idx: Sample index
            sample: Sample to cache
        """
        if idx in self.cache:
            self.cache.move_to_end(idx)
            self.cache[idx] = sample
            return

        # Evict oldest if at capacity
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[idx] = sample

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self.cache)


class IndexManager:
    """Manages shuffle indices and epoch transitions.

    Generates a two-stage sample order: a persistent dataset shuffle followed
    by the epoch's sampler permutation. Tracks current position within the
    epoch.
    """

    def __init__(
        self,
        total_size: int,
        seed: int = 42,
        index_pool: Iterable[int] | None = None,
        dataset_seed_offset: int = 0,
    ):
        """Initialize the index manager.

        Args:
            total_size: Total number of samples
            seed: Random seed for reproducible shuffling
            index_pool: Optional physical row IDs to shuffle instead of
                ``range(total_size)``.
            dataset_seed_offset: Number of dataset RNG seeds consumed before
                the persistent dataset shuffle.
        """
        self.total_size = total_size
        self.seed = seed
        self.dataset_seed_offset = dataset_seed_offset
        self.index_pool = tuple(range(total_size)) if index_pool is None else tuple(index_pool)
        if len(self.index_pool) != total_size:
            raise ValueError(f"index_pool length {len(self.index_pool)} does not match total_size {total_size}")
        self.current_epoch = -1
        self.indices: Optional[list[int]] = None
        self.position = 0

    def shuffle(self, epoch_id: int) -> None:
        """Generate shuffle permutation for a new epoch.

        Args:
            epoch_id: Epoch identifier (used with seed for reproducibility)
        """
        if epoch_id == self.current_epoch:
            return

        # Stage 1: a dataset shuffle whose seed is drawn from RandomState(seed),
        # stable across epochs. Stage 2: a per-epoch ``torch.randperm(epoch)``
        # sampler permutation. Both run over positions ``[0, total_size)`` and are
        # then mapped through ``index_pool``, so a subsetted pool (physical row
        # IDs) is reordered by the same order rather than assuming the identity
        # pool ``range(total_size)``.
        random_state = np.random.RandomState(self.seed)
        dataset_seeds = random_state.randint(0, np.iinfo(np.int32).max, size=self.dataset_seed_offset + 1)
        dataset_seed = int(dataset_seeds[-1])
        dataset_indices = np.random.default_rng(dataset_seed).permutation(self.total_size)
        import torch

        generator = torch.Generator().manual_seed(epoch_id)
        sampler_indices = torch.randperm(self.total_size, generator=generator).numpy()
        pool = np.asarray(self.index_pool)
        self.indices = pool[dataset_indices[sampler_indices]].tolist()
        self.current_epoch = epoch_id
        self.position = 0

        logger.info(f"Shuffled dataset for epoch {epoch_id}")

    def get_next_indices(self, n: int) -> tuple[list[int], bool]:
        """Get next n indices from the current epoch.

        If we reach the end of the epoch and still need more samples, lazily
        advances to the next epoch (re-shuffles).  Exhausting the current epoch
        exactly (n == remaining) does NOT count as crossing — the boundary is
        only crossed when at least one returned sample comes from the next
        epoch's permutation.

        Args:
            n: Number of indices to get

        Returns:
            (indices, crossed_epoch): List of indices and whether at least one
            sample was drawn from the next epoch.
        """
        if self.indices is None:
            self.shuffle(0)

        indices = []
        crossed_epoch = False

        remaining = n
        while remaining > 0:
            if self.position >= self.total_size:
                new_epoch = self.current_epoch + 1
                self.shuffle(new_epoch)
                crossed_epoch = True

            available = self.total_size - self.position
            take = min(remaining, available)

            indices.extend(self.indices[self.position : self.position + take])
            self.position += take
            remaining -= take

        return indices, crossed_epoch

    def reset(self, position: int = 0, epoch_id: int = 0) -> None:
        """Reset to a specific position and epoch.

        Args:
            position: Position within the epoch
            epoch_id: Epoch to set
        """
        self.shuffle(epoch_id)
        self.position = position

    def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {
            "epoch_id": self.current_epoch,
            "position": self.position,
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint."""
        epoch_id = state.get("epoch_id", 0)
        position = state.get("position", 0)
        self.reset(position=position, epoch_id=epoch_id)


class PrefetchBuffer:
    """Background prefetch buffer for multimodal data loading.

    Run a background thread that pre-loads and pre-processes samples
    (including heavy video/image I/O) **in the exact order** they will be consumed.

    Key design:
    - ``set_index_order(indices)`` is called once (at ``shuffle`` time) with
      the **entire** upcoming index sequence.  The background thread starts
      fetching immediately, well before ``get_batch`` is called.
    - ``get(idx)`` pops from the cache (near-zero latency on hit) or falls
      back to a synchronous single-sample fetch on miss.
    - ``get_cached(idx)`` lets async producers avoid that fallback and wait
      for the background worker instead.
    - The cache is bounded by ``max_cached``; when full the prefetch thread
      pauses until consumers free space via ``get()`` calls.
    - A ``ThreadPoolExecutor`` is used inside the prefetch thread to
      parallelize video/image decoding across multiple files within a chunk,
      since PyAV/FFmpeg releases the GIL during C-level decoding.

    Lifecycle::

        buf = PrefetchBuffer(process_fn, chunk_size=16, max_cached=256, num_workers=4)
        buf.set_index_order([3, 7, 1, 5, ...])  # triggers background loading
        sample = buf.get(3)  # instant cache hit
    """

    def __init__(
        self,
        process_fn,
        chunk_size: int = 32,
        max_cached: int = 256,
        num_workers: int = 4,
    ):
        """Initialize the prefetch buffer.

        Args:
            process_fn: ``fn(idx: int) -> Optional[Sample]`` — load and
                process a single sample by index.
            chunk_size: Number of indices to submit to the thread-pool at
                a time inside the prefetch loop.
            max_cached: Maximum number of samples to keep in the cache
                before the prefetch thread pauses.
            num_workers: Number of parallel workers in the internal
                ``ThreadPoolExecutor`` for I/O-bound decoding.
        """
        self._process_fn = process_fn
        self._chunk_size = max(1, min(chunk_size, max_cached)) if max_cached > 0 else max(1, chunk_size)
        self._max_cached = max_cached
        self._num_workers = num_workers

        # Thread-safe cache: idx -> Optional[Sample]
        self._cache: dict[int, Optional[Sample]] = {}
        self._fallback_fetched_indices: set[int] = set()
        self._lock = threading.Lock()
        self._cache_updated = threading.Condition(self._lock)

        # Ordered index sequence set by set_index_order
        self._indices: list[int] = []
        self._pos: int = 0

        # Flow control: cleared when cache is full, set when space is freed
        self._space_available = threading.Event()
        self._space_available.set()

        # Stats
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        self._prefetch_stale_drops = 0

        # Thread lifecycle
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        logger.info(
            f"PrefetchBuffer created: max_cached={max_cached}, chunk_size={chunk_size}, "
            f"effective_chunk_size={self._chunk_size}, num_workers={num_workers}"
        )

    # -- Public API --------------------------------------------------------

    def set_index_order(self, indices: list[int]) -> None:
        """Reset the cache and start prefetching in *indices* order.

        Called at the beginning of each epoch (from
        ``StreamingDataset.shuffle``) with the full upcoming index sequence.
        The prefetch thread starts loading immediately so that later ``get()``
        calls hit the cache.
        """
        # Stop any running prefetch thread
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                logger.warning("Previous prefetch thread did not stop within 10s; it will exit on its own stop-event")

        with self._cache_updated:
            self._cache.clear()
            self._fallback_fetched_indices.clear()
            self._indices = list(indices)
            self._pos = 0
            self._cache_updated.notify_all()

        # Create a fresh stop-event for the new thread so the old thread
        # (if still draining) keeps seeing its own set() signal and exits.
        self._stop = threading.Event()
        self._space_available.set()
        stop_event = self._stop
        self._thread = threading.Thread(target=self._run, args=(stop_event,), daemon=True, name="prefetch-worker")
        self._thread.start()
        logger.info(f"PrefetchBuffer: started prefetching {len(indices)} samples")

    def get(self, idx: int) -> Optional[Sample]:
        """Return the sample for *idx*.

        Pops from the prefetch cache on hit.  On miss, performs a blocking
        single-index fetch via ``process_fn``.
        """
        with self._lock:
            if idx in self._cache:
                sample = self._cache.pop(idx)
                self._prefetch_hits += 1
                # Signal prefetch thread that space is available
                self._space_available.set()
                return sample
            self._prefetch_misses += 1
            self._fallback_fetched_indices.add(idx)

        # Cache miss — synchronous fallback
        try:
            return self._process_fn(idx)
        except Exception:
            logger.exception(f"Prefetch fallback failed for index {idx}")
            return None

    def get_cached(self, idx: int, *, record_miss: bool = True) -> tuple[bool, Optional[Sample]]:
        """Pop a prefetched sample without synchronous fallback.

        Returns ``(available, sample)`` so callers can distinguish a missing
        cache entry from a cached ``None`` sample that should be skipped.
        """
        with self._lock:
            if idx in self._cache:
                sample = self._cache.pop(idx)
                self._prefetch_hits += 1
                self._space_available.set()
                return True, sample
            if record_miss:
                self._prefetch_misses += 1
            return False, None

    def wait_for(self, idx: int, timeout: float | None = None) -> bool:
        """Wait until *idx* is present in the cache, or timeout/stop occurs."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cache_updated:
            while idx not in self._cache and not self._stop.is_set():
                thread = self._thread
                if thread is None or not thread.is_alive():
                    break
                if deadline is None:
                    self._cache_updated.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cache_updated.wait(timeout=remaining)
            return idx in self._cache

    @property
    def is_alive(self) -> bool:
        """Return whether the current prefetch thread is alive."""
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def stop(self) -> None:
        """Signal the prefetch thread to stop."""
        self._stop.set()
        # Unblock if waiting on space
        self._space_available.set()
        with self._cache_updated:
            self._cache_updated.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                logger.warning("Prefetch thread did not terminate within 15s")

    def clear(self) -> None:
        """Clear the cache and reset position (without stopping the thread)."""
        with self._cache_updated:
            self._cache.clear()
            self._cache_updated.notify_all()
        self._space_available.set()

    @property
    def hit_rate(self) -> float:
        """Return prefetch cache hit rate."""
        total = self._prefetch_hits + self._prefetch_misses
        return self._prefetch_hits / total if total > 0 else 0.0

    @property
    def cache_size(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)

    # -- Background thread -------------------------------------------------

    def _run(self, stop_event: threading.Event) -> None:
        """Background prefetch loop.

        Iterates through ``self._indices`` in order, loading chunks in parallel
        via a ``ThreadPoolExecutor``.  Pauses when the cache is full and
        resumes when ``get()`` frees space.

        Args:
            stop_event: Thread-local stop signal.  Each thread receives its
                own ``Event`` so that ``set_index_order`` can replace
                ``self._stop`` for the next thread without accidentally
                un-stopping this one.
        """
        _MAX_SUBMIT_RETRIES = 3
        consecutive_failures = 0

        with ThreadPoolExecutor(max_workers=self._num_workers, thread_name_prefix="pf") as pool:
            while not stop_event.is_set():
                # 1. Get next chunk of indices
                with self._lock:
                    if self._pos >= len(self._indices):
                        break  # All indices have been dispatched
                    chunk = self._indices[self._pos : self._pos + self._chunk_size]
                    self._pos += len(chunk)

                # 2. Filter out indices already in cache
                with self._lock:
                    to_fetch = []
                    for idx in chunk:
                        if idx in self._cache:
                            continue
                        if idx in self._fallback_fetched_indices:
                            self._fallback_fetched_indices.discard(idx)
                            self._prefetch_stale_drops += 1
                            continue
                        to_fetch.append(idx)
                if not to_fetch:
                    continue

                # 3. Wait until cache has room for this chunk
                while not stop_event.is_set():
                    with self._lock:
                        pending = []
                        for idx in to_fetch:
                            if idx in self._cache:
                                continue
                            if idx in self._fallback_fetched_indices:
                                self._fallback_fetched_indices.discard(idx)
                                self._prefetch_stale_drops += 1
                                continue
                            pending.append(idx)
                        to_fetch = pending
                        if not to_fetch:
                            break
                        if len(self._cache) + len(to_fetch) <= self._max_cached:
                            break
                        self._space_available.clear()
                    # Wait for consumers to pop entries
                    if not self._space_available.wait(timeout=0.1):
                        continue

                if not to_fetch:
                    continue

                if stop_event.is_set():
                    return

                # 4. Parallel-fetch all samples in the chunk
                try:
                    futures = {idx: pool.submit(self._process_fn, idx) for idx in to_fetch}
                    results = {}
                    for idx, fut in futures.items():
                        try:
                            results[idx] = fut.result(timeout=120)
                        except Exception:
                            logger.warning(f"Prefetch failed for index {idx}", exc_info=True)
                            results[idx] = None
                    consecutive_failures = 0
                except Exception:
                    consecutive_failures += 1
                    logger.exception(
                        f"Prefetch chunk submission failed (attempt {consecutive_failures}/{_MAX_SUBMIT_RETRIES})"
                    )
                    if consecutive_failures >= _MAX_SUBMIT_RETRIES:
                        logger.error("Prefetch thread aborting after too many consecutive submission failures")
                        break
                    time.sleep(0.5)
                    with self._lock:
                        self._pos -= len(chunk)
                    continue

                # 5. Store results in cache
                with self._cache_updated:
                    for idx, sample in results.items():
                        if idx in self._fallback_fetched_indices:
                            self._fallback_fetched_indices.discard(idx)
                            self._prefetch_stale_drops += 1
                            continue
                        self._cache[idx] = sample
                    self._cache_updated.notify_all()

        with self._cache_updated:
            self._cache_updated.notify_all()
        logger.info(
            f"Prefetch thread finished. Hit rate: {self.hit_rate:.1%} "
            f"(hits={self._prefetch_hits}, misses={self._prefetch_misses}, "
            f"stale_drops={self._prefetch_stale_drops})"
        )


class StreamingDataset(BaseDataset):
    """Memory-efficient streaming dataset with on-demand loading.

    Inherits from BaseDataset and implements lazy loading with LRU caching.

    Features:
    - Lazy loading: Only loads data when accessed
    - LRU caching: Caches recently accessed samples
    - Background prefetching: Pre-loads multimodal data in background thread
    - Shuffle support: Epoch-based reproducible shuffling
    - Filter support: Length filtering done at access time

    Example:
        dataset = StreamingDataset(
            path="data.jsonl",
            tokenizer=tokenizer,
            processor=processor,
            max_length=2048,
        )

        samples, crossed = dataset.get_batch(32)
    """

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        processor: Any,
        max_length: Optional[int],
        *,
        prompt_key: str = "text",
        multimodal_keys: Optional[dict] = None,
        label_key: Optional[str] = None,
        tool_key: Optional[str] = None,
        metadata_key: str = "metadata",
        system_prompt: Optional[str] = None,
        seed: int = 42,
        apply_chat_template: bool = False,
        apply_chat_template_kwargs: Optional[dict] = None,
        use_audio_in_video: bool = False,
        buffer_size: int = 10000,
        prefetch_chunk_size: int = 32,
        prefetch_max_cached: int = 256,
        prefetch_num_workers: int = 1,
        multimodal_config: MultimodalConfig = None,
        custom_prompt_func=None,
        teacher_prompt_key: Optional[str] = None,
        teacher_multimodal_keys: Optional[dict] = None,
    ):
        """Initialize the streaming dataset.

        Args:
            path: Path to data file (JSONL or Parquet)
            tokenizer: Tokenizer for length checking and chat template
            processor: Processor for multimodal inputs
            max_length: Maximum prompt length (samples exceeding this are filtered)
            prompt_key: Key for prompt in data
            multimodal_keys: Mapping of multimodal types to data keys
            label_key: Key for labels in data
            tool_key: Key for tools in data
            metadata_key: Key for metadata in data
            system_prompt: System prompt key
            seed: Random seed for shuffling
            apply_chat_template: Whether to apply chat template
            apply_chat_template_kwargs: Additional kwargs for chat template
            use_audio_in_video: Whether to extract audio from video files for multimodal processing
            buffer_size: Maximum samples to cache in LRU buffer
            prefetch_chunk_size: Number of samples dispatched to the thread-pool
                in each prefetch round
            prefetch_max_cached: Maximum number of pre-loaded samples in the
                prefetch cache. Set to 0 to disable prefetching.
            prefetch_num_workers: Number of parallel worker threads inside the
                prefetch buffer for I/O-bound media decoding.  Set to 1 to
                serialise decoding (avoids FFmpeg thread-safety issues).
            multimodal_config: MultimodalConfig for multimodal processing
        """
        # Initialize base class
        super().__init__(
            tokenizer=tokenizer,
            processor=processor,
            max_length=max_length,
            prompt_key=prompt_key,
            multimodal_keys=multimodal_keys,
            label_key=label_key,
            tool_key=tool_key,
            metadata_key=metadata_key,
            system_prompt=system_prompt,
            seed=seed,
            apply_chat_template=apply_chat_template,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            use_audio_in_video=use_audio_in_video,
            multimodal_config=multimodal_config,
            custom_prompt_func=custom_prompt_func,
            teacher_prompt_key=teacher_prompt_key,
            teacher_multimodal_keys=teacher_multimodal_keys,
        )

        # Streaming-specific components
        paths, row_slice = resolve_path_plan(path)
        if len(paths) == 1 and row_slice is None:
            self.reader = StreamingReader(paths[0])
        else:
            self.reader = CompositeStreamingReader(paths, row_slice)
        self.buffer = SampleBuffer(max_size=buffer_size)
        self.index_manager = IndexManager(len(self.reader), seed=seed)

        self._filter_count = 0
        self._total_processed = 0

        # Prefetch buffer for overlapping multimodal I/O with compute.
        # Only enabled when multimodal_keys are set and prefetch_max_cached > 0.
        self._prefetch_buffer: Optional[PrefetchBuffer] = None
        if multimodal_keys and prefetch_max_cached > 0:
            self._prefetch_buffer = PrefetchBuffer(
                process_fn=self._prefetch_process_single,
                chunk_size=prefetch_chunk_size,
                max_cached=prefetch_max_cached,
                num_workers=prefetch_num_workers,
            )
            logger.info(
                f"StreamingDataset: prefetch enabled with "
                f"chunk_size={prefetch_chunk_size}, max_cached={prefetch_max_cached}, "
                f"num_workers={prefetch_num_workers}"
            )
        self._prefetch_hits_log_counter = 0

    def _prefetch_process_single(self, idx: int) -> Optional[Sample]:
        """Process a single index for prefetching.

        Called by the PrefetchBuffer's worker threads.  Each call loads
        a single sample (including heavy video/image I/O) and returns it.

        NOTE: We intentionally do NOT access ``self.buffer`` (SampleBuffer)
        here because SampleBuffer is not thread-safe and these calls run
        in parallel worker threads.
        """
        try:
            raw_data = self.reader[idx]
            sample = self._process_raw_data(raw_data)
            return sample
        except Exception as e:
            logger.warning(f"Prefetch: error processing index {idx}: {e}")
            return None

    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        return len(self.reader)

    def shuffle(self, epoch_id: int) -> None:
        """Shuffle the dataset for a new epoch.

        When prefetch is enabled, passes the **remaining** shuffled index
        sequence (from the current position onward) to the
        ``PrefetchBuffer`` so the background thread starts loading
        immediately — well before ``get_batch`` is called.

        Args:
            epoch_id: Epoch identifier
        """
        self.index_manager.shuffle(epoch_id)
        self.epoch_id = epoch_id
        # Trigger prefetch with the remaining upcoming index order
        if self._prefetch_buffer is not None and self.index_manager.indices is not None:
            remaining = self.index_manager.indices[self.index_manager.position :]
            self._prefetch_buffer.set_index_order(list(remaining))
            logger.info(
                f"Prefetch: triggered for epoch {epoch_id}, "
                f"{len(remaining)} indices remaining (position={self.index_manager.position})"
            )

    def _process_raw_data(self, data: dict) -> Optional[Sample]:
        """Process raw data into a Sample.

        Uses the shared _process_data from BaseDataset.

        Args:
            data: Raw data dict from file

        Returns:
            Sample or None if filtered out
        """
        self._total_processed += 1

        try:
            sample = self._process_data(data)

            # Filter by length if max_length is set
            if self.max_length is not None:
                if not check_sample_length(
                    sample,
                    self.tokenizer,
                    self.processor,
                    self.max_length,
                    apply_chat_template_kwargs=self.apply_chat_template_kwargs,
                ):
                    self._filter_count += 1
                    return None

            return sample

        except Exception as e:
            logger.warning(f"Error processing data: {e}")
            return None

    def get_batch(self, n: int) -> tuple[list[Sample], bool]:
        """Get a batch of n valid samples.

        Automatically skips filtered samples and handles epoch boundaries.

        When prefetch is enabled, indices are consumed **one at a time** so
        that ``IndexManager.position`` stays exactly in sync with the index
        sequence given to ``PrefetchBuffer.set_index_order()``.

        Without prefetch, indices are fetched in small batches for
        efficiency (acceptable since there is no ordering contract to
        honour with a background thread).

        Args:
            n: Number of samples to get

        Returns:
            (samples, crossed_epoch): List of samples and whether an epoch boundary was crossed
        """
        if self._prefetch_buffer is not None:
            return self._get_batch_prefetch(n)
        return self._get_batch_no_prefetch(n)

    def _get_batch_prefetch(self, n: int) -> tuple[list[Sample], bool]:
        """Prefetch-aware path: consume indices one-by-one to stay aligned."""
        samples: list[Sample] = []
        crossed_epoch = False
        max_attempts = n * 10

        for _ in range(max_attempts):
            if len(samples) >= n:
                break

            indices, epoch_crossed = self.index_manager.get_next_indices(1)
            if epoch_crossed and not crossed_epoch:
                crossed_epoch = True
                # IndexManager already shuffled the new epoch internally.
                # Re-trigger prefetch immediately for the remaining indices
                # so subsequent get() calls hit the cache instead of falling
                # back to synchronous loading.
                remaining = self.index_manager.indices[self.index_manager.position :]
                self._prefetch_buffer.set_index_order(list(remaining))
                logger.info(
                    f"Prefetch: epoch crossing detected, re-triggered with "
                    f"{len(remaining)} indices (epoch={self.index_manager.current_epoch})"
                )
            idx = indices[0]

            sample = self._prefetch_buffer.get(idx)

            if sample is None:
                # Prefetch returned None — either the sample was filtered
                # out during prefetch or prefetch failed; skip it.
                continue

            samples.append(sample)

        if len(samples) < n:
            logger.warning(
                f"Could only get {len(samples)}/{n} samples after {max_attempts} attempts. "
                f"Filter rate: {self._filter_count}/{self._total_processed}"
            )

        if self._prefetch_hits_log_counter % 10 == 0:
            logger.info(
                f"Prefetch stats: hit_rate={self._prefetch_buffer.hit_rate:.1%}, "
                f"cache_size={self._prefetch_buffer.cache_size}"
            )
        self._prefetch_hits_log_counter += 1

        return samples, crossed_epoch

    def _get_batch_no_prefetch(self, n: int) -> tuple[list[Sample], bool]:
        """Non-prefetch path: fetch indices in small batches for efficiency."""
        samples: list[Sample] = []
        crossed_epoch = False
        max_attempts = n * 10
        attempts = 0

        while len(samples) < n and attempts < max_attempts:
            need = n - len(samples)
            fetch_size = min(need * 2, 100)

            indices, epoch_crossed = self.index_manager.get_next_indices(fetch_size)
            crossed_epoch = crossed_epoch or epoch_crossed

            for idx in indices:
                if len(samples) >= n:
                    break

                attempts += 1

                sample = self.buffer.get(idx)
                if sample is None:
                    raw_data = self.reader[idx]
                    sample = self._process_raw_data(raw_data)
                    if sample is not None:
                        self.buffer.put(idx, sample)

                if sample is not None:
                    samples.append(sample)

        if len(samples) < n:
            logger.warning(
                f"Could only get {len(samples)}/{n} samples after {attempts} attempts. "
                f"Filter rate: {self._filter_count}/{self._total_processed}"
            )

        return samples, crossed_epoch

    def __getitem__(self, idx: int) -> Optional[Sample]:
        """Get a single sample by index.

        Args:
            idx: Sample index

        Returns:
            Sample or None if filtered
        """
        sample = self.buffer.get(idx)
        if sample is not None:
            return sample

        raw_data = self.reader[idx]
        sample = self._process_raw_data(raw_data)

        if sample is not None:
            self.buffer.put(idx, sample)

        return sample

    @property
    def samples(self) -> "StreamingDatasetSamplesProxy":
        """Provide a proxy for samples access.

        This allows code that uses `dataset.samples[start:end]` to work, though
        it's less efficient than using get_batch().
        """
        return StreamingDatasetSamplesProxy(self)

    def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {
            **self.index_manager.get_state(),
            "filter_count": self._filter_count,
            "total_processed": self._total_processed,
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint."""
        self.index_manager.load_state(state)
        self._filter_count = state.get("filter_count", 0)
        self._total_processed = state.get("total_processed", 0)

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        return {
            "total_size": len(self),
            "buffer_size": len(self.buffer),
            "buffer_hit_rate": self.buffer.hit_rate,
            "filter_count": self._filter_count,
            "total_processed": self._total_processed,
            "current_epoch": self.index_manager.current_epoch,
            "current_position": self.index_manager.position,
        }


class StreamingDatasetSamplesProxy:
    """Proxy class to support `dataset.samples[start:end]` access pattern.

    This provides compatibility with code that expects the original Dataset
    interface, but with streaming behavior.
    """

    def __init__(self, dataset: StreamingDataset):
        self.dataset = dataset

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop or len(self.dataset)
            step = key.step or 1

            indices = list(range(start, stop, step))
            samples = []

            for idx in indices:
                sample = self.dataset[idx]
                if sample is not None:
                    samples.append(sample)

            return samples
        else:
            return self.dataset[key]

    def __len__(self):
        return len(self.dataset)
