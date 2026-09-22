# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Monkey-patch for Megatron checkpoint writing to avoid fork() with CUDA.

``FileSystemWriterAsync.write_preloaded_data_multiproc`` uses
``mp.get_context("fork")`` to fork child processes for parallel disk writes.
``TemporalAsyncCaller.schedule_async_call`` also uses ``mp.get_context("fork")``
to fork the main training process for background checkpoint writes.

Forking a process with active CUDA contexts is unsafe and can cause SIGSEGV
(typically on the **second** checkpoint save when cached strategies skip NCCL
synchronisation that would otherwise settle the CUDA state).

This module patches both code paths to avoid fork when CUDA is initialised:

1. ``write_preloaded_data_multiproc`` — uses a ``ThreadPoolExecutor`` for
   parallel disk writes instead of forking child processes.  By this point
   all tensors have been staged to CPU, so the writes are pure I/O that
   releases the GIL — true parallelism is preserved without fork().
2. ``TemporalAsyncCaller.schedule_async_call`` — uses a background
   ``threading.Thread`` instead of ``mp.Process`` with ``fork`` context.

Usage::

    from relax.utils.checkpoint_write_patch import patch_checkpoint_write

    patch_checkpoint_write()  # idempotent, safe to call multiple times
"""

import inspect
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import torch

from relax.utils import device as device_utils
from relax.utils.device import device_module
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_patched = False


# ---------------------------------------------------------------------------
# Thread wrapper that mimics mp.Process interface for TemporalAsyncCaller
# ---------------------------------------------------------------------------
class _KillableThread(threading.Thread):
    """A ``threading.Thread`` subclass that exposes a ``kill()`` method.

    ``TemporalAsyncCaller.close(abort=True)`` calls ``self.process.kill()``.
    Threads cannot truly be killed, but we set a flag and let ``join()`` handle
    the rest so the caller doesn't crash with ``AttributeError``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon = True  # don't block interpreter exit

    def kill(self):
        """No-op for threads — included for mp.Process API compatibility."""
        pass


# ---------------------------------------------------------------------------
# Threaded bucket writer (replaces fork-based parallel writer)
# ---------------------------------------------------------------------------
def _write_single_bucket(transform_list, use_msc, bucket_index, write_bucket):
    """Write a single checkpoint bucket to disk.

    This function is submitted to a ``ThreadPoolExecutor`` and runs in a
    worker thread.  All data has already been staged to CPU, so the work
    is pure I/O (``open`` / ``write`` / ``fsync``) which releases the GIL.

    Args:
        transform_list: list of storage writer transforms.
        use_msc: whether to use multistorageclient for I/O.
        bucket_index: integer index of this bucket (for result mapping).
        write_bucket: a ``WriteBucket`` tuple ``(file_name, storage_key, (bytes_data, tensor_data))``.

    Returns:
        Tuple of ``(bucket_index, list_of_WriteResult)``.

    Raises:
        RuntimeError: if the write fails.
    """
    from torch.distributed.checkpoint.filesystem import _write_item

    file_name, storage_key, (bytes_data, tensor_data) = write_bucket
    extra_kwargs = {}
    if "serialization_format" in inspect.signature(_write_item).parameters:
        from torch.distributed.checkpoint.filesystem import SerializationFormat

        extra_kwargs["serialization_format"] = SerializationFormat.TORCH_SAVE
    if use_msc:
        import multistorageclient as msc

        open_file = msc.open
    else:
        open_file = open

    local_results = []
    with open_file(file_name, "wb") as stream:
        for write_item, data in bytes_data:
            local_results.append(
                _write_item(
                    *transform_list,
                    stream,
                    data,
                    write_item,
                    storage_key,
                    **extra_kwargs,
                )
            )
        for write_item, tensor in tensor_data:
            assert tensor.is_cpu
            local_results.append(
                _write_item(
                    *transform_list,
                    stream,
                    tensor,
                    write_item,
                    storage_key,
                    **extra_kwargs,
                )
            )
        if not use_msc:
            os.fsync(stream.fileno())
        else:
            stream.fsync()
    return bucket_index, local_results


def _write_buckets_threaded(transform_list, use_msc, write_buckets):
    """Write checkpoint buckets in parallel using threads (no fork).

    Replicates the parallelism of the original fork-based
    ``FileSystemWriterAsync.write_preloaded_data_multiproc`` but uses a
    ``ThreadPoolExecutor`` instead of ``mp.get_context("fork")``.

    Since all tensors have been pre-staged to CPU, the write work is pure
    disk I/O which releases the GIL — threads achieve real parallelism here.

    Args:
        transform_list: list of storage writer transforms.
        use_msc: whether to use multistorageclient for I/O.
        write_buckets: list of ``WriteBucket`` tuples to persist.

    Returns:
        dict mapping bucket index to list of ``WriteResult``, or an Exception.
    """
    write_results: dict = {}
    num_buckets = len(write_buckets)

    if num_buckets == 0:
        return write_results

    # Single bucket — no need for thread overhead
    if num_buckets == 1:
        try:
            idx, results = _write_single_bucket(transform_list, use_msc, 0, write_buckets[0])
            write_results[idx] = results
        except Exception as e:
            return RuntimeError(f"Threaded checkpoint write failed for bucket 0: {e}")
        return write_results

    try:
        with ThreadPoolExecutor(max_workers=num_buckets) as executor:
            futures = {
                executor.submit(_write_single_bucket, transform_list, use_msc, i, wb): i
                for i, wb in enumerate(write_buckets)
            }
            for future in as_completed(futures):
                bucket_idx = futures[future]
                try:
                    idx, results = future.result()
                    write_results[idx] = results
                except Exception as e:
                    return RuntimeError(f"Threaded checkpoint write failed for bucket {bucket_idx}: {e}")
    except Exception as e:
        return RuntimeError(f"ThreadPoolExecutor failed during checkpoint write: {e}")

    return write_results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def patch_checkpoint_write():
    """Monkey-patch Megatron checkpoint writing to avoid ``fork()`` with CUDA.

    Two patches are applied:

    1. ``FileSystemWriterAsync.write_preloaded_data_multiproc`` — when CUDA is
       initialised, uses a ``ThreadPoolExecutor`` for parallel disk writes
       instead of forking child processes.  All tensors have been pre-staged
       to CPU, so disk I/O releases the GIL and threads achieve real
       parallelism — no performance regression vs the original fork path.
    2. ``TemporalAsyncCaller.schedule_async_call`` — when CUDA is initialised,
       uses a background ``threading.Thread`` instead of ``mp.Process`` with
       ``fork`` context.  By this point ``preload_fn`` has already copied all
       tensors to CPU, so the async function only performs disk I/O and does
       not need GPU access.

    This function is idempotent — calling it multiple times is safe.
    """
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

    global _patched
    # NOTE(wuhuan): the latest Megatron-LM of 20260506 use write_preloaded_data_multithread instead of
    # write_preloaded_data_multiproc, which has solved this issue.
    can_patch = hasattr(FileSystemWriterAsync, "write_preloaded_data_multiproc")
    if _patched or not can_patch:
        return

    _patch_write_preloaded_data_multiproc()
    _patch_temporal_async_caller()

    _patched = True
    logger.info(
        "Patched FileSystemWriterAsync.write_preloaded_data_multiproc "
        "and TemporalAsyncCaller.schedule_async_call to avoid fork() with CUDA"
    )


def _patch_write_preloaded_data_multiproc():
    """Patch ``write_preloaded_data_multiproc`` to use threaded parallel
    writes."""
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

    # In Python 3, accessing a @staticmethod via the class returns the
    # underlying function directly (no descriptor wrapper), so we can
    # call ``_original_write(...)`` without ``.__func__``.
    _original_write = FileSystemWriterAsync.write_preloaded_data_multiproc

    @staticmethod
    def _patched_write_preloaded_data_multiproc(
        transform_list,
        use_msc,
        rank,
        write_buckets,
        global_results_queue,
    ):
        _logger = logging.getLogger(__name__)
        w_start = time()

        # When CUDA is initialised, forking child processes is unsafe and can
        # cause SIGSEGV.  Use threaded parallel writes instead — all tensors
        # are already on CPU so the I/O releases the GIL and threads achieve
        # real parallelism without duplicating the CUDA context.
        cuda_initialised = device_utils.is_available() and device_module.is_initialized()
        if cuda_initialised:
            _logger.debug(
                f"rank: {rank}, device initialised – using threaded parallel "
                f"(no-fork) checkpoint write for {len(write_buckets)} buckets"
            )
            write_results_or_exc = _write_buckets_threaded(transform_list, use_msc, write_buckets)
            global_results_queue.put(write_results_or_exc)
            w_end = time()
            _logger.debug(f"{w_end}, rank: {rank}, write(threaded,no-fork): {w_end - w_start}")
            return

        # CUDA not initialised — safe to use the original fork-based path.
        return _original_write(transform_list, use_msc, rank, write_buckets, global_results_queue)

    FileSystemWriterAsync.write_preloaded_data_multiproc = _patched_write_preloaded_data_multiproc


def _patch_temporal_async_caller():
    """Patch ``TemporalAsyncCaller.schedule_async_call`` to use threads."""
    from megatron.core.dist_checkpointing.strategies.async_utils import (
        TemporalAsyncCaller,
        _disable_gc,
    )

    _original_schedule = TemporalAsyncCaller.schedule_async_call

    @_disable_gc()
    def _patched_schedule_async_call(self, async_req):
        """Replacement for ``TemporalAsyncCaller.schedule_async_call``.

        When CUDA is initialised, uses ``threading.Thread`` instead of
        ``mp.get_context('fork').Process`` to run the async checkpoint writer
        in the background.  By this point, ``preload_fn`` has already staged
        all tensors to CPU memory, so the async function only performs disk
        I/O.
        """
        if async_req.async_fn is None:
            return  # nothing to do

        cuda_initialised = device_utils.is_available() and device_module.is_initialized()
        if not cuda_initialised:
            # CUDA not initialised — safe to use the original fork path.
            return _original_schedule(self, async_req)

        # --- CUDA initialised path: use thread instead of fork ---
        _logger = logging.getLogger(__name__)

        async_fn_args = list(async_req.async_fn_args)
        if async_req.preload_fn:
            # Stage GPU tensors to CPU.  This is done in the main process
            # before spawning the background writer.
            async_fn_args[1] = async_req.preload_fn()

        rank = torch.distributed.get_rank()
        start_sync = time()
        device_module.synchronize()
        end_sync = time()
        _logger.debug(f"rank: {rank}, takes {end_sync - start_sync} to finish D2H ")

        self.start_time = time()
        # Use a thread instead of fork to avoid SIGSEGV from
        # duplicating the CUDA context.
        self.process = _KillableThread(
            target=async_req.async_fn,
            args=tuple(async_fn_args),
            kwargs=async_req.async_fn_kwargs,
        )
        self.process.start()
        _logger.debug(f"rank: {rank}, takes {time() - self.start_time} to schedule async ckpt (thread, no-fork)")

    TemporalAsyncCaller.schedule_async_call = _patched_schedule_async_call
