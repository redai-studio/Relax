# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in hybrid snapshots for Megatron torch_dist checkpoint saves."""

import io
from argparse import Namespace
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.planner import SavePlan, SavePlanner, WriteItem, WriteItemType

from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger


if TYPE_CHECKING:
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest

logger = get_logger(__name__)

_GIB = 2**30
_GPU_SNAPSHOT_BYTES = 32 * _GIB
_GPU_HEADROOM_BYTES = 16 * _GIB
_GPU_BLOCK_BYTES = 2 * _GIB
_WRITE_WINDOW_BYTES = _GIB

_WriteTensor = tuple[WriteItem, torch.Tensor]


def _snapshot_tensors(tensors: list[_WriteTensor], device: torch.device) -> list[_WriteTensor]:
    """Freeze each planned tensor before returning control to training."""
    snapshot: list[_WriteTensor] = []
    gpu_items: dict[torch.dtype, list[int]] = defaultdict(list)
    gpu_bytes = 0
    cpu_bytes = 0
    cuda_copies = False
    gpu_limit = 0
    if any(tensor.is_cuda for _, tensor in tensors):
        free, _ = torch.cuda.mem_get_info(device)
        gpu_limit = min(_GPU_SNAPSHOT_BYTES, max(0, free - _GPU_HEADROOM_BYTES))
    try:
        for item, source in tensors:
            source = source.detach()
            if source.is_cuda:
                if source.device != device:
                    raise ValueError("Hybrid async checkpoint expects one CUDA device per training rank")
                if "dequantize" in type(source).__dict__:
                    # Dequantization may enqueue work before the snapshot allocation.
                    cuda_copies = True
                    source = source.dequantize()
                if gpu_bytes + source.nbytes <= gpu_limit:
                    gpu_items[source.dtype].append(len(snapshot))
                    gpu_bytes += source.nbytes
                    snapshot.append((item, source))
                    continue
                host = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
                cuda_copies = True
                host.copy_(source, non_blocking=True)
            else:
                if source.device.type != "cpu":
                    raise ValueError(f"Unsupported checkpoint tensor device {source.device}")
                host = source.clone(memory_format=torch.contiguous_format)
            snapshot.append((item, host))
            cpu_bytes += host.nbytes
        for dtype, indices in gpu_items.items():
            block: list[int] = []
            block_bytes = 0
            for index in indices:
                size = snapshot[index][1].nbytes
                if block and block_bytes + size > _GPU_BLOCK_BYTES:
                    cuda_copies = True
                    _pack_gpu_block(snapshot, block, dtype, device)
                    block = []
                    block_bytes = 0
                block.append(index)
                block_bytes += size
            cuda_copies = True
            _pack_gpu_block(snapshot, block, dtype, device)
    finally:
        if cuda_copies:
            # Complete all copies, including those submitted before a failure.
            torch.cuda.synchronize(device)
    logger.info("Frozen checkpoint snapshot with %d GPU bytes and %d CPU bytes", gpu_bytes, cpu_bytes)
    return snapshot


def _pack_gpu_block(tensors: list[_WriteTensor], indices: list[int], dtype: torch.dtype, device: torch.device) -> None:
    block = torch.empty(sum(tensors[index][1].numel() for index in indices), dtype=dtype, device=device)
    offset = 0
    for index in indices:
        item, source = tensors[index]
        end = offset + source.numel()
        view = block[offset:end].view(source.shape)
        view.copy_(source, non_blocking=True)
        tensors[index] = (item, view)
        offset = end


def _any_rank(flag: bool, group: dist.ProcessGroup) -> bool:
    status = torch.tensor(int(flag), device="cpu")
    dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
    return bool(status)


class _ThreadCaller:
    """Run one NVRx write request off the training thread."""

    def __init__(self, group: dist.ProcessGroup) -> None:
        self._group = group
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="relax-async-checkpoint")
        self._future: Future[None] | None = None

    def schedule_async_call(self, request: "AsyncRequest") -> None:
        if self._future is not None or request.preload_fn is not None:
            raise RuntimeError("Hybrid async checkpoint requires a frozen snapshot and one in-flight request")
        if request.async_fn is not None:
            self._future = self._executor.submit(
                request.async_fn, *request.async_fn_args, **(request.async_fn_kwargs or {})
            )

    def is_current_async_call_done(self, blocking: bool = False, no_dist: bool = False) -> bool:
        if blocking and self._future is not None:
            self._future.result()
        pending = self._future is not None and not self._future.done()
        if not no_dist:
            pending = _any_rank(pending, self._group)
        if not pending:
            self._future = None
        return not pending

    def close(self, abort: bool = False) -> None:
        self._executor.shutdown(wait=not abort, cancel_futures=abort)


def _write_snapshot(
    write: Callable[..., None],
    transforms: list[Any],
    rank: int,
    buckets: list[Any],
    results: Queue,
    copy_stream: torch.cuda.Stream,
) -> None:
    try:
        write(transforms, False, rank, buckets, results, _WRITE_WINDOW_BYTES, copy_stream=copy_stream)
    except Exception as exc:
        # NVRx collects per-rank write errors during distributed finalization.
        results.put(exc)
    finally:
        # The request and finalizer both hold the bucket list until commit.
        buckets.clear()


def install(args: Namespace) -> None:
    """Install with --custom-megatron-init-path on every training rank."""
    if not (
        args.ckpt_format == "torch_dist"
        and args.async_save
        and args.use_persistent_ckpt_worker
        and args.async_strategy == "nvrx"
        and not getattr(args, "async_ckpt_use_cpu_shm", False)
    ):
        raise ValueError("Hybrid async checkpoint requires torch_dist, NVRx async save, and CPU SHM disabled")
    if getattr(args, "offload_train", False):
        raise ValueError(
            "Hybrid async checkpoint holds GPU memory until the save finishes and requires offload_train=False"
        )

    from megatron.core.dist_checkpointing.strategies import torch as megatron_dcp
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync as MCoreWriter
    from megatron.training import async_utils
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue
    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import DEFAULT_SUFFIX, FileSystemWriterAsync

    if async_utils._async_calls_queue is not None:
        raise RuntimeError("Install hybrid async checkpoint before the Megatron save queue is initialized")

    group = get_gloo_group()
    rank = dist.get_rank(group=group)
    device = torch.device("cuda", torch.cuda.current_device())
    copy_stream = torch.cuda.Stream(device=device)
    bounded_write = MCoreWriter.write_data_synchronously
    original_strategy = megatron_dcp.get_async_strategy

    class HybridWriter(FileSystemWriterAsync):
        def prepare_write_data(self, plan: SavePlan, planner: SavePlanner) -> None:
            error = None
            try:
                self._prepare_snapshot(plan, planner)
            except Exception as exc:
                error = exc
            if _any_rank(error is not None, group):
                self._snapshot_buckets = []
                if error is not None:
                    raise error
                raise RuntimeError("Hybrid async checkpoint could not prepare a snapshot on another rank")

        def _prepare_snapshot(self, plan: SavePlan, planner: SavePlanner) -> None:
            if (
                self.use_msc
                or self.use_cached_data_structure
                or self.use_cpu_shm_for_gpu_tensors
                or self.separation_hint
                or self.thread_count != 1
                or self.is_multi_proc_io
            ):
                raise ValueError(
                    "Hybrid async checkpoint requires one standard filesystem writer per rank and no plan cache"
                )
            self.has_data_to_write = bool(plan.items)
            if not self.has_data_to_write:
                return

            byte_items: list[tuple[WriteItem, io.BytesIO]] = []
            tensor_items: list[_WriteTensor] = []
            for item in plan.items:
                data = planner.resolve_data(item)
                if item.type == WriteItemType.BYTE_IO:
                    assert isinstance(data, io.BytesIO)
                    byte_items.append((item, data))
                else:
                    assert isinstance(data, torch.Tensor)
                    tensor_items.append((item, data))
            storage_key = f"{plan.storage_data.prefix}0{DEFAULT_SUFFIX}"
            path = Path(self.checkpoint_dir) / storage_key
            self._snapshot_buckets = [(path, storage_key, (byte_items, _snapshot_tensors(tensor_items, device)))]
            self.results_queue = Queue()

        def get_save_function_and_args(self) -> tuple[Callable[..., None] | None, None, list[Any]]:
            if not self.has_data_to_write:
                return None, None, []
            transforms = [self.transforms] if hasattr(self, "transforms") else []
            write = partial(_write_snapshot, bounded_write, transforms, copy_stream=copy_stream)
            return write, None, [rank, self._snapshot_buckets, self.results_queue]

    class HybridAsyncCallsQueue(AsyncCallsQueue):
        def _get_async_caller(self) -> _ThreadCaller:
            if self.persistent_caller is None:
                self.persistent_caller = _ThreadCaller(group)
            return self.persistent_caller

    def get_async_strategy(async_strategy: str = "nvrx", module: str | None = None) -> tuple[str, Any]:
        strategy, modules = original_strategy(async_strategy)
        if strategy == "nvrx":
            modules = {**modules, "FileSystemWriterAsync": HybridWriter, "AsyncCallsQueue": HybridAsyncCallsQueue}
        return strategy, modules if not module else modules[module]

    megatron_dcp.get_async_strategy = get_async_strategy
    async_utils.get_async_strategy = get_async_strategy
