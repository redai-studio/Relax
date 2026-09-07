# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import logging
import pickle
import time
from argparse import Namespace
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core import mpu
from tensordict import TensorDict
from transfer_queue.dataloader.streaming_dataloader import StreamingDataLoader
from transfer_queue.dataloader.streaming_dataset import StreamingDataset

from relax.utils import device as device_utils
from relax.utils.env import Envs
from relax.utils.opd.opd_utils import iter_opd_cp_float_fields
from relax.utils.timer import timer


logger = logging.getLogger(__name__)

# Throttle counter for the opt-in pickle-size diagnostic.  See
# ``_maybe_log_tgd_pickle_diag`` below for usage.
_tgd_diag_call_count = 0

# Same-purpose throttle for the per_rank_fetch byte-size diagnostic; kept
# separate so the two paths' counters don't interfere when toggling modes.
_per_rank_fetch_diag_call_count = 0


def _maybe_log_per_rank_fetch_diag(rollout_data: list) -> None:
    """Cheap payload-size diagnostic for the ``per_rank_fetch`` path.

    Unlike ``_maybe_log_tgd_pickle_diag`` this never calls ``pickle.dumps``
    (which would re-introduce the multi-second cost we use this path to
    avoid).  Instead it sums ``element_size * numel`` over every tensor it
    can reach so the operator can see how much data each rank just pulled
    from TQ and judge whether SimpleStorageUnit bandwidth is the new
    bottleneck.

    Gated by env var ``RELAX_TGD_PROFILE`` (default ``0``); same throttle
    schedule (first 3 calls then every ``RELAX_TGD_PROFILE_EVERY``).  Only
    logs from global rank 0 to avoid N-rank-duplicated noise — payload size
    is identical across ranks in this mode (TQ sampler cache guarantees
    byte-identical sample ids per dp_rank).
    """
    if rollout_data[0] is None:
        return
    if not Envs.RELAX_TGD_PROFILE:
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    global _per_rank_fetch_diag_call_count
    _per_rank_fetch_diag_call_count += 1
    every = Envs.RELAX_TGD_PROFILE_EVERY
    if _per_rank_fetch_diag_call_count > 3 and _per_rank_fetch_diag_call_count % every != 0:
        return

    def _tensor_bytes(obj) -> int:
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.numel()
        if isinstance(obj, dict):
            return sum(_tensor_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_tensor_bytes(v) for v in obj)
        return 0

    td = rollout_data[0]
    per_field: list[tuple[str, float]] = []
    if isinstance(td, TensorDict):
        for k in td.keys():
            try:
                size_mb = _tensor_bytes(td.get(k)) / 1024 / 1024
            except Exception:  # noqa: BLE001
                size_mb = -1.0
            per_field.append((k, size_mb))
    else:
        per_field.append((f"<{type(td).__name__}>", _tensor_bytes(td) / 1024 / 1024))
    per_field.sort(key=lambda x: x[1], reverse=True)
    total_mb = sum(mb for _, mb in per_field if mb > 0)
    top = ", ".join(f"{k}={mb:.1f}MB" for k, mb in per_field[:5])

    logger.info(
        "[per_rank_fetch_diag] call=%d payload_total=%.1fMB top_fields: %s",
        _per_rank_fetch_diag_call_count,
        total_mb,
        top,
    )


def _agree_on_fetch(
    rollout_data: list,
    fetch_once: Callable[[], list],
    include_pipeline: bool,
    context: str,
) -> list:
    """Make model-parallel ranks for one logical DP slot agree on data state.

    In ``per_rank_fetch`` every TP/CP/PP rank calls ``get_meta`` itself. Ranks
    arriving before the producer fills ``ready_indexes`` can get empty meta,
    while later ranks get real samples and warm the TQ sampler replay state for
    this ``batch_index``. Empty ranks can then re-fetch the same logical request
    and converge.

    The reduction sums ``[has_data, not has_data]`` over TP x CP, plus PP when
    pipeline stages are in lockstep: ``missing == 0`` means everyone holds the
    batch, while ``got == 0`` is a normal empty poll. We intentionally do not
    synchronize the data-parallel group because different DP ranks consume
    different TQ slices and may legitimately arrive at different times.
    """
    groups = [mpu.get_tensor_and_context_parallel_group()]
    if include_pipeline:
        groups.append(mpu.get_pipeline_model_parallel_group())
    max_retries = Envs.RELAX_FETCH_SPLIT_MAX_RETRIES

    for attempt in range(max_retries + 1):
        has_data = rollout_data[1] is not None and rollout_data[1].size > 0
        tally = torch.tensor(
            [1, 0] if has_data else [0, 1],
            dtype=torch.int32,
            device=device_utils.make_current_torch_device(),
        )
        for group in groups:
            dist.all_reduce(tally, op=dist.ReduceOp.SUM, group=group)
        got, missing = tally.tolist()

        if not missing or not got:
            return rollout_data
        if not has_data:
            logger.warning(
                "[per_rank_fetch] %s: %d/%d ranks got data, re-fetching (attempt %d)",
                context,
                got,
                got + missing,
                attempt + 1,
            )
            # Ranks holding data block in the next agreement all-reduce while
            # empty ranks repeat the rank-local TQ RPC.
            time.sleep(0.25)
            rollout_data = fetch_once()

    raise RuntimeError(
        f"[per_rank_fetch] {context}: model-parallel replica still split after {max_retries} retries "
        f"({got}/{got + missing} ranks have data); continuing would desync the next collective. "
        f"Raise RELAX_FETCH_SPLIT_MAX_RETRIES if the producer is legitimately this slow."
    )


def _maybe_log_tgd_pickle_diag(rollout_data: list, should_fetch: bool) -> None:
    """Opt-in diagnostic: log pickle cost and per-field byte size on the
    tp_rank-0 fetcher so we can see how much of ``broadcast_object_list`` is
    pickle vs NCCL, and which TensorDict field dominates the payload.

    Gated by env var ``RELAX_TGD_PROFILE=1``.  Logs the first 3 calls then
    every ``RELAX_TGD_PROFILE_EVERY`` (default 50) calls thereafter.  Only
    fires on the rank that actually holds non-empty data — empty-poll cycles
    (``batch_meta.size == 0`` → ``rollout_data[0] is None``) are skipped so the
    log isn't drowned by hundreds of empty polls per second.
    """
    if not should_fetch:
        return
    if rollout_data[0] is None:
        return
    if not Envs.RELAX_TGD_PROFILE:
        return

    global _tgd_diag_call_count
    _tgd_diag_call_count += 1
    every = Envs.RELAX_TGD_PROFILE_EVERY
    if _tgd_diag_call_count > 3 and _tgd_diag_call_count % every != 0:
        return

    td = rollout_data[0]
    t0 = time.perf_counter()
    full_bytes = pickle.dumps(rollout_data, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_ms = (time.perf_counter() - t0) * 1000.0
    pickle_mb = len(full_bytes) / 1024 / 1024

    if isinstance(td, TensorDict):
        per_field: list[tuple[str, float]] = []
        for k in td.keys():
            try:
                size_mb = len(pickle.dumps(td.get(k), protocol=pickle.HIGHEST_PROTOCOL)) / 1024 / 1024
            except Exception:  # noqa: BLE001
                size_mb = -1.0
            per_field.append((k, size_mb))
        per_field.sort(key=lambda x: x[1], reverse=True)
        top = ", ".join(f"{k}={mb:.1f}MB" for k, mb in per_field[:5])
    else:
        top = f"<not-a-tensordict: {type(td).__name__}>"

    logger.info(
        "[tgd_profile] call=%d pickle_total=%.1fMB pickle_ms=%.1f top_fields: %s",
        _tgd_diag_call_count,
        pickle_mb,
        pickle_ms,
        top,
    )


def create_stream_dataloader(
    args: Namespace,
    rollout_id: int,
    task_name: str,
    data_fields: list,
    dp_rank: int,
):
    """Create a streaming dataloader and micro-batch plan for a rollout.

    This function constructs a `StreamingDataset` and wraps it with a
    `StreamingDataLoader`. It then builds a list of dataloader iterators
    (one per virtual pipeline parallel stage) and a list describing the
    number of microbatches to use for each step in the rollout.

    Args:
        args (Namespace): Configuration / runtime arguments. Expected to
            contain `tq_config`, `micro_batch_size`, `n_samples_per_prompt`,
            `rollout_batch_size`, and `global_batch_size` attributes.
        rollout_id (int): Identifier for the current rollout partition.
        task_name (str): Name of the task to fetch from the transfer queue.
        data_fields (list): List of data field names to request from the
            transfer queue.
        dp_rank (int): Data-parallel rank (used by the dataset/queue).

    Returns:
        Tuple[List[StreamingDataLoader], List[int]]: A tuple where the first
        element is a list of `StreamingDataLoader` objects (one per virtual
        pipeline stage) and the second element is a list with the number of
        microbatches for each step in the rollout.
    """

    # Choose the appropriate fetch function based on fully_async mode
    # Use partial to bind the broadcast_pp parameter
    # broadcast_pp is the inverse of fully_async: True for colocate, False for fully async
    fetch_batch_fn = partial(
        get_data_from_transfer_queue, args=args, broadcast_pp=not getattr(args, "fully_async", False)
    )
    dataset = StreamingDataset(
        config=args.tq_config,
        batch_size=args.micro_batch_size * args.n_samples_per_prompt,
        micro_batch_size=args.micro_batch_size,
        data_fields=data_fields,
        partition_id=f"train_{rollout_id}",
        task_name=task_name,
        dp_rank=dp_rank,
        fetch_batch_fn=fetch_batch_fn,
        process_batch_fn=split_dict,
    )

    dataloader = StreamingDataLoader(dataset)

    # Virtual pipeline parallel size may be None when not using vpp.
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1

    # Provide one iterator per virtual pipeline stage. Each element is the
    # same dataloader instance; downstream code uses one per stage.
    data_iterator = [dataloader for _ in range(vpp_size)]

    # Compute how many forward steps (global batch splits) occur per rollout,
    # then compute the number of microbatches for each of those steps.
    num_steps_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size

    num_microbatches = [
        args.global_batch_size
        // mpu.get_data_parallel_world_size(with_context_parallel=False)
        // args.micro_batch_size
        for _ in range(num_steps_per_rollout)
    ]

    return data_iterator, num_microbatches


class MicroBatchListIterator:
    """Thin iterator wrapping a fixed list of pre-packed (batch_dict,
    batch_meta) tuples.

    Used by the fully-async + dynamic-batch path in `actor.train_async` after
    draining the per-DP bucket from `TokenBudgetPackedDataset`:

    1. Drain produces `K_local` packed mbs.
    2. Cross-DP `all_reduce(MAX)` gives `K_global`.
    3. The bucket is padded with `K_global - K_local` copies of the shortest
       real mb, marked with `__is_dummy__=True`.
    4. The full list is wrapped in this iterator and passed to `train()`
       (which Megatron iterates via `next(iter)` per micro-batch).

    Interface intentionally mirrors `StreamingDataLoader` (`__iter__`/`__next__`
    return `(batch_dict, batch_meta)`, plus `get_buffer()` for logging and a
    no-op `step()` for API symmetry).
    """

    def __init__(
        self,
        mbs: List[Tuple[Dict[str, Any], Any]],
        dummy_after: int | None = None,
        loss_scale: float | None = None,
    ) -> None:
        self.mbs = mbs
        self.dummy_after = dummy_after
        self.loss_scale = loss_scale
        self.offset = 0

    def __iter__(self) -> "MicroBatchListIterator":
        self.offset = 0
        return self

    def __next__(self) -> Tuple[Dict[str, Any], Any]:
        if self.offset >= len(self.mbs):
            raise StopIteration
        batch, meta = self.mbs[self.offset]
        # Shallow-copy so injected scalar fields don't mutate the cached entry
        # (the same `mbs` list may be re-iterated via __iter__ → reset offset).
        out = dict(batch)
        if self.loss_scale is not None:
            out["__loss_scale__"] = self.loss_scale
        if self.dummy_after is not None and self.offset >= self.dummy_after:
            out["__is_dummy__"] = True
        self.offset += 1
        return out, meta

    def get_buffer(self) -> List[Tuple[Dict[str, Any], Any]]:
        """Return the full mb list — used by actor for end-of-rollout
        logging."""
        return self.mbs

    def step(self, partition_id: str) -> None:  # noqa: ARG002
        """API parity with `StreamingDataLoader.step`; no-op for this iterator
        (a fresh `MicroBatchListIterator` is constructed per rollout)."""
        return


def split_dict(data_dict: Dict[str, Any], batch_meta, micro_batch_size: int) -> List[Tuple[Dict[str, Any], Any]]:
    """Split a batched dictionary into a list of smaller micro-batch
    dictionaries.

    The function slices each tensor or list in `data_dict` along the batch
    dimension (dimension 0) into chunks of size `micro_batch_size`. The
    corresponding `batch_meta` is also split into matching chunks via
    `batch_meta.chunk(...)` and paired with each data chunk.

    Args:
        data_dict (Dict[str, Any]): Mapping from field name to batched value.
            All values must share the same batch size in dimension 0.
        batch_meta: An auxiliary object describing the batch (must have a
            `.size` attribute and a `.chunk(n)` method that returns a list of
            `n` metadata pieces matching the data chunks).
        micro_batch_size (int): Desired size for each micro-batch. The last
            chunk may be smaller if `batch_meta.size` is not divisible by
            `micro_batch_size`.

    Returns:
        List[Tuple[Dict[str, Any], Any]]: A list of tuples where each tuple
        contains (chunked_data_dict, chunked_batch_meta).

    Raises:
        ValueError: If `micro_batch_size` is not positive.
    """

    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    total_size = batch_meta.size
    num_chunks = (total_size + micro_batch_size - 1) // micro_batch_size

    result: List[Tuple[Dict[str, Any], Any]] = []
    batch_meta_list: List = batch_meta.chunk(num_chunks)
    for i in range(num_chunks):
        start = i * micro_batch_size
        end = start + micro_batch_size
        chunk = {key: value[start:end] for key, value in data_dict.items()}
        result.append((chunk, batch_meta_list[i]))

    return result


def _broadcast_routed_experts(
    values: "torch.Tensor | None",
    offsets: "torch.Tensor | None",
    is_src: bool,
    cuda_dev: torch.device,
    broadcast_pp: bool,
    keep_on_gpu: bool = False,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Broadcast rollout_routed_experts tensors via NCCL dist.broadcast.

    On the source rank (*is_src* = True), *values* and *offsets* are the
    NestedTensor jagged internals.  On non-source ranks they are None and
    will be allocated here.

    Broadcast order: CP → PP → TP.  Only (TP=0, PP=0, CP=0) holds the
    source data (``should_fetch`` requires all three ranks to be 0).
    The CP step fans data to all CP partners of (TP=0, PP=0), then the
    PP step (among tp_rank==0 peers) fans to all PP stages, and finally
    the TP step fans from tp_rank==0 to the remaining TP ranks.

    Using ``dist.broadcast`` on contiguous GPU tensors is orders of magnitude
    faster than ``broadcast_object_list`` which pickles everything (~14 s for
    377 MB vs sub-second via NCCL).
    """

    def _bcast_tensor(tensor, is_sender, dtype):
        """Broadcast a tensor (any shape) across CP, PP, then TP groups.

        Order: CP first, then PP (among tp_rank==0), then TP.
        Only (tp_rank==0, pp_rank==0, cp_rank==0) has the data initially.
        """

        def _bcast_unknown_shape(t, has_data, src_global, group):
            """Broadcast a tensor whose shape is unknown to receivers.

            Broadcasts ndim -> shape -> data in three NCCL calls. Returns the
            broadcast tensor on *cuda_dev* with *dtype*.
            """
            if has_data and t is not None:
                ndim_t = torch.tensor([t.ndim], dtype=torch.long, device=cuda_dev)
            else:
                ndim_t = torch.tensor([0], dtype=torch.long, device=cuda_dev)
            dist.broadcast(ndim_t, src=src_global, group=group)
            ndim = ndim_t.item()

            if has_data and t is not None:
                shape_t = torch.tensor(list(t.shape), dtype=torch.long, device=cuda_dev)
            else:
                shape_t = torch.empty(ndim, dtype=torch.long, device=cuda_dev)
            dist.broadcast(shape_t, src=src_global, group=group)
            shape = torch.Size(shape_t.tolist())

            if has_data and t is not None:
                buf = t.to(dtype=dtype, device=cuda_dev).contiguous()
            else:
                buf = torch.empty(shape, dtype=dtype, device=cuda_dev)
            dist.broadcast(buf, src=src_global, group=group)
            return buf

        # Short-circuit: when CP, TP and PP groups are all trivial (size 1),
        # skip the GPU round-trip entirely and return the source tensor.
        cp_trivial = mpu.get_context_parallel_world_size() <= 1
        tp_trivial = mpu.get_tensor_model_parallel_world_size() <= 1
        pp_trivial = (not broadcast_pp) or mpu.get_pipeline_model_parallel_world_size() <= 1
        if cp_trivial and tp_trivial and pp_trivial:
            if is_sender and tensor is not None:
                return tensor.to(dtype=dtype).contiguous()
            # Shouldn't happen (sender has the tensor), but be safe.
            return torch.empty(0, dtype=dtype)

        is_cp_rank0 = mpu.get_context_parallel_rank() == 0
        is_pp_rank0 = mpu.get_pipeline_model_parallel_rank() == 0
        is_tp_rank0 = mpu.get_tensor_model_parallel_rank() == 0

        # --- Step 0: CP broadcast (cp_rank==0 -> others in each CP group) ---
        # Only the (TP=0, PP=0) CP group needs this: it fans data from
        # CP=0 to all CP peers.  Other CP groups are skipped — their
        # ranks receive data later via the PP and TP broadcasts (the
        # PP/TP source ranks already have data from this CP step).
        # All members of a CP group share the same TP/PP rank, so the
        # guard is uniform within each group — no hang risk.
        if not cp_trivial and is_tp_rank0 and is_pp_rank0:
            cp_group = mpu.get_context_parallel_group()
            cp_src_global = dist.get_global_rank(cp_group, 0)
            tensor = _bcast_unknown_shape(tensor, is_cp_rank0, cp_src_global, cp_group)

        # --- Step 1: PP broadcast (only among tp_rank==0 ranks) ---
        # After CP broadcast, every (TP=0, PP=0, CP=*) rank has data, so
        # the PP sender is identified by pp_rank==0.
        if not pp_trivial and is_tp_rank0:
            pp_group = mpu.get_pipeline_model_parallel_group()
            pp_src_global = dist.get_global_rank(pp_group, 0)
            tensor = _bcast_unknown_shape(tensor, is_pp_rank0, pp_src_global, pp_group)

        # --- Step 2: TP broadcast (tp_rank==0 -> others in each TP group) ---
        if not tp_trivial:
            tp_group = mpu.get_tensor_model_parallel_group()
            tp_src_global = dist.get_global_rank(tp_group, 0)
            tensor = _bcast_unknown_shape(tensor, is_tp_rank0, tp_src_global, tp_group)

        return tensor

    values_out = _bcast_tensor(values, is_src, torch.int32)
    offsets_out = _bcast_tensor(offsets, is_src, torch.long)

    if keep_on_gpu:
        # When optimize_routing_replay is enabled, keep tensors on GPU to
        # avoid a redundant GPU→CPU→GPU round-trip.  fill_routing_replay's
        # RoutingReplay.record() handles GPU→CPU-pinned copy automatically.
        # _bcast_tensor may short-circuit and return CPU tensors when all
        # groups are trivial (size 1); ensure GPU residency in that case.
        if not values_out.is_cuda:
            values_out = values_out.to(device=cuda_dev)
        if not offsets_out.is_cuda:
            offsets_out = offsets_out.to(device=cuda_dev)
        return values_out, offsets_out

    # Move back to CPU for downstream consumption (fill_routing_replay etc.)
    return values_out.cpu(), offsets_out.cpu()


def _bcast_known_tensor(tensor, is_src, dtype, shape, cuda_dev, broadcast_pp):
    """Broadcast a single tensor of *known* dtype/shape across CP, TP, then
    PP."""

    def _bcast(t, contribute, group):
        # The group's rank-0 contributes its current buffer when it holds real
        # data; otherwise every member allocates a (correctly shaped)
        # placeholder that a later stage overwrites.
        if contribute and t is not None:
            buf = t.to(device=cuda_dev, dtype=dtype).contiguous()
        else:
            buf = torch.empty(shape, dtype=dtype, device=cuda_dev)
        dist.broadcast(buf, src=dist.get_global_rank(group, 0), group=group)
        return buf

    # --- Short-circuit: skip all GPU round-trips when every group is trivial ---
    cp_trivial = mpu.get_context_parallel_world_size() <= 1
    tp_trivial = mpu.get_tensor_model_parallel_world_size() <= 1
    pp_trivial = (not broadcast_pp) or mpu.get_pipeline_model_parallel_world_size() <= 1

    if cp_trivial and tp_trivial and pp_trivial:
        # No actual broadcast needed — return the source tensor on CPU directly,
        # avoiding the costly CPU → GPU → NCCL self-send → GPU → CPU round-trip.
        if tensor is not None:
            return tensor.to(dtype=dtype).contiguous()
        return torch.empty(shape, dtype=dtype)

    # --- Step 1: CP broadcast (CP=0 -> other CP ranks of TP=0/PP=0) ---
    # Only the global source's CP group has real data on its rank-0; the rest
    # broadcast a placeholder that the TP / PP stages below overwrite.
    if not cp_trivial:
        tensor = _bcast(tensor, is_src, mpu.get_context_parallel_group())

    # --- Step 2: TP broadcast (tp_rank==0 -> others in each TP group) ---
    if not tp_trivial:
        tensor = _bcast(tensor, mpu.get_tensor_model_parallel_rank() == 0, mpu.get_tensor_model_parallel_group())

    # --- Step 3: PP broadcast (pp_rank==0 -> others in each PP group) ---
    if not pp_trivial:
        tensor = _bcast(tensor, mpu.get_pipeline_model_parallel_rank() == 0, mpu.get_pipeline_model_parallel_group())

    return tensor


def _encode_multimodal_inputs(mm_list):
    """Split a per-sample multimodal list into a tiny pickle-able spec and a
    flat, traversal-ordered list of the raw tensors to stream via NCCL.

    Returns ``(spec, tensors)`` where *spec* mirrors ``mm_list`` but replaces
    every tensor with its ``{"dtype", "shape"}`` descriptor (a few bytes), and
    *tensors* is the ordered list of tensors referenced by the spec. Tensors
    are deliberately kept out of the pickle so ``broadcast_object_list`` only
    serialises kilobytes instead of gigabytes.
    """
    spec: List[Any] = []
    tensors: List[torch.Tensor] = []
    for sample in mm_list:
        if sample is None:
            spec.append(None)
            continue
        entry: Dict[str, Any] = {}
        for key, val in sample.items():
            if isinstance(val, torch.Tensor):
                entry[key] = {"t": "tensor", "dtype": val.dtype, "shape": tuple(val.shape)}
                tensors.append(val)
            elif isinstance(val, list) and val and all(isinstance(x, torch.Tensor) for x in val):
                entry[key] = {"t": "list", "items": [{"dtype": x.dtype, "shape": tuple(x.shape)} for x in val]}
                tensors.extend(val)
            else:
                # Non-tensor (python scalar / small list); carry it inline.
                entry[key] = {"t": "raw", "value": val}
        spec.append(entry)
    return spec, tensors


def _broadcast_multimodal_inputs(spec, send_tensors, is_src, cuda_dev, broadcast_pp):
    """Reconstruct ``multimodal_train_inputs`` on every rank by streaming the
    raw tensors via NCCL (zero pickle) instead of through
    ``broadcast_object_list``."""
    if spec is None:
        return None

    out: List[Any] = []
    idx = 0
    for entry in spec:
        if entry is None:
            out.append(None)
            continue
        sample: Dict[str, Any] = {}
        for key, enc in entry.items():
            if enc["t"] == "tensor":
                src_t = send_tensors[idx] if is_src else None
                idx += 1
                sample[key] = _bcast_known_tensor(
                    src_t, is_src, enc["dtype"], enc["shape"], cuda_dev, broadcast_pp
                ).cpu()
            elif enc["t"] == "list":
                items: List[Any] = []
                for sub in enc["items"]:
                    src_t = send_tensors[idx] if is_src else None
                    idx += 1
                    items.append(
                        _bcast_known_tensor(src_t, is_src, sub["dtype"], sub["shape"], cuda_dev, broadcast_pp).cpu()
                    )
                sample[key] = items
            else:  # raw
                sample[key] = enc["value"]
        out.append(sample)
    return out


def _tensor_to_python_values(value: torch.Tensor) -> list[Any]:
    if not value.is_nested:
        return value.tolist()

    rows = value.unbind(0)

    try:
        dense = torch.stack(rows)
    except RuntimeError:
        # Preserve row boundaries for truly ragged values.
        return [row.tolist() for row in rows]

    # TransferQueue may reconstruct scalar fields as singleton rows when
    # the per-sample shape () is reported as (1,).
    if dense.ndim == 2 and dense.shape[1] == 1:
        return dense.squeeze(-1).tolist()

    return dense.tolist()


def get_data_from_transfer_queue(
    args,
    tq_client,
    data_fields,
    batch_size,
    partition_id,
    task_name,
    sampling_config,
    batch_index,
    broadcast_pp: bool = True,
    per_rank_fetch: bool = False,
    token_budget: int | None = None,
    allow_underfill: bool = True,
    post_process: bool = True,
    synchronize_per_rank_fetch: bool = True,
):
    """Fetch a batch from the transfer queue and broadcast it across tensor-
    parallel and optionally pipeline-parallel ranks.

    The function queries the transfer queue client (`tq_client`) for
    metadata and data on the appropriate rank(s) based on the broadcast_pp
    parameter. The retrieved pair (data, meta) is then broadcast across
    tensor-parallel ranks and optionally across pipeline-parallel ranks
    using torch.distributed.broadcast_object_list so that every rank has
    the same batch information.

    If the returned `rollout_data` is an instance of `TensorDict`, we
    convert it into a plain Python dictionary. This conversion turns
    tensor-valued entries into lists (so downstream code may index into
    them per-sample) and converts special fields like lengths/reward into
    Python lists as well.

    Args:
        args: Configuration / runtime arguments (used for post-processing).
        tq_client: Transfer-queue client with `get_meta` and `get_data` API.
        data_fields: List of field names to request.
        batch_size: Desired batch size to request.
        partition_id: Partition identifier string for the queue.
        task_name: Task name used by the queue.
        sampling_config: Extra sampling configuration passed to the queue.
        batch_index: Index of the batch to request (used for replay semantics).
        broadcast_pp: Whether to broadcast across pipeline parallel ranks.
            True for colocate mode, False for fully async mode.
        per_rank_fetch: When True, every TP/PP rank independently calls
            ``get_meta`` + ``get_data`` (relying on the TQ sampler's replay
            state for the same logical ``dp_rank`` and ``batch_index`` to return
            identical sample id lists across model-parallel ranks), and all
            TP/PP broadcasts are skipped.  Trades a single rank-0 pickle + one
            NCCL bcast for N parallel ZMQ deserialises — wins when pickle
            dominates ``tgd_bcast_tp_time``.  Caller must ensure
            ``rollout_routed_experts`` is not in ``data_fields`` (its bcast
            path is incompatible) — actor.py guards this.
        post_process: Move and reshape rollout fields for Megatron. Set to
            False only for CPU-only background prefetch; the training thread
            must materialize each micro-batch before use.
        synchronize_per_rank_fetch: When True, per-rank fetches agree on
            empty-vs-data state across the model-parallel replica before
            returning. Set to False only when the caller has its own foreground
            synchronization point and must keep this call collective-free.

    Returns:
        Tuple[Optional[dict], Optional[Any]]: A tuple of (rollout_data, batch_meta).
        If no data is available, both elements are None.
    """
    if not synchronize_per_rank_fetch and not per_rank_fetch:
        raise ValueError("synchronize_per_rank_fetch=False requires per_rank_fetch=True")

    # Compose request configuration and ask the queue for metadata.
    config = {**sampling_config, "batch_index": batch_index, "partition_id": partition_id}
    if token_budget is not None:
        # Token-budget fetch mode: the streaming sampler needs dp_size and
        # allow_underfill in sampling_config to decide bucket assignment and
        # end-of-stream behaviour.  dp_rank is already in sampling_config.
        config["allow_underfill"] = allow_underfill

    # Determine which rank should fetch data
    #
    # CP=0 must be in the predicate (alongside TP=0 / PP=0) — otherwise every CP
    # partner of (TP=0, PP=0) independently calls tq_client.get_meta / get_data
    # and they race the producer: a fetcher arriving before the producer fills
    # `ready_indexes` gets back `[], []` and the sampler does NOT cache an
    # empty result, while a fetcher arriving after gets the real samples and
    # writes the cache. So 8 CP partners → split into "got data" and "got None"
    # subsets. With downstream TP/PP broadcast, each CP rank's result fans out
    # to its (TP, PP) cohort: half the world enters train_actor and hangs at
    # the first cross-rank collective, the other half loops, sees
    # all_consumed=True (because the winners consumed the partition), and
    # returns to main_loop → 16 idle + 16 hung on TP2/PP2/CP8/DP1.
    if per_rank_fetch:
        # Each rank pulls its own copy from TQ; broadcasts are skipped below.
        # Safe because all model-parallel ranks present the same logical
        # dp_rank/batch_index to the TQ sampler, so cache/replay returns
        # byte-identical sample ids.
        should_fetch = True
    elif broadcast_pp:
        # Colocate mode: only (tp_rank, pp_rank, cp_rank) == (0, 0, 0) fetches data
        should_fetch = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
        )
    else:
        # Fully async mode: only (tp_rank, cp_rank) == (0, 0) fetches data per PP stage
        should_fetch = mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0

    # tgd_fetch: time spent in the Ray transfer-queue RPC on the fetching rank.
    # Non-fetching ranks record ~0s, which by itself confirms whether the
    # collective is waiting on fetch (rank0 large, others ~0) or on broadcast.
    # In per_rank_fetch mode every rank records a real value (no broadcast
    # below) so the metric becomes wall-clock fetch+deserialise per rank.
    fetch_timer_name = "per_rank_fetch" if per_rank_fetch else "tgd_fetch"

    def _fetch_once() -> list:
        """One get_meta (+ get_data) round-trip on this rank."""
        if token_budget is not None:
            batch_meta = tq_client.get_meta(
                data_fields=data_fields,
                token_budget=token_budget,
                partition_id=partition_id,
                sampling_config=config,
                task_name=task_name,
            )  # type: ignore
        else:
            batch_meta = tq_client.get_meta(
                data_fields=data_fields,
                batch_size=batch_size,
                partition_id=partition_id,
                sampling_config=config,
                task_name=task_name,
            )  # type: ignore

        if batch_meta.size == 0:
            # Keep the (empty) meta so its extra_info (e.g. dummy_round) rides
            # the broadcast to every consumer rank — the consumer decides
            # real vs dummy vs end from what it fetched, no extra RPC.
            return [None, batch_meta]
        return [tq_client.get_data(batch_meta), batch_meta]

    with timer(fetch_timer_name):
        if should_fetch:
            rollout_data = _fetch_once()
        else:
            # Non-fetching ranks start with an empty placeholder and
            # will receive the real data via broadcast.
            rollout_data = [None, None]

    if per_rank_fetch and synchronize_per_rank_fetch:
        # No broadcast follows, so a producer race can split this logical DP
        # rank into "got data" and "empty meta" model-parallel subsets.
        rollout_data = _agree_on_fetch(
            rollout_data,
            _fetch_once,
            include_pipeline=broadcast_pp,
            context=f"partition={partition_id} batch_index={batch_index}",
        )

    # Use an explicit device so the communication backend (e.g. NCCL)
    # can bind to a known device context.
    cuda_dev = None if per_rank_fetch else device_utils.make_current_torch_device()

    # --- Extract rollout_routed_experts BEFORE broadcast_object_list ---
    # broadcast_object_list uses pickle for the entire payload. When
    # rollout_routed_experts is present (~377 MB for Qwen3-30B-A3B), pickle
    # serialization dominates train_get_data_time (~14s).  We extract it and
    # broadcast the underlying contiguous tensors via dist.broadcast (NCCL
    # zero-copy) instead, reducing the time to sub-second.
    has_routed_experts = "rollout_routed_experts" in data_fields
    routed_experts_values = None
    routed_experts_offsets = None

    if has_routed_experts and not per_rank_fetch and should_fetch and rollout_data[0] is not None:
        td = rollout_data[0]
        if isinstance(td, TensorDict) and "rollout_routed_experts" in td.keys():
            nt = td["rollout_routed_experts"]
            # NestedTensor jagged internals: _values (total_tokens, inner_dim), _offsets (batch+1,)
            routed_experts_values = nt._values.contiguous()
            routed_experts_offsets = nt._offsets.contiguous()
            # Remove from TensorDict so broadcast_object_list only pickles ~4 MB
            del td["rollout_routed_experts"]
            rollout_data[0] = td

    # --- Extract multimodal_train_inputs BEFORE broadcast_object_list ---
    # Only on the broadcast path: in per_rank_fetch mode every rank already
    # pulled its own multimodal_train_inputs from TQ, so it stays inside the
    # TensorDict and is converted to a per-sample list below (mirrors the
    # routed_experts handling).
    has_multimodal = "multimodal_train_inputs" in data_fields
    mm_spec = None
    mm_send_tensors: List[torch.Tensor] = []

    if has_multimodal and not per_rank_fetch and should_fetch and rollout_data[0] is not None:
        td = rollout_data[0]
        if isinstance(td, TensorDict) and "multimodal_train_inputs" in td.keys():
            from tensordict.tensorclass import NonTensorData

            mm_list: List[Any] = []
            for item in list(td["multimodal_train_inputs"]):
                raw = item.data if isinstance(item, NonTensorData) else item
                if raw is None:
                    mm_list.append(None)
                elif isinstance(raw, dict):
                    mm_list.append(raw)
                else:
                    mm_list.append(dict(raw.items()) if hasattr(raw, "items") else dict(raw.data))
            mm_spec, mm_send_tensors = _encode_multimodal_inputs(mm_list)
            # Remove from TensorDict so broadcast_object_list only pickles the spec.
            del td["multimodal_train_inputs"]
            rollout_data[0] = td

    # Carry the (tiny) multimodal spec alongside the payload so every rank
    # learns the dtype/shape of each tensor it is about to receive via NCCL.
    # In per_rank_fetch mode this is None (each rank reconstructs locally).
    rollout_data.append(mm_spec)

    if per_rank_fetch:
        # Cheap byte-only diagnostic; never pickles (that would defeat the
        # whole point of per_rank_fetch).
        _maybe_log_per_rank_fetch_diag(rollout_data)
    if not per_rank_fetch:
        # Always broadcast across tensor parallel ranks (now without routed_experts)
        _maybe_log_tgd_pickle_diag(rollout_data, should_fetch)
        # CP broadcast must come FIRST: only (TP=0, PP=0, CP=0) fetched, so we
        # need to fan out the result to the other CP partners of (TP=0, PP=0)
        # before TP / PP broadcasts can propagate it across the rest of the
        # world. Skipping this is what caused the 16-idle / 16-hung split.
        if mpu.get_context_parallel_world_size() > 1:
            with timer("tgd_bcast_cp"):
                dist.broadcast_object_list(
                    rollout_data,
                    device=cuda_dev,
                    group=mpu.get_context_parallel_group(),
                    group_src=0,
                )
        if mpu.get_tensor_model_parallel_world_size() > 1:
            with timer("tgd_bcast_tp"):
                dist.broadcast_object_list(
                    rollout_data,
                    device=cuda_dev,
                    group=mpu.get_tensor_model_parallel_group(),
                    group_src=0,
                )

        # Conditionally broadcast across pipeline parallel ranks
        if broadcast_pp and mpu.get_pipeline_model_parallel_world_size() > 1:
            with timer("tgd_bcast_pp"):
                dist.broadcast_object_list(
                    rollout_data,
                    device=cuda_dev,
                    group=mpu.get_pipeline_model_parallel_group(),
                    group_src=0,
                )

    # Unpack the broadcasted triple.
    rollout_data, batch_meta, mm_spec = rollout_data[0], rollout_data[1], rollout_data[2]

    if rollout_data is None:
        return None, batch_meta

    # --- Stream multimodal tensors via NCCL (zero-copy, CPU-resident result) ---
    mm_inputs = None
    if has_multimodal:
        with timer("tgd_bcast_mm"):
            mm_inputs = _broadcast_multimodal_inputs(mm_spec, mm_send_tensors, should_fetch, cuda_dev, broadcast_pp)

    # --- Broadcast routed_experts tensors via efficient dist.broadcast ---
    # Skipped entirely in per_rank_fetch mode: each rank already received the
    # NestedTensor inside its own get_data() return value; the conversion to
    # per-sample list happens below.
    if has_routed_experts and not per_rank_fetch:
        with timer("tgd_bcast_rexp"):
            routed_experts_values, routed_experts_offsets = _broadcast_routed_experts(
                routed_experts_values,
                routed_experts_offsets,
                should_fetch,
                cuda_dev,
                broadcast_pp,
                keep_on_gpu=getattr(args, "optimize_routing_replay", False),
            )

    # If the received object is a Tensordict, convert it into a plain Python
    # dict so downstream code can mix tensors and Python lists freely.
    if isinstance(rollout_data, TensorDict):
        new_rollout_data: Dict[str, Any] = {}
        for k, v in rollout_data.items():
            # Keep scalar metadata as Python values across the TensorDict boundary,
            # including NestedTensor values reconstructed by TransferQueue.
            if "lengths" in k or "reward" in k or k == "sample_index_mask_sums":
                new_rollout_data[k] = _tensor_to_python_values(v)
            elif k == "multimodal_train_inputs":
                # Only reached on the per_rank_fetch path (the broadcast path
                # extracts and NCCL-streams these before broadcast). Stored as a
                # list of tensordicts / dicts; some entries may be None for
                # text-only samples in a multimodal batch. Turn each non-None
                # entry into a plain dict.
                from tensordict.tensorclass import NonTensorData

                new_rollout_data[k] = []
                for item in list(v):
                    # NonTensorStack iteration yields NonTensorData wrappers
                    raw = item.data if isinstance(item, NonTensorData) else item
                    if raw is None:
                        new_rollout_data[k].append(None)
                    elif isinstance(raw, dict):
                        new_rollout_data[k].append(raw)
                    else:
                        # TensorDict or similar — convert to plain dict
                        new_rollout_data[k].append(dict(raw.items()) if hasattr(raw, "items") else dict(raw.data))
            elif k == "rollout_routed_experts":
                # rollout_routed_experts is stored as a NonTensorStack /
                # LinkedList in TensorDict (raw numpy arrays).  Iterating may
                # yield NonTensorData wrappers, so unwrap via `.data` when
                # needed to get the underlying numpy array.
                from tensordict.tensorclass import NonTensorData

                new_rollout_data[k] = [item.data if isinstance(item, NonTensorData) else item for item in v]
            elif isinstance(v, torch.Tensor):
                # Expand a tensor with batch dimension into a Python list of
                # per-sample tensors so downstream code can index them.
                new_rollout_data[k] = [tensor for tensor in v]  # noqa: C416
            else:
                raise TypeError(f"Unsupported rollout_data type for key '{k}': {type(v)}")

        rollout_data = new_rollout_data

    # Re-attach routed_experts as a list of 2D tensors (per-sample) — only on
    # the bcast path, where the NestedTensor was extracted into ``routed_experts_values``
    # before broadcast.  per_rank_fetch never strips it (each rank pulls its own
    # copy from TQ), so the TensorDict→dict conversion above already produced
    # the per-sample list under "rollout_routed_experts".
    if has_routed_experts and not per_rank_fetch:
        rollout_data["rollout_routed_experts"] = [
            routed_experts_values[routed_experts_offsets[i] : routed_experts_offsets[i + 1]]
            for i in range(len(routed_experts_offsets) - 1)
        ]

    # Re-attach the NCCL-streamed multimodal inputs (CPU-resident; moved to GPU
    # per micro-batch by get_batch).
    if has_multimodal and mm_inputs is not None:
        rollout_data["multimodal_train_inputs"] = mm_inputs

    if post_process:
        post_process_rollout_data(args, rollout_data)

    return rollout_data, batch_meta


def post_process_rollout_data(args, rollout_data):
    # move tokens/loss_masks to GPU in-place as a list of tensors (downstream
    # code in this module expects lists of sequence tensors for packing)
    from relax.backends.megatron.cp_utils import maybe_padded_total_lengths, slice_log_prob_with_cp

    cuda_dev = device_utils.make_current_torch_device()
    rollout_data["tokens"] = [torch.as_tensor(t, dtype=torch.long, device=cuda_dev) for t in rollout_data["tokens"]]
    rollout_data["loss_masks"] = [
        torch.as_tensor(t, dtype=torch.int, device=cuda_dev) for t in rollout_data["loss_masks"]
    ]
    if "classification_labels" in rollout_data:
        label_dtype = (
            torch.long
            if getattr(args, "problem_type", "single_label_classification") == "single_label_classification"
            else torch.float32
        )
        rollout_data["classification_labels"] = [
            torch.as_tensor(label, dtype=label_dtype, device=cuda_dev)
            for label in rollout_data["classification_labels"]
        ]
    if "sample_weights" in rollout_data:
        rollout_data["sample_weights"] = [
            torch.as_tensor(weight, dtype=torch.float32, device=cuda_dev) for weight in rollout_data["sample_weights"]
        ]
    # NOTE: multimodal_train_inputs are intentionally left on CPU here. Moving
    # the whole batch's pixel tensors to GPU up front would spike memory
    if args.qkv_format == "bshd":
        # TODO: micro-batch wise dynamic, possibly move to @data.py:get_data_iterator
        max_seq_len = max(rollout_data["total_lengths"])

        # pad to reduce memory fragmentation and maybe make the computation faster
        pad_size = mpu.get_tensor_model_parallel_world_size() * args.data_pad_size_multiplier
        max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

        rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

    padded_total_lengths = maybe_padded_total_lengths(
        rollout_data["total_lengths"],
        args.qkv_format,
        getattr(args, "is_vl_model", False)
        or "multimodal_train_inputs" in rollout_data
        or getattr(args, "uses_unsplit_forward", False),
    )

    response_float_fields = [
        "log_probs",
        "ref_log_probs",
        "rollout_log_probs",
        "advantages",
        "returns",
        "values",
        *iter_opd_cp_float_fields(),
    ]
    rewards = rollout_data.get("rewards", [])
    if rewards and isinstance(rewards[0], list):
        rollout_data["rewards"] = [
            row * response_length if len(row) == 1 else row
            for row, response_length in zip(rewards, rollout_data["response_lengths"], strict=True)
        ]
        response_float_fields.append("rewards")

    for key in response_float_fields:
        if key not in rollout_data:
            continue
        # Dynamic CP: keep per-sample log-prob fields FULL-length at ingestion.
        # Each micro-batch picks its own CP size at train time and get_batch
        # re-slices these fields per-mb (data.py dynamic-CP reslice); log_rollout_data
        # / compute_advantages consume them as full (cp_size=1). Slicing here with the
        # static max-CP zig-zag would leave rollout_log_probs (never recomputed) mis-
        # sharded — log_probs/ref_log_probs happen to be overwritten by the merged
        # forward, but rollout_log_probs is not, so it would reach logging CP-split.
        if getattr(args, "dynamic_context_parallel", False):
            rollout_data[key] = [
                torch.as_tensor(log_prob, device=cuda_dev, dtype=torch.float32) for log_prob in rollout_data[key]
            ]
            continue
        rollout_data[key] = [
            torch.as_tensor(
                slice_log_prob_with_cp(
                    log_prob,
                    total_length,
                    response_length,
                    args.qkv_format,
                    rollout_data["max_seq_lens"][i] if args.qkv_format == "bshd" else None,
                    padded_total_length=padded_total_lengths[i] if padded_total_lengths is not None else None,
                ),
                device=cuda_dev,
                dtype=torch.float32,
            )
            for i, (log_prob, total_length, response_length) in enumerate(
                zip(
                    rollout_data[key],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            )
        ]

    if args.use_opd:
        from relax.utils.opd.opd_main_worker import restore_opd_topk_rollout_fields

        restore_opd_topk_rollout_fields(rollout_data, args, cuda_dev)

    if "rollout_routed_experts" in rollout_data:
        from tensordict.tensorclass import NonTensorData

        rollout_data["rollout_routed_experts"] = [
            torch.as_tensor(r.data if isinstance(r, NonTensorData) else r, dtype=torch.long, device=cuda_dev)
            for r in rollout_data["rollout_routed_experts"]
        ]


class StreamingTQIterator:
    """Streaming iterator that pulls micro-batches from TransferQueue on
    demand.

    Each call to ``__next__`` blocks until a token-budget-sized micro-batch is
    available, then returns a ``(batch_dict, batch_meta)`` tuple with
    ``__loss_scale__`` injected.  ``StopIteration`` is raised when
    ``all_consumed_fn()`` returns True and the queue is empty.

    Every PP stage constructs its own instance.  The sampler result cache in
    ``StreamingTokenBudgetSampler`` guarantees that identical
    ``(dp_rank, batch_index)`` requests return the same indexes regardless of
    which PP stage is asking, so all stages raise ``StopIteration`` at the same
    micro-batch count — keeping p2p send/recv pairs aligned.

    Args:
        args: Runtime arguments (used for ``get_data_from_transfer_queue``).
        tq_client: TransferQueue client with ``get_meta`` / ``get_data`` API.
        data_fields: Field names to request from the queue.
        rollout_id: Current rollout partition identifier.
        token_budget: Target accumulated token count per micro-batch fetch.
        loss_scale: Scalar injected as ``__loss_scale__`` into each batch dict.
        all_consumed_fn: Callable returning True when the rollout is fully consumed.
        dp_rank: Data-parallel rank of this worker.
        dp_size: Data-parallel world size.
        task_name: TQ task name (default ``"actor_train"``).
        window_quota: Global per-window sample count (``mini_global_samples``),
            forwarded to the TQ sampler. The sampler stops dispatching this
            rollout-mini window once ``window_quota`` samples have been handed
            out across ALL DPs; per-DP counts stay uneven (token-budget driven)
            and the dummy-pad aligns cross-DP micro-batch counts. Termination is
            the per-window drained signal (``all_consumed_fn``), NOT a per-DP
            count target. ``None`` (fwd/logprob path) → no window cap.
        rollout_mini_index: Rollout-mini window id, passed to the TQ sampler
            (keys the sampler's per-window dispatch counter).
        start_batch_index: First batch index for this iterator; used to keep
            sampler cache keys distinct across rollout-mini windows.
        max_empty_sleep: Maximum sleep duration (seconds) between empty-poll retries.
        max_stream_stall_s: Wall-clock timeout for consecutive empty polls while
            the rollout is not yet drained. Exceeding it raises (deadlock guard).
    """

    def __init__(
        self,
        args,
        tq_client,
        data_fields: List[str],
        rollout_id: int,
        token_budget: int,
        loss_scale: float,
        all_consumed_fn: Callable[[], bool],
        dp_rank: int,
        dp_size: int,
        task_name: str = "actor_train",
        window_quota: Optional[int] = None,
        rollout_mini_index: int = 0,
        start_batch_index: int = 0,
        max_empty_sleep: float = 2.0,
        max_stream_stall_s: float = 3600.0,
        per_round_dummy: bool = False,
    ) -> None:
        if window_quota is not None and window_quota <= 0:
            raise ValueError(f"window_quota must be positive when set, got {window_quota}")
        if max_stream_stall_s <= 0:
            raise ValueError(f"max_stream_stall_s must be positive, got {max_stream_stall_s}")
        self.args = args
        self.tq_client = tq_client
        self.data_fields = data_fields
        self.rollout_id = rollout_id
        self.token_budget = token_budget
        self.loss_scale = loss_scale
        self.all_consumed_fn = all_consumed_fn
        # per_round_dummy: train path only. When True, on an empty fetch the
        # iterator reads the DUMMY-round flag from the fetched meta's extra_info
        # (sampler-driven per-round alignment) and emits a zero-grad dummy micro-
        # batch so all DPs run equal micro-batch counts (MoE EP all-to-all matches
        # by call order). The forward-only path keeps this False and uses the
        # legacy end-phase dummy-pad (a dummy there would corrupt put-back data).
        self.per_round_dummy = per_round_dummy
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.task_name = task_name
        # Global per-window sample quota (mini_global_samples), forwarded to the
        # TQ sampler so it stops dispatching this window once the quota is met
        # across all DPs.  The consumer no longer enforces a per-DP count target;
        # termination is the per-window drained signal (all_consumed_fn).
        self.window_quota = window_quota
        self.rollout_mini_index = rollout_mini_index
        self._sample_count: int = 0
        self.max_empty_sleep = max_empty_sleep
        # Wall-clock guard: if the stream produces no data for this long while
        # the rollout is NOT yet drained, raise instead of polling forever.
        # A genuine deadlock (e.g. a sample the sampler never dispatched to any
        # bucket, so all_consumed can never flip true) then surfaces as a fast,
        # diagnosable error rather than a silent hang that ends in an OOM kill.
        self.max_stream_stall_s = max_stream_stall_s

        self._batch_index: int = start_batch_index
        self._mb_count: int = 0
        # Per-mb timing: (tq_wait_s, compute_start_s) — compute time logged externally.
        self._tq_wait_times: List[float] = []
        # Buffer of (batch_dict, meta) tuples — populated by __next__ so that
        # get_buffer() can be used for end-of-rollout logging (mirrors MicroBatchListIterator).
        self._buffer: List[Tuple[Dict[str, Any], Any]] = []

        # ── cross-DP micro-batch alignment (dummy padding) ───────────────
        # With dynamic batch + DP>1, each DP packs a different number of
        # micro-batches (variable sample lengths → different token packing),
        # but all DP ranks must run the SAME number of fwd/bwd so the gradient
        # all-reduce stays in lockstep.  After real data is exhausted, the
        # iterator MAX-reduces its real mb count across DP and yields dummy
        # mbs (tagged __is_dummy__, zero gradient) up to that maximum.
        self._last_batch: Optional[Tuple[Dict[str, Any], Any]] = None
        self._dummies_remaining: Optional[int] = None  # computed at end-of-stream

        # ── tail prefetch ────────────────────────────────────────────────
        # While the trainer computes mb N, warm the controller-side sampler
        # cache for batch_index N+1 (balance round + all-dp dispatch) so the
        # next __next__ get_meta is a cache HIT instead of waiting on a balance
        # round / poll.  This is a meta-only prefetch (NO get_data transfer) —
        # it just populates the controller cache, keeping consumption marking
        # idempotent and avoiding GPU/stream complexity.  Single worker so at
        # most one warm-up is in flight; failures are best-effort (the real
        # request falls back to the normal path on a cache miss).
        self._prefetch_executor = None
        self._prefetched_index: int = -1
        if mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
            from concurrent.futures import ThreadPoolExecutor

            self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-tq-prefetch")

    def _sampling_config(self, batch_index: int) -> Dict[str, Any]:
        return {
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            "task_name": self.task_name,
            "batch_index": batch_index,
            "partition_id": f"train_{self.rollout_id}",
            "allow_underfill": True,
            "rollout_mini_index": self.rollout_mini_index,
            # Global per-window cap for the sampler (mini_global_samples). The
            # sampler stops dispatching this window once the quota is met across
            # all DPs; None on the fwd/logprob path → no cap.
            "window_quota": self.window_quota,
        }

    @staticmethod
    def _num_samples(data: Dict[str, Any]) -> int:
        for key in ("tokens", "total_lengths", "response_lengths", "loss_masks"):
            value = data.get(key)
            if value is not None:
                return len(value)
        return 0

    def _warm_next_batch_index(self, next_index: int) -> None:
        """Fire-and-forget controller cache warm-up for ``next_index``.

        Runs the same token-budget ``get_meta`` the real fetch will issue (same
        sampling_config / partition / batch_index) so the controller sampler
        prepares & caches all dp slices.  Meta-only: result is discarded.
        """
        if self._prefetch_executor is None or next_index <= self._prefetched_index:
            return
        self._prefetched_index = next_index

        def _task() -> None:
            try:
                config = self._sampling_config(next_index)
                self.tq_client.get_meta(
                    data_fields=self.data_fields,
                    token_budget=self.token_budget,
                    partition_id=f"train_{self.rollout_id}",
                    sampling_config=config,
                    task_name=self.task_name,
                )
            except Exception as e:  # best-effort; real fetch will retry on miss
                logger.debug("[stream-prefetch] warm batch_idx=%d failed: %s", next_index, e)

        try:
            self._prefetch_executor.submit(_task)
        except RuntimeError:
            # Executor already shut down (iterator finishing) — ignore.
            pass

    def _shutdown_prefetch(self) -> None:
        if self._prefetch_executor is not None:
            self._prefetch_executor.shutdown(wait=False)
            self._prefetch_executor = None

    def __iter__(self) -> "StreamingTQIterator":
        return self

    def _dp_max_microbatches(self, local_count: int) -> int:
        """MAX-reduce the local real mb count across the data-parallel group.

        All PP stages within a DP have the same real count (the sampler caches
        per-(dp_rank, batch_index) results, so stages stay aligned), so each PP
        stage's all-reduce over its DP-CP group yields the same global maximum.
        """
        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        t = torch.tensor([local_count], dtype=torch.int, device=device_utils.make_current_torch_device())
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=dp_cp_group)
        return int(t.item())

    def _make_dummy_batch(self) -> Tuple[Dict[str, Any], Any]:
        """Build a dummy micro-batch from the last real one, tagged so the loss
        contributes zero gradient (keeps all DP ranks at the same micro-batch
        count)."""
        if self._last_batch is None:
            # A dp asked to emit a dummy before ever seeing a real batch = it got
            # ZERO samples this window. With the fair per-window quota
            # (mini_global_samples % dp_size == 0) every dp normally gets >= 1
            # real sample, so this is pathological — fail loudly rather than
            # fabricate a shape-less batch.
            raise RuntimeError(
                f"StreamingTQIterator rollout={self.rollout_id} mini={self.rollout_mini_index} "
                f"dp={self.dp_rank} must emit a dummy micro-batch but received zero real samples "
                "this window (no template to build a dummy from)."
            )
        data, meta = self._last_batch
        dummy = dict(data)
        dummy["__is_dummy__"] = True
        dummy["__loss_scale__"] = self.loss_scale
        return dummy, meta

    def _emit_dummy_batch(self) -> Tuple[Dict[str, Any], Any]:
        """Emit one zero-grad dummy micro-batch for a DUMMY round (this dp is
        short this round but another dp still has a real slice).

        Advances batch_index + mb_count for lockstep with other DPs; does NOT
        count toward sample_count or the log buffer (it carries no real data).
        """
        dummy = self._make_dummy_batch()
        self._batch_index += 1
        self._mb_count += 1
        if self._mb_count % 20 == 0:
            logger.info(
                "[StreamingTQIterator] rollout=%s mini=%d dp=%d dummy round mb=%d batch_index=%d",
                self.rollout_id,
                self.rollout_mini_index,
                self.dp_rank,
                self._mb_count,
                self._batch_index,
            )
        return dummy

    def _finish_real_batches(self, reason: str) -> Tuple[Dict[str, Any], Any]:
        # Real data exhausted. Align mb count across DP ranks: MAX-reduce the
        # real count, then enter the dummy-padding phase so every DP yields the
        # same total number of micro-batches.
        k_real = self._mb_count
        k_global = self._dp_max_microbatches(k_real)
        total_wait = sum(self._tq_wait_times)
        logger.info(
            "[StreamingTQIterator] rollout=%s mini=%d dp=%d real mbs=%d k_global=%d "
            "samples=%d%s (pad %d dummy) total_tq_wait=%.3fs reason=%s",
            self.rollout_id,
            self.rollout_mini_index,
            self.dp_rank,
            k_real,
            k_global,
            self._sample_count,
            f"/{self.window_quota}" if self.window_quota is not None else "",
            max(0, k_global - k_real),
            total_wait,
            reason,
        )
        pad = k_global - k_real
        if pad > 0 and self._last_batch is None:
            self._shutdown_prefetch()
            raise RuntimeError(
                f"StreamingTQIterator rollout={self.rollout_id} mini={self.rollout_mini_index} "
                f"dp={self.dp_rank} must pad {pad} dummy micro-batches but consumed zero real micro-batches"
            )
        self._dummies_remaining = pad
        return self.__next__()

    def _emit_data_batch(
        self,
        data: Dict[str, Any],
        meta: Any,
        tq_wait: float,
    ) -> Tuple[Dict[str, Any], Any]:
        n_samples = self._num_samples(data)
        if n_samples <= 0:
            raise RuntimeError(
                f"StreamingTQIterator rollout={self.rollout_id} mini={self.rollout_mini_index} "
                "received a non-empty batch with zero sample-aligned fields"
            )

        # The sampler enforces the per-window global quota, so a fetched mb never
        # straddles a window boundary — no consumer-side split/spill needed.
        data["__loss_scale__"] = self.loss_scale
        self._sample_count += n_samples
        self._buffer.append((data, meta))
        # Snapshot for dummy padding.  ``get_batch`` mutates the batch
        # dict IN PLACE (e.g. reassigns ``batch["tokens"]`` to the
        # concatenated+padded packed tensor), and the schedule passes
        # this very ``data`` object to ``get_batch``.  If we kept a
        # reference to it, a later ``_make_dummy_batch`` would copy the
        # ALREADY-PACKED ``tokens`` while ``loss_masks`` / ``total_lengths``
        # stay per-sample lists → loss_mask/token shape mismatch in
        # ``get_batch`` (tok length = sum of two samples).  Store a
        # shallow dict copy (new top-level dict, same per-sample list
        # values, which get_batch does not mutate) so the dummy always
        # rebuilds from the pristine raw fields.
        self._last_batch = (dict(data), meta)
        self._batch_index += 1
        self._mb_count += 1

        adv_info = ""
        if "advantages" in data:
            advs = data["advantages"]
            if isinstance(advs, list) and len(advs) > 0:
                adv_vals = [a.float().mean().item() if hasattr(a, "mean") else float(a) for a in advs]
                adv_info = (
                    f" adv_means=[{','.join(f'{v:.4f}' for v in adv_vals[:4])}{'...' if len(adv_vals) > 4 else ''}]"
                )
        # ``samples`` reports GLOBAL window progress (sum across all dps) vs the
        # window quota, so it monotonically approaches ``window_quota`` (e.g.
        # 256).  The controller piggybacks this global count on the fetched
        # meta's extra_info (``window_dispatched``) — no cross-DP all_reduce.
        # Falls back to this dp's local count when the global count is absent
        # (legacy/non-streaming fetches).
        extra = getattr(meta, "extra_info", None) or {}
        window_progress = extra.get("window_dispatched")
        if window_progress is None:
            window_progress = self._sample_count
        logger.info(
            "[StreamingTQIterator] rollout=%s mini=%d dp=%d/%d mb=%d source=%s tq_wait=%.3fs "
            "n_samples=%d samples=%d%s loss_scale=%.6f%s",
            self.rollout_id,
            self.rollout_mini_index,
            self.dp_rank,
            self.dp_size,
            self._mb_count,
            "tq",
            tq_wait,
            n_samples,
            window_progress,
            f"/{self.window_quota}" if self.window_quota is not None else "",
            self.loss_scale,
            adv_info,
        )
        # Warm the controller cache for the next mb while the trainer
        # computes this one (tail prefetch; meta-only).
        self._warm_next_batch_index(self._batch_index)
        return data, meta

    def __next__(self) -> Tuple[Dict[str, Any], Any]:
        # Dummy-padding phase: real data exhausted, emit dummies up to k_global.
        if self._dummies_remaining is not None:
            if self._dummies_remaining <= 0:
                self._shutdown_prefetch()
                raise StopIteration
            self._dummies_remaining -= 1
            self._mb_count += 1
            logger.info(
                "[StreamingTQIterator] rollout=%s dp=%d dummy mb=%d (padding to k_global)",
                self.rollout_id,
                self.dp_rank,
                self._mb_count,
            )
            return self._make_dummy_batch()

        t0 = time.monotonic()
        empty_streak = 0
        partition_id = f"train_{self.rollout_id}"

        while True:
            sampling_config = self._sampling_config(self._batch_index)
            data, meta = get_data_from_transfer_queue(
                args=self.args,
                tq_client=self.tq_client,
                data_fields=self.data_fields,
                batch_size=None,
                partition_id=partition_id,
                task_name=self.task_name,
                sampling_config=sampling_config,
                batch_index=self._batch_index,
                broadcast_pp=False,
                token_budget=self.token_budget,
                allow_underfill=True,
            )

            if data is not None:
                tq_wait = time.monotonic() - t0
                self._tq_wait_times.append(tq_wait)
                return self._emit_data_batch(data, meta, tq_wait=tq_wait)

            # Data not yet available for this (dp, batch_index).
            if self.per_round_dummy:
                # Train path: the consumer decides from WHAT IT FETCHED — the
                # sampler carries a DUMMY-round flag in the (empty) meta's
                # extra_info. DUMMY → this dp is short this round but another dp
                # still got a real slice; emit a zero-grad dummy and advance so all
                # dps run the SAME number of micro-batches (MoE EP all-to-all
                # matches by call order). drained → StopIteration. No extra RPC,
                # no cross-DP all_reduce.
                extra = getattr(meta, "extra_info", None) or {}
                if extra.get("dummy_round"):
                    return self._emit_dummy_batch()
                if extra.get("stream_end"):
                    self._shutdown_prefetch()
                    raise StopIteration
            else:
                # Forward-only / legacy path: end-phase dummy-pad via all_reduce
                # (a per-round dummy here would corrupt put-back log-probs).
                if self.all_consumed_fn():
                    return self._finish_real_batches("transfer queue window drained (underfill ok)")

            empty_streak += 1
            elapsed = time.monotonic() - t0
            if empty_streak % 20 == 0:
                logger.info(
                    "[StreamingTQIterator] rollout=%s dp=%d polling: empty_streak=%d "
                    "batch_index=%d mb_count=%d elapsed=%.1fs",
                    self.rollout_id,
                    self.dp_rank,
                    empty_streak,
                    self._batch_index,
                    self._mb_count,
                    elapsed,
                )
            # Deadlock guard: no data for max_stream_stall_s while not drained
            # means the stream can never complete (e.g. a produced sample that
            # the sampler never dispatched to any bucket, so all_consumed stays
            # False forever).  Fail fast and diagnosably instead of hanging.
            if elapsed >= self.max_stream_stall_s:
                self._shutdown_prefetch()
                raise RuntimeError(
                    f"StreamingTQIterator rollout={self.rollout_id} mini={self.rollout_mini_index} "
                    f"dp={self.dp_rank} stalled {elapsed:.0f}s (>{self.max_stream_stall_s:.0f}s) with no "
                    f"data and rollout not drained: samples={self._sample_count} "
                    f"batch_index={self._batch_index} empty_streak={empty_streak}. Likely a sample was "
                    "never dispatched to a DP bucket by the sampler, or the drained predicate is stuck."
                )
            sleep_s = min(0.05 * empty_streak, self.max_empty_sleep)
            time.sleep(sleep_s)

    def __del__(self) -> None:
        # Safety net: ensure the prefetch worker is released even if iteration
        # ended without reaching the StopIteration branch.
        try:
            self._shutdown_prefetch()
        except Exception:
            pass

    def get_buffer(self) -> List[Tuple[Dict[str, Any], Any]]:
        """Return all micro-batches emitted so far — used for end-of-rollout
        logging."""
        return self._buffer

    def step(self, partition_id: str) -> None:  # noqa: ARG002
        """API parity with StreamingDataLoader.step; no-op (iterator is per-
        rollout)."""
        return

    @property
    def mb_count(self) -> int:
        return self._mb_count

    @property
    def tq_wait_times(self) -> List[float]:
        return self._tq_wait_times
