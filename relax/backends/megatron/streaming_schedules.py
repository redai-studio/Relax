# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Streaming forward-backward schedules for fully-async dynamic-batch training.

These schedules replace Megatron's standard ``forward_backward_*`` functions
when ``--use-dynamic-batch-size`` and ``--fully-async`` are both active.

Key difference from the standard schedules:
- ``num_microbatches`` is NOT known upfront; each stage discovers it by
  pulling micro-batches from a ``StreamingTQIterator`` until ``StopIteration``.
- All micro-batches run inside ``no_sync`` (gradient accumulation without
  inter-rank reduction); ``finalize_model_grads_func`` is called once at the end.
- Loss normalisation is handled entirely via ``__loss_scale__`` injected by
  the iterator.  Megatron's ``output_tensor /= num_microbatches`` becomes a
  no-op because we pass ``num_microbatches=1`` to ``forward_step``.

PP > 1 correctness:
  Every PP stage independently constructs a ``StreamingTQIterator``.  The
  ``StreamingTokenBudgetSampler`` result cache guarantees that identical
  ``(dp_rank, batch_index)`` requests return the same sample indexes regardless
  of which PP stage is asking.  Therefore all stages exhaust their iterators
  at the same micro-batch count, which keeps p2p send/recv pairs aligned.
"""

import contextlib
from typing import Iterator, List, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_step,
    get_tensor_shapes,
)
from megatron.core.utils import get_model_config

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PP = 1  (no pipelining)
# ---------------------------------------------------------------------------


def streaming_forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,  # ignored — iterator drives the count
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: Optional[int] = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    **kwargs,
):
    """Streaming forward+backward with no pipeline parallelism (PP=1).

    Iterates ``data_iterator`` until ``StopIteration``, running one
    forward+backward pass per micro-batch.  All passes use ``no_sync``
    (gradient accumulation); ``finalize_model_grads_func`` syncs once at the
    end.  ``num_microbatches`` is silently ignored.
    """
    if isinstance(model, list):
        assert len(model) == 1, "streaming no-pipelining schedule requires single model chunk"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, "streaming no-pipelining schedule requires single iterator"
        data_iterator = data_iterator[0]

    config = get_model_config(model)

    pg_collection = kwargs.get("pg_collection", None)
    force_all_reduce = kwargs.get("force_all_reduce", False)

    if pg_collection is None:
        from megatron.core.pipeline_parallel.schedules import ProcessGroupCollection

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = parallel_state.get_tensor_model_parallel_group()
        pg_collection.cp = parallel_state.get_context_parallel_group()
        pg_collection.embd = parallel_state.get_embedding_group(check_initialized=False)
        pg_collection.pos_embd = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection.pp = parallel_state.get_pipeline_model_parallel_group()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)

    cp_size = pg_collection.cp.size()

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    mb_idx = 0
    input_tensor, output_tensor_grad = None, None

    dp_rank = parallel_state.get_data_parallel_rank()

    # All micro-batches run inside no_sync (gradient accumulation).
    # Cross-DP micro-batch alignment (dummy padding to the per-DP max) is done
    # inside the StreamingTQIterator, so every DP rank yields the same number
    # of micro-batches here — keeping the DP gradient all-reduce in lockstep.
    with no_sync_func():
        for batch in data_iterator:
            single_mb_iter = iter([batch])
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                single_mb_iter,
                model,
                1,  # num_microbatches=1: makes Megatron's /num_microbatches a no-op
                input_tensor,
                forward_data_store,
                config,
                cp_group_size=cp_size,
                collect_non_loss_data=collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, mb_idx == 0),
                current_microbatch=mb_idx,
            )
            total_num_tokens += num_tokens
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, config)
            if mb_idx == 0:
                logger.info(
                    "[streaming-pp1] dp=%d first mb: output_tensor=%.6f num_tokens=%s",
                    dp_rank,
                    output_tensor.item() if output_tensor.numel() == 1 else float("nan"),
                    num_tokens,
                )
            mb_idx += 1

    if mb_idx == 0:
        logger.error(
            "[streaming-pp1] dp=%d data_iterator yielded ZERO micro-batches — "
            "grad_norm will be 0 and weights will not update. "
            "Check StreamingTQIterator / all_consumed / TQ sampler.",
            dp_rank,
        )
    else:
        logger.info(
            "[streaming-pp1] dp=%d processed %d micro-batches, total_num_tokens=%s", dp_rank, mb_idx, total_num_tokens
        )

    if config.finalize_model_grads_func is not None and not forward_only:
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    return forward_data_store


# ---------------------------------------------------------------------------
# PP > 1  (non-interleaved 1F1B, streaming / prefetch-driven)
# ---------------------------------------------------------------------------


def streaming_forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,  # ignored — iterator drives the count
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn=None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection=None,
    force_all_reduce: Optional[bool] = False,
    **kwargs,
):
    """Streaming 1F1B schedule for PP > 1, prefetch-driven, no fixed
    num_microbatches.

    Design
    ------
    Each PP stage independently pulls data from its own ``StreamingTQIterator``
    (the sampler cache ensures all stages see the same sample sequence).
    The warmup depth is ``pp_size - pp_rank - 1`` (same as standard 1F1B).
    The steady phase uses prefetch: ``next(data_iterator)`` is called one
    step ahead so we can detect the last micro-batch before choosing between
    ``send_backward`` and ``send_backward_recv_forward``.

    All stages raise ``StopIteration`` at the same micro-batch count → p2p
    send/recv pairs stay aligned without broadcasting ``num_microbatches``.
    """
    if isinstance(model, list):
        assert len(model) == 1, "streaming pipelining schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, "streaming pipelining schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)

    if config.overlap_p2p_comm:
        raise ValueError("Streaming pipeline schedule does not support overlap_p2p_comm")

    tp_group, cp_group, cp_size = None, None, None

    if p2p_communicator is None and pg_collection is None:
        from megatron.core.pipeline_parallel.schedules import ProcessGroupCollection

        p2p_communicator = P2PCommunicator(pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
    else:
        assert hasattr(pg_collection, "cp"), "pg_collection must have cp"
        cp_group = pg_collection.cp
        cp_size = cp_group.size()
        tp_group = getattr(pg_collection, "tp", None)

    # Needed for embedding wgrad deferral.
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model, p2p_communicator.is_pp_last_stage)

    # tensor shapes for p2p communication
    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_group=getattr(p2p_communicator, "pp_group", None),
        is_recv=True,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
        pp_group=getattr(p2p_communicator, "pp_group", None),
        is_recv=False,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(recv_tensor_shapes, send_tensor_shapes)

    # Grad sync setup — all passes run in no_sync; one final sync at the end.
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    # Storage for in-flight tensors during warmup/steady (needed for backward).
    input_tensors: List = []
    output_tensors: List = []

    # Warmup depth = standard 1F1B formula.
    max_warmup = p2p_communicator.total_stages - p2p_communicator.current_stage - 1

    # Helper: try to get the next item from the iterator; return _STOP sentinel on exhaustion.
    _STOP = object()

    def _try_next():
        try:
            return next(data_iterator)
        except StopIteration:
            return _STOP

    def _make_single_iter(batch):
        """Wrap one (data, meta) batch in a one-shot iterator for
        forward_step."""
        return iter([batch])

    mb_total = 0  # total micro-batches processed across all phases

    # ── Warmup ──────────────────────────────────────────────────────────────
    # Pull up to max_warmup mbs.  StopIteration means we have fewer mbs total
    # than the warmup depth — record actual_warmup so cooldown matches.
    actual_warmup = 0
    warmup_batches = []
    for _ in range(max_warmup):
        batch = _try_next()
        if batch is _STOP:
            break
        warmup_batches.append(batch)

    for i, batch in enumerate(warmup_batches):
        input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, p2p_communicator.is_pp_first_stage)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            _make_single_iter(batch),
            model,
            1,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=mb_total,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
        total_num_tokens += num_tokens
        mb_total += 1
        actual_warmup += 1

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # ── Steady 1F1B ─────────────────────────────────────────────────────────
    # Prefetch the first steady-state mb.  If none, steady phase is empty.
    current_batch = _try_next()

    if current_batch is not _STOP:
        # Recv the first forward activation before entering the steady loop.
        input_tensor = p2p_communicator.recv_forward(recv_tensor_shapes, p2p_communicator.is_pp_first_stage)

        while current_batch is not _STOP:
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                _make_single_iter(current_batch),
                model,
                1,
                input_tensor,
                forward_data_store,
                config,
                cp_group_size=cp_size,
                collect_non_loss_data=collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, (mb_total == 0)),
                current_microbatch=mb_total,
                is_last_stage=p2p_communicator.is_pp_last_stage,
            )
            total_num_tokens += num_tokens
            mb_total += 1

            # Prefetch next batch BEFORE deciding the p2p pattern.
            next_batch = _try_next()
            is_last_steady = next_batch is _STOP

            if forward_only:
                p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
                if not is_last_steady:
                    input_tensor = p2p_communicator.recv_forward(
                        recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                    )
            else:
                output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                    output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
                )

                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)
                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

                # Pop the oldest in-flight pair for backward.
                input_tensor_bwd = input_tensors.pop(0)
                output_tensor_bwd = output_tensors.pop(0)

                if is_last_steady and actual_warmup == 0:
                    # No cooldown will follow → sync grads now.
                    if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                        enable_grad_sync()

                input_tensor_grad = backward_step(input_tensor_bwd, output_tensor_bwd, output_tensor_grad, config)

                if is_last_steady:
                    p2p_communicator.send_backward(input_tensor_grad, p2p_communicator.is_pp_first_stage)
                    # No need to recv_forward — cooldown will handle the remaining bwds.
                else:
                    input_tensor = p2p_communicator.send_backward_recv_forward(
                        input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                    )

            current_batch = next_batch

    # ── Cooldown ────────────────────────────────────────────────────────────
    if not forward_only:
        for i in range(actual_warmup):
            if i == actual_warmup - 1:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor_c = input_tensors.pop(0)
            output_tensor_c = output_tensors.pop(0)

            output_tensor_grad = p2p_communicator.recv_backward(send_tensor_shapes, p2p_communicator.is_pp_last_stage)
            input_tensor_grad = backward_step(input_tensor_c, output_tensor_c, output_tensor_grad, config)
            p2p_communicator.send_backward(input_tensor_grad, p2p_communicator.is_pp_first_stage)

        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if mb_total == 0:
        logger.warning(
            "[streaming-pp%d] data_iterator yielded zero micro-batches for this DP rank",
            p2p_communicator.total_stages,
        )

    if config.finalize_model_grads_func is not None and not forward_only:
        finish_embedding_wgrad_compute(config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group)
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    return forward_data_store
