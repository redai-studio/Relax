# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import contextlib
import copy
import os
from dataclasses import replace
from typing import Iterator, List, Optional, Union

import torch
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    check_first_val_step,
    deallocate_output_tensor,
    forward_step,
    get_tensor_shapes,
)
from megatron.core.utils import get_model_config

from examples.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, require_runtime
from examples.chunkflow.unit_batch import slice_parent_batch
from examples.chunkflow.units import Unit, completed_groups, make_model_plan, merge_model_plans
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def _unit_shapes(unit, config, tp_group, cp_group, pp_group):
    rows = (unit.num_tokens + 127) // 128 * 128
    return tuple(
        get_tensor_shapes(
            seq_length=rows,
            micro_batch_size=1,
            decoder_seq_length=None,
            config=config,
            tp_group=tp_group,
            cp_group=cp_group,
            pp_group=pp_group,
            is_recv=is_recv,
        )
        for is_recv in (True, False)
    )


def _check_tensor_shapes(tensors, expected) -> None:
    tensors = tensors if isinstance(tensors, list) else [tensors]
    shapes = [tuple(tensor.shape) for tensor in tensors]
    if shapes != expected:
        raise RuntimeError(f"ChunkFlow planned P2P shapes {expected} do not match tensor shapes {shapes}")


def chunkflow_forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
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
    if forward_only:
        raise NotImplementedError("chunkflow pipeline schedule is training-only")
    if isinstance(model, list):
        assert len(model) == 1, "chunkflow pipeline schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, "chunkflow pipeline schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError("chunkflow pipeline schedule does not support overlap_p2p_comm")

    from megatron.training.global_vars import get_args

    from examples.chunkflow.schedules import (
        _check_ep_schedule,
        _next_parent,
        _no_loss_step_func,
        _num_microbatches,
        _plan,
        _start_grad_sync,
    )

    args = get_args()
    chunk_size = int(args.chunkflow_chunk_size)
    policy = getattr(args, "chunkflow_pack_policy", "staggered")
    keep_last_k = getattr(args, "chunkflow_keep_last_k", 0) or 1 << 30
    runtime = require_runtime(model)
    parent_window = int(os.environ.get("RELAX_CHUNKFLOW_PARENT_WINDOW", "0"))
    if parent_window < 0:
        raise ValueError("RELAX_CHUNKFLOW_PARENT_WINDOW must be nonnegative (0 uses all step parents)")
    if p2p_communicator is None and pg_collection is None:
        from megatron.core.pipeline_parallel.schedules import ProcessGroupCollection

        p2p_communicator = P2PCommunicator(pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = parallel_state.get_pipeline_model_parallel_group()
        pg_collection.embd = parallel_state.get_embedding_group(check_initialized=False)
        pg_collection.pos_embd = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
    else:
        cp_group = pg_collection.cp
        cp_size = cp_group.size()
        tp_group = getattr(pg_collection, "tp", None)

    is_first = p2p_communicator.is_pp_first_stage
    is_last = p2p_communicator.is_pp_last_stage
    planned_shapes = (
        cp_size == 1
        and adjust_tensor_shapes_fn is None
        and not getattr(config, "mtp_standalone", False)
        and not getattr(config, "enable_hyper_connections", False)
    )
    if planned_shapes:
        p2p_communicator = copy.copy(p2p_communicator)
        p2p_communicator.config = copy.copy(config)
        p2p_communicator.config.variable_seq_lengths = False
        # Native individual P2P waits on requests without the batched API's device-wide sync.
        p2p_communicator.config.batch_p2p_comm = False
        if not getattr(runtime, "_planned_p2p_logged", False):
            logger.info("chunkflow: planned P2P shapes enabled; individual P2P skips shape exchange and batched sync")
            runtime._planned_p2p_logged = True

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

    # Synchronize gradients once, after all units.
    no_sync_func = config.no_sync_func or contextlib.nullcontext

    forward_data_store: list = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    fp_step_func = _no_loss_step_func(forward_step_func)
    first_forward = True

    with no_sync_func():
        num_parents = _num_microbatches(num_microbatches)
        parent_window = parent_window or max(1, num_parents)
        _check_ep_schedule((num_parents, parent_window), args, layout=True)
        for mb_start in range(0, num_parents, parent_window):
            by_id, owners, plans = {}, {}, []
            group_offset = chunk_offset = 0
            for mb_idx in range(mb_start, min(mb_start + parent_window, num_parents)):
                parent = _next_parent(data_iterator, args)
                chunks, units = _plan(parent, chunk_size, policy)
                plan = make_model_plan(units, chunks, keep_last_k, p2p_communicator.total_stages)
                plans.append(plan)
                groups = {
                    g: group_offset + i for i, g in enumerate(sorted({c.group for c in chunks if c.is_dependent}))
                }
                remapped = {
                    c.chunk_id: replace(c, chunk_id=chunk_offset + i, group=groups.get(c.group))
                    for i, c in enumerate(chunks)
                }
                for unit in units:
                    uid = len(by_id)
                    by_id[uid] = Unit(uid, [remapped[c.chunk_id] for c in unit.chunks])
                    owners[uid] = (parent, mb_idx)
                group_offset += len(groups)
                chunk_offset += len(chunks)
            plan = merge_model_plans(plans, p2p_communicator.total_stages)
            _check_ep_schedule((plan.forward, plan.backward, tuple(sorted(plan.recompute)), plan.warmups), args)
            fwd_seq, bwd_seq, need_fr = plan.forward, plan.backward, plan.recompute
            W, n = plan.warmups[p2p_communicator.current_stage], len(fwd_seq)
            unit_shapes = {
                uid: _unit_shapes(
                    unit,
                    p2p_communicator.config,
                    tp_group,
                    cp_group,
                    getattr(p2p_communicator, "pp_group", None),
                )
                if planned_shapes
                else (recv_tensor_shapes, send_tensor_shapes)
                for uid, unit in by_id.items()
            }

            def do_forward(unit_id, input_tensor):
                nonlocal first_forward
                parent, mb_idx = owners[unit_id]
                unit = by_id[unit_id]
                pad_token_id = getattr(args, "pad_token_id", 0) or 0
                batch = slice_parent_batch(parent, unit, pad_token_id=pad_token_id, args=args)
                batch["packed_seq_params"].__setattr__(UNIT_ATTR, unit)
                recompute = unit_id in need_fr
                if recompute:
                    batch["packed_seq_params"].__setattr__(RECOMPUTE_ATTR, False)
                with torch.no_grad() if recompute else contextlib.nullcontext():
                    output_tensor, num_tokens = forward_step(
                        fp_step_func if recompute else forward_step_func,
                        iter([(batch, {})]),
                        model,
                        1,
                        input_tensor,
                        [] if recompute else forward_data_store,
                        config,
                        cp_group_size=cp_size,
                        collect_non_loss_data=False,
                        is_first_microbatch=first_forward
                        if recompute
                        else check_first_val_step(first_val_step, forward_only, first_forward),
                        current_microbatch=mb_idx,
                        is_last_stage=is_last,
                    )
                first_forward = False
                if planned_shapes and not is_last:
                    _check_tensor_shapes(output_tensor, unit_shapes[unit_id][1])
                return output_tensor, input_tensor, None if recompute else num_tokens

            in_flight: dict[int, tuple] = {}
            fp_inputs: dict[int, torch.Tensor] = {}

            def backward(unit_id, output_tensor_grad):
                fr_num_tokens = None
                parent, mb_idx = owners.pop(unit_id)
                if unit_id in need_fr:
                    input_tensor = fp_inputs.pop(unit_id)
                    unit = by_id[unit_id]
                    batch = slice_parent_batch(
                        parent, unit, pad_token_id=getattr(args, "pad_token_id", 0) or 0, args=args
                    )
                    batch["packed_seq_params"].__setattr__(UNIT_ATTR, unit)
                    batch["packed_seq_params"].__setattr__(RECOMPUTE_ATTR, True)
                    output_tensor, fr_num_tokens = forward_step(
                        forward_step_func,
                        iter([(batch, {})]),
                        model,
                        1,
                        input_tensor,
                        forward_data_store,
                        config,
                        cp_group_size=cp_size,
                        collect_non_loss_data=False,
                        is_first_microbatch=False,
                        current_microbatch=mb_idx,
                        is_last_stage=is_last,
                    )
                    # Megatron custom_backward requires pseudo-freed outputs, including Fr.
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                else:
                    input_tensor, output_tensor = in_flight.pop(unit_id)
                input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)
                if planned_shapes and not is_first:
                    _check_tensor_shapes(input_tensor_grad, unit_shapes[unit_id][0])
                for group in completed_groups(by_id[unit_id]):
                    runtime.release_group(group)
                return input_tensor_grad, fr_num_tokens

            for i in range(W):
                input_tensor = (
                    None if is_first else p2p_communicator.recv_forward(unit_shapes[fwd_seq[i]][0], is_first)
                )
                output_tensor, fwd_input, num_tokens = do_forward(fwd_seq[i], input_tensor)
                if num_tokens is not None:
                    total_num_tokens += num_tokens
                    in_flight[fwd_seq[i]] = (fwd_input, output_tensor)
                else:
                    fp_inputs[fwd_seq[i]] = fwd_input
                if not is_last:
                    p2p_communicator.send_forward(output_tensor, is_last)
                if not is_last or num_tokens is not None:
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            n_steady = n - W
            input_tensor = None
            if n_steady > 0 and not is_first:
                input_tensor = p2p_communicator.recv_forward(unit_shapes[fwd_seq[W]][0], is_first)
            for i in range(n_steady):
                last_iter = i == n_steady - 1
                output_tensor, fwd_input, num_tokens = do_forward(fwd_seq[W + i], input_tensor)
                if num_tokens is not None:
                    total_num_tokens += num_tokens
                    in_flight[fwd_seq[W + i]] = (fwd_input, output_tensor)
                else:
                    fp_inputs[fwd_seq[W + i]] = fwd_input
                output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                    output_tensor, unit_shapes[bwd_seq[i]][1], is_last
                )
                if not is_last or num_tokens is not None:
                    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
                input_tensor_grad, fr_num_tokens = backward(bwd_seq[i], output_tensor_grad)
                if fr_num_tokens is not None:
                    total_num_tokens += fr_num_tokens
                if last_iter:
                    if not is_first:
                        p2p_communicator.send_backward(input_tensor_grad, is_first)
                else:
                    input_tensor = p2p_communicator.send_backward_recv_forward(
                        input_tensor_grad, unit_shapes[fwd_seq[W + i + 1]][0], is_first
                    )

            for i in range(n_steady, n):
                output_tensor_grad = (
                    None if is_last else p2p_communicator.recv_backward(unit_shapes[bwd_seq[i]][1], is_last)
                )
                input_tensor_grad, fr_num_tokens = backward(bwd_seq[i], output_tensor_grad)
                if fr_num_tokens is not None:
                    total_num_tokens += fr_num_tokens
                if not is_first:
                    p2p_communicator.send_backward(input_tensor_grad, is_first)

            if owners or in_flight or fp_inputs:
                raise RuntimeError(
                    f"chunkflow pipeline: parent window {mb_start} ended with {len(in_flight)} forwarded and "
                    f"{len(fp_inputs)} unrecomputed unit(s) -- schedule bug"
                )
            runtime.assert_empty(f"chunkflow PP={p2p_communicator.total_stages} parent window {mb_start}")

    _start_grad_sync(model, config)

    if config.finalize_model_grads_func is not None:
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    return forward_data_store
