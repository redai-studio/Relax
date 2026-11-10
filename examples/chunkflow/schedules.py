# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

import contextlib
import os
from argparse import Namespace
from collections.abc import Callable
from typing import Iterator, List, Optional, Union

import torch
from megatron.core import parallel_state

from examples.chunkflow.runtime import RECOMPUTE_ATTR, UNIT_ATTR, require_runtime
from examples.chunkflow.scheduler import build_chunks
from examples.chunkflow.unit_batch import slice_parent_batch
from examples.chunkflow.units import build_micro_batches, completed_groups, gen_unit_ops


_BATCH_KEYS = (
    "tokens",
    "pre_shifted_labels",
    "packed_seq_params",
    "total_lengths",
    "response_lengths",
    "loss_masks",
    "log_probs",
    "ref_log_probs",
    "values",
    "advantages",
    "returns",
    "rollout_log_probs",
    "max_seq_lens",
)


def get_schedule(args: Namespace, pp_size: int) -> Callable:
    if pp_size <= 1:
        return chunkflow_forward_backward_no_pipelining
    if os.environ.get("RELAX_CHUNKFLOW_PIPELINE_UNTESTED") != "1":
        raise NotImplementedError(
            "chunkflow's experimental pipeline executor requires RELAX_CHUNKFLOW_PIPELINE_UNTESTED=1."
        )
    from .pipeline_schedules import chunkflow_forward_backward_pipelining_without_interleaving

    return chunkflow_forward_backward_pipelining_without_interleaving


def _plan(parent: dict, chunk_size: int, policy: str):
    cu = parent["packed_seq_params"].cu_seqlens_q_cpu
    lengths = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    source_lengths = parent.get("total_lengths")
    if source_lengths is not None and len(lengths) > 1 and lengths[:-1] == list(source_lengths):
        lengths.pop()  # get_batch appends alignment padding as a separate sequence.
    chunks = build_chunks(lengths, chunk_size=chunk_size)
    return chunks, build_micro_batches(chunks, policy)


def _num_microbatches(num_microbatches) -> int:
    if isinstance(num_microbatches, (list, tuple)):
        num_microbatches = num_microbatches[0]
    elif torch.is_tensor(num_microbatches):
        num_microbatches = int(num_microbatches.reshape(-1)[0])
    return int(num_microbatches)


def chunkflow_forward_backward_no_pipelining(
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
    **kwargs,
):
    del seq_length, micro_batch_size, decoder_seq_length, collect_non_loss_data, first_val_step
    if isinstance(model, list):
        assert len(model) == 1, "chunkflow schedule requires a single model chunk"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, "chunkflow schedule requires a single iterator"
        data_iterator = data_iterator[0]

    num_mb = _num_microbatches(num_microbatches)

    from megatron.training.global_vars import get_args

    from relax.backends.megatron.data import get_batch

    args = get_args()
    chunk_size = int(args.chunkflow_chunk_size)
    policy = getattr(args, "chunkflow_pack_policy", "staggered")
    # K=0 retains all activations; otherwise earlier chunks use Fp/Fr.
    keep_last_k = getattr(args, "chunkflow_keep_last_k", 0) or 1 << 30
    runtime = require_runtime(model)

    config = model.config if hasattr(model, "config") else None
    no_sync_func = getattr(config, "no_sync_func", None) or contextlib.nullcontext

    forward_data_store: list = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    with no_sync_func():
        for _ in range(num_mb):
            parent_batch = data_iterator.get_next(_BATCH_KEYS)
            parent = get_batch(
                iter([(parent_batch, {})]),
                _BATCH_KEYS,
                args.data_pad_size_multiplier,
                args.qkv_format,
                args.allgather_cp,
            )
            chunks, units = _plan(parent, chunk_size, policy)
            ops = gen_unit_ops(units, chunks, keep_last_k)

            pending: dict[int, torch.Tensor] = {}
            for kind, unit_id in ops:
                unit = units[unit_id]
                if kind == "B":
                    loss = pending.pop(unit_id)
                    if not forward_only:
                        loss.backward()
                    for group in completed_groups(unit):
                        runtime.release_group(group)
                    continue

                batch = slice_parent_batch(parent, unit, pad_token_id=getattr(args, "pad_token_id", 0) or 0, args=args)
                batch["packed_seq_params"].__setattr__(UNIT_ATTR, unit)
                batch["packed_seq_params"].__setattr__(RECOMPUTE_ATTR, kind == "Fr")
                with torch.no_grad() if kind == "Fp" else contextlib.nullcontext():
                    output, loss_callback = forward_step_func(iter([(batch, {})]), model)
                if kind == "Fp":
                    del output, loss_callback
                    continue

                loss, num_tokens, log = loss_callback(output)
                total_num_tokens += num_tokens
                forward_data_store.append(log)
                pending[unit_id] = loss

            runtime.assert_empty("chunkflow PP=1 micro-batch")

    if getattr(config, "finalize_model_grads_func", None) is not None and not forward_only:
        from megatron.core.pipeline_parallel.schedules import ProcessGroupCollection

        pg = kwargs.get("pg_collection")
        if pg is None:
            pg = ProcessGroupCollection()
            pg.tp = parallel_state.get_tensor_model_parallel_group()
            pg.cp = parallel_state.get_context_parallel_group()
            pg.pp = parallel_state.get_pipeline_model_parallel_group()
            pg.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=False)
            pg.embd = parallel_state.get_embedding_group(check_initialized=False)
            pg.pos_embd = parallel_state.get_position_embedding_group(check_initialized=False)
            pg.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if getattr(config, "calculate_per_token_loss", False) else None,
            pg_collection=pg,
            force_all_reduce=kwargs.get("force_all_reduce", False),
        )

    return forward_data_store
