# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Agree native DeepEP v2 buffer sizes for variable-length Qwen SFT
microbatches."""

from argparse import Namespace
from datetime import timedelta
from functools import partial
from typing import Any

import torch
import torch.distributed as dist


class _ExpertCapacity:
    def __init__(self, metadata_group: dist.ProcessGroup) -> None:
        self.group = metadata_group
        self.rows = 0
        self.needs_max = True

    def begin_forward(self, module: torch.nn.Module, inputs: tuple) -> None:
        self.needs_max = True

    def for_rows(self, local_rows: int) -> int:
        # All decoder and MTP MoE layers see the same packed sequence. Negotiate
        # once on the first dispatch, including when only a remote rank grew.
        if self.needs_max:
            rows = torch.tensor(local_rows, device="cpu", dtype=torch.int64)
            dist.all_reduce(rows, op=dist.ReduceOp.MAX, group=self.group)
            # Recomputed layers may revisit an earlier, longer microbatch.
            self.rows = max(self.rows, int(rows))
            self.needs_max = False
        if local_rows > self.rows:
            raise RuntimeError(f"DeepEP input grew within one Qwen forward: {local_rows} > {self.rows}")
        return self.rows


def _dispatch(
    manager: Any,
    capacity: _ExpertCapacity,
    hidden_states: torch.Tensor,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
) -> torch.Tensor:
    from megatron.core.transformer.moe import fused_a2a

    rows = capacity.for_rows(hidden_states.shape[0])
    manager.buffer = fused_a2a.get_elastic_buffer(
        manager.group,
        num_max_tokens_per_rank=rows,
        hidden=hidden_states.shape[1],
        num_topk=manager.token_indices.shape[1],
    )
    manager.token_probs = manager.token_probs.float()
    output, manager.dispatched_indices, manager.dispatched_probs, manager.tokens_per_expert, manager.handle = (
        fused_a2a.deepepv2_dispatch(
            manager.buffer,
            hidden_states,
            manager.token_indices,
            manager.token_probs,
            manager.num_experts,
            num_max_tokens_per_rank=rows,
            expert_alignment=1,
            num_sms=manager.num_sms,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    )
    return output


def install_group_capacity(args: Namespace, model: list[torch.nn.Module]) -> None:
    from megatron.core.transformer.moe.token_dispatcher import _DeepepV2Manager

    if len(model) != 1 or args.pipeline_model_parallel_size != 1:
        raise ValueError("The Qwen variable-length DeepEP hook requires PP1 without virtual stages")
    chunk = model[0]

    managers = []
    for layer in chunk.modules():
        manager = getattr(getattr(layer, "token_dispatcher", None), "_comm_manager", None)
        if manager is not None:
            managers.append(manager)
    if not managers or any(type(manager) is not _DeepepV2Manager for manager in managers):
        raise ValueError("The Qwen variable-length DeepEP hook requires native DeepEP v2 managers")
    group = managers[0].group
    if any(manager.group is not group for manager in managers):
        raise ValueError("Qwen decoder and MTP layers must use the same expert communication group")

    # Expert groups are disjoint in this recipe. Only their members join the
    # matching CPU group; the training tensors stay on their original devices.
    metadata_group = dist.new_group(
        ranks=dist.get_process_group_ranks(group),
        backend="gloo",
        use_local_synchronization=True,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
    capacity = _ExpertCapacity(metadata_group)
    chunk.register_forward_pre_hook(capacity.begin_forward)
    for manager in managers:
        manager.dispatch = partial(_dispatch, manager, capacity)
