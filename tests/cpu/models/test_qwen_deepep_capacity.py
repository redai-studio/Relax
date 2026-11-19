# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Unequal packed lengths must take the same native DeepEP allocation and
dispatch paths."""

import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.models.qwen3_5.deepep_capacity import install_group_capacity


def _exercise_capacity(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    group = dist.group.WORLD
    calls = []
    allocations = []
    buffer = SimpleNamespace(num_max_tokens_per_rank=0)

    def get_buffer(actual_group, num_max_tokens_per_rank, hidden, num_topk):
        assert actual_group is group
        if buffer.num_max_tokens_per_rank < num_max_tokens_per_rank:
            allocations.append(num_max_tokens_per_rank)
            buffer.num_max_tokens_per_rank = num_max_tokens_per_rank
        return buffer

    def dispatch(actual_buffer, x, indices, probs, experts, *, num_max_tokens_per_rank, **kwargs):
        assert actual_buffer is buffer and num_max_tokens_per_rank <= buffer.num_max_tokens_per_rank
        assert x.shape[0] == indices.shape[0] == probs.shape[0] and probs.dtype == torch.float32
        calls.append((x.shape[0], num_max_tokens_per_rank))
        return x, indices, probs, [x.shape[0]], object()

    class NativeManager:
        def __init__(self):
            self.group = group
            self.num_experts = 2
            self.num_sms = 80

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_dispatcher = SimpleNamespace(_comm_manager=NativeManager())

        def forward(self, x):
            manager = self.token_dispatcher._comm_manager
            manager.token_indices = torch.zeros(len(x), 1, dtype=torch.long)
            manager.token_probs = torch.ones(len(x), 1, dtype=torch.bfloat16)
            return manager.dispatch(x)

    modules = {
        "megatron.core.transformer.moe": SimpleNamespace(
            fused_a2a=SimpleNamespace(get_elastic_buffer=get_buffer, deepepv2_dispatch=dispatch)
        ),
        "megatron.core.transformer.moe.token_dispatcher": SimpleNamespace(_DeepepV2Manager=NativeManager),
    }
    try:
        with patch.dict(sys.modules, modules), patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce:
            model = torch.nn.Sequential(Layer(), Layer())
            install_group_capacity(
                SimpleNamespace(pipeline_model_parallel_size=1, distributed_timeout_minutes=1), [model]
            )
            # Both start small. Rank 1 grows, then rank 0 alone needs the larger buffer.
            batches = [(4, 4), (4, 36), (36, 4), (2, 4)]
            for lengths, expected_capacity in zip(batches, [4, 36, 36, 36], strict=True):
                x = torch.randn(lengths[rank], 2)
                assert model(x) is x
                assert calls[-2:] == [(len(x), expected_capacity)] * 2
            assert allocations == [4, 36]
            assert reduce.call_count == len(batches)
    finally:
        dist.destroy_process_group(group)


def test_group_capacity_with_unequal_packed_rows(tmp_path):
    mp.spawn(_exercise_capacity, args=((tmp_path / "gloo").as_uri(),), nprocs=2)
