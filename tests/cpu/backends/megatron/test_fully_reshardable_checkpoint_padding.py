# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch


def test_fully_reshardable_checkpoint_slices_coalesced_optimizer_states():
    optimizer_module = pytest.importorskip("megatron.core.optimizer.distrib_optimizer", exc_type=ImportError)
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    param_sizes = (4, 3, 2)
    bucket_size = 8
    model_state = {}
    param_index_map = {}
    buckets = []
    for bucket_id, size in enumerate(param_sizes):
        name = f"param_{bucket_id}"
        parameter = torch.nn.Parameter(torch.zeros(size))
        model_state[name] = ShardedTensor.from_rank_offsets(name, parameter)
        start = bucket_id * bucket_size
        param_index_map[parameter] = (start, start + size, bucket_id)
        buckets.append(SimpleNamespace(grad_data=torch.empty(bucket_size), numel_unpadded=size))

    # The gather concatenates each bucket's unpadded contents.
    total_size = sum(param_sizes)
    packed_states = {
        key: torch.arange(base, base + total_size, dtype=torch.float32)
        for key, base in (("param", 1), ("exp_avg", 21), ("exp_avg_sq", 41))
    }

    def gather(**_kwargs):
        return {0: {torch.float32: {**packed_states, "numel_unpadded": total_size}}}

    optimizer = SimpleNamespace(
        buffers=[SimpleNamespace(buckets=buckets, param_index_map=param_index_map, numel_unpadded=total_size)],
        gbuf_ranges=[{torch.float32: [{}, {}, {}]}],
        get_parameter_state_dp_zero=gather,
    )
    original_indices = param_index_map.copy()

    state = optimizer_module.DistributedOptimizer.sharded_param_state_fully_reshardable(
        optimizer, model_state, metadata={}
    )

    assert list(state) == [0, 1, 2]
    compact_start = 0
    for bucket_id, size in enumerate(param_sizes):
        assert state[bucket_id].keys() == packed_states.keys()
        compact_end = compact_start + size
        for key, packed in packed_states.items():
            shard = state[bucket_id][key]
            assert shard.key == f"optimizer.state.{key}.param_{bucket_id}"
            torch.testing.assert_close(shard.data, packed[compact_start:compact_end], rtol=0, atol=0)
        compact_start = compact_end
    assert optimizer.buffers[0].param_index_map is param_index_map
    assert param_index_map == original_indices


def test_dp_reshardable_load_skips_alignment_padding_and_restores_next_bucket():
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    parameter = torch.nn.Parameter(torch.zeros(4))
    restored = {}

    def restore(model_param, state):
        restored[model_param] = state["exp_avg"]

    optimizer = SimpleNamespace(
        data_parallel_group=SimpleNamespace(rank=lambda: 0, size=lambda: 2),
        buffers=[SimpleNamespace(buckets=[SimpleNamespace(grad_data=torch.empty(8), numel_unpadded=8)] * 2)],
        gbuf_ranges=[{torch.float32: [{"param_map": {}}, {"param_map": {parameter: {}}}]}],
        _set_main_param_and_optimizer_states=restore,
    )
    exp_avg = torch.arange(1.0, 5.0)
    state = {0: {torch.float32: [[{"padding": True}], [{"padding": False, "exp_avg": exp_avg}]]}}

    DistributedOptimizer.load_parameter_state_from_dp_reshardable(optimizer, state)

    assert list(restored) == [parameter]
    torch.testing.assert_close(restored[parameter], exp_avg)
