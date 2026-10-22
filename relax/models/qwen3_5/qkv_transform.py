# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Specialize native QKV extraction with guarded fallback."""


def _supported(module, mixed, key_value_states, output_gate, head_wise_gate, split_qkv):
    import torch
    import torch.distributed as dist

    return (
        output_gate
        and not head_wise_gate
        and split_qkv
        and key_value_states is None
        and module.world_size == 4
        and module.config.context_parallel_size == 1
        and module.config.num_query_groups == 2
        and module.num_attention_heads_per_partition == 8
        and module.num_query_groups_per_partition == 1
        and module.hidden_size_per_attention_head == 256
        and mixed.is_cuda
        and mixed.dtype == torch.bfloat16
        and mixed.ndim == 3
        and 0 < mixed.shape[0] <= 262144
        and mixed.shape[1:] == (1, 2304)
        and mixed.is_contiguous()
        and dist.get_world_size(module._qwen_kv_pair_group) == 2
        and dist.get_rank(module._qwen_kv_pair_group) == dist.get_rank(module.pg_collection.tp) % 2
    )
