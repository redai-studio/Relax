# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Owned pair-AG specialization of native TP4 Qwen gated QKV extraction."""

from typing import Any


_INSTALLED = None


def pair_plan(tp_groups: list[list[int]], world_size: int) -> list[list[int]]:
    groups = sorted({tuple(g) for g in tp_groups})
    assert len(groups) * 4 == world_size and sorted(x for g in groups for x in g) == list(range(world_size))
    assert all(len(g) == 4 and len(set(g)) == 4 for g in groups)
    # Preserve actual TP group rank ordering; do not infer EP or WORLD adjacency.
    return [list(g[i : i + 2]) for g in groups for i in (0, 2)]


def install(args: Any, model: list) -> None:
    global _INSTALLED
    identity = tuple(id(c) for c in model)
    if _INSTALLED is not None:
        assert identity == _INSTALLED
        return
    import torch.distributed as dist
    from megatron.core import mpu, parallel_state
    from megatron.core.transformer.attention import SelfAttention

    assert (
        args.tensor_model_parallel_size,
        args.context_parallel_size,
        args.pipeline_model_parallel_size,
        args.expert_model_parallel_size,
        args.expert_tensor_parallel_size,
    ) == (4, 1, 1, 8, 1)
    world = dist.get_world_size(group=dist.group.WORLD)
    assert world in (4, 64)
    tp = mpu.get_tensor_model_parallel_group()
    local = list(dist.get_process_group_ranks(tp))
    reports = [None] * world
    dist.all_gather_object(reports, local, group=dist.group.WORLD)
    plan = pair_plan(reports, world)
    rank = dist.get_rank(group=dist.group.WORLD)
    selected = None
    # All WORLD ranks create every pair in the exact same order, once at startup.
    for ranks in plan:
        group = parallel_state.create_group(ranks=ranks, backend="nccl", group_desc="qwen-tp4-kv-pair")
        if rank in ranks:
            selected = group
    assert (
        selected is not None
        and dist.get_process_group_ranks(selected)
        == local[(local.index(rank) // 2) * 2 : (local.index(rank) // 2 + 1) * 2]
    )
    targets = {}
    for chunk in model:
        root = chunk
        while hasattr(root, "module"):
            root = root.module
        root = getattr(root, "language_model", root)
        for name, module in root.named_modules():
            if isinstance(module, SelfAttention):
                targets[id(module)] = (name, module)
    assert len(targets) == 11
    for name, module in targets.values():
        assert module.world_size == 4 and module.config.num_query_groups == 2
        assert module.hidden_size_per_attention_head == 256
        assert module.pg_collection.tp is tp
        assert module.get_query_key_value_tensors.__func__ is SelfAttention.get_query_key_value_tensors
    for name, module in targets.values():
        module._qwen_kv_pair_group = selected
    _INSTALLED = identity
