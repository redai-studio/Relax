# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Install fused extraction only after the native pair specialization."""

_INSTALLED = None


def install(args, model):
    global _INSTALLED
    identity = tuple(id(c) for c in model)
    if _INSTALLED is not None:
        assert _INSTALLED == identity
        return None
    assert (
        args.tensor_model_parallel_size,
        args.context_parallel_size,
        args.pipeline_model_parallel_size,
        args.expert_model_parallel_size,
    ) == (4, 1, 1, 8)
    from megatron.core.transformer.attention import SelfAttention

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
        assert hasattr(module, "_qwen_kv_pair_group")
        assert module.world_size == 4 and module.config.num_query_groups == 2
        assert module.num_attention_heads_per_partition == 8 and module.num_query_groups_per_partition == 1
        assert module.hidden_size_per_attention_head == 256
    for name, module in targets.values():
        module._qwen_kv_fused_name = name
        module._qwen_qkv_fused = True
    _INSTALLED = identity
    return dict(
        attention_modules=11,
        names=[name for name, _ in targets.values()],
        pair_collectives_unchanged=True,
        native_projection_norm_and_fallback_preserved=True,
    )
