# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Materialize compact value storage before the attention autograd boundary."""


def install(model, SelfAttention):
    chunks = model if isinstance(model, (list, tuple)) else [model]
    targets = []
    for i, chunk in enumerate(chunks):
        while hasattr(chunk, "module"):
            chunk = chunk.module
        for name, module in chunk.language_model.named_modules():
            if isinstance(module, SelfAttention):
                assert name.startswith(("decoder.layers.", "mtp.")), name
                targets.append((f"{i}:{name}", module.core_attention))
    assert len(targets) == 11 and len({id(m) for _, m in targets}) == 11
    assert sum(":decoder.layers." in n for n, _ in targets) == 10
    assert sum(":mtp." in n for n, _ in targets) == 1
    for _, core in targets:
        c = core.config
        assert c.tensor_model_parallel_size == 4 and c.context_parallel_size == 1
        assert c.kv_channels == 256 and c.num_query_groups == 4
        assert callable(core.forward)
    return [{"name": name, "witness": None} for name, _ in targets]
