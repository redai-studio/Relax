# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Materialize compact gate storage before the compiled output gate."""

import torch


def install(model, SelfAttention):
    targets = []
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for i, chunk in enumerate(chunks):
        while hasattr(chunk, "module"):
            chunk = chunk.module
        language = chunk.language_model
        for name, module in language.named_modules():
            if isinstance(module, SelfAttention):
                assert name.startswith(("decoder.layers.", "mtp.")), name
                targets.append((f"{i}:{name}", module))
    assert len(targets) == 11 and len({id(m) for _, m in targets}) == 11
    assert sum(":decoder.layers." in n for n, _ in targets) == 10
    assert sum(":mtp." in n for n, _ in targets) == 1
    for _, module in targets:
        assert module.config.tensor_model_parallel_size == 4
        assert module.config.context_parallel_size == 1
        assert callable(module._apply_output_gate)
    return [{"name": name, "witness": None} for name, _ in targets]


@torch.compiler.disable(recursive=False)
def compact_gate(gate):
    assert gate.ndim == 4 and gate.dtype == torch.bfloat16
    return gate.contiguous()
