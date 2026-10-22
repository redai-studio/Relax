# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Absolute Bridge RoPE integration for a fresh isolated runtime only."""

import hashlib
import inspect


_CACHE = None
_STATS = {"calls": 0, "fallbacks": 0}


def install(args, model):
    global _CACHE
    if _CACHE is not None:
        return None
    assert (args.tensor_model_parallel_size, args.context_parallel_size, args.pipeline_model_parallel_size) == (
        4,
        1,
        1,
    )
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import rope
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention

    from relax.models.qwen3_5.rotary_cache import Cache

    targets = {}
    for part in model:
        root = part
        while hasattr(root, "module"):
            root = root.module
        root = getattr(root, "language_model", root)
        for name, m in root.named_modules():
            if isinstance(m, Qwen3VLSelfAttention):
                targets[id(m)] = m
    assert len(targets) == 11
    for m in targets.values():
        assert m.config.context_parallel_size == 1 and not m.config.rotary_interleaved
        assert m.config.kv_channels == 256 and not m.config.apply_rope_fusion
        assert not m.config.apply_rotary_pos_emb_in_fp32
    original = rope._apply_rotary_pos_emb_bshd
    _CACHE = Cache(2)
    return dict(
        attention_modules=len(targets),
        capacity=2,
        original_function_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
        entry="Bridge absolute BSHD reference",
        same_stream_identity_version_cache=True,
        absolute_frequencies_not_reindexed=True,
    )


def apply_rotary(original, *args, **kwargs):
    if _CACHE is None:
        return original(*args, **kwargs)
    from .rotary_cache import make_wrapper

    return make_wrapper(original, _CACHE, _STATS)(*args, **kwargs)
