# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit QSA execution stages used by the build-time Megatron
integration."""

import os

from relax.utils.env import Envs


def cpu_forward(self, *args, **kwargs):
    if Envs.RELAX_QWEN38_QSA_CPU_LAYOUT:
        from .qsa_cpu_layout import _indexer_forward

        return _indexer_forward(self, *args, **kwargs)
    return self._relax_native_forward(*args, **kwargs)


def tp_forward(self, *args, **kwargs):
    if os.environ.get("RELAX_QWEN38_QSA_TP_SELECTOR", "0") == "1":
        if Envs.RELAX_QWEN38_QSA_COMMUNITY_BACKEND != "mixed":
            raise ValueError("TP QSA selector requires the mixed community attention backend")
        from .ops.qsa_mixed_ops.selector import forward

        return forward(self, *args, **kwargs)
    return cpu_forward(self, *args, **kwargs)


def replay_forward(self, *args, **kwargs):
    if Envs.RELAX_QWEN38_QSA_RECOMPUTE_REPLAY:
        from .qsa_cp import _indexer_forward

        return _indexer_forward(self, *args, **kwargs)
    return tp_forward(self, *args, **kwargs)


def selector_forward(self, *args, **kwargs):
    if Envs.RELAX_QWEN38_QSA_FUSED_SELECTOR:
        from .qsa_select import indexer_forward

        return indexer_forward(self, *args, **kwargs)
    return replay_forward(self, *args, **kwargs)


def indexer_forward(self, *args, **kwargs):
    if Envs.RELAX_QWEN38_QSA_COMMUNITY_BACKEND == "mixed":
        from .ops.qsa_mixed_ops import forward

        return forward(self, *args, **kwargs)
    return selector_forward(self, *args, **kwargs)


def pool_keys(self, *args, **kwargs):
    if Envs.RELAX_QWEN38_QSA_FIXED_POOL:
        from .qsa_fixed_pool import pool_keys as fixed_pool

        return fixed_pool(self, *args, **kwargs)
    if Envs.RELAX_QWEN38_QSA_CPU_LAYOUT:
        from .qsa_cpu_layout import _pool_keys

        return _pool_keys(self, *args, **kwargs)
    return self._relax_native_pool_keys(*args, **kwargs)


def cpu_select(self, *args, **kwargs):
    if Envs.RELAX_QWEN38_QSA_CPU_LAYOUT:
        from .qsa_cpu_layout import _select_blocks

        return _select_blocks(self, *args, **kwargs)
    return self._relax_native_select_blocks(*args, **kwargs)


def select_blocks(self, *args, **kwargs):
    if Envs.RELAX_QWEN38_QSA_FUSED_SELECTOR:
        from .qsa_select import select_blocks as fused_select

        return fused_select(self, *args, **kwargs)
    return cpu_select(self, *args, **kwargs)


def block_flex(self, *args, **kwargs):
    if Envs.RELAX_QWEN38_QSA_SPARSE_BLOCKMASK:
        from .qsa_block_mask import _sparse_flex_forward

        return _sparse_flex_forward(self, *args, **kwargs)
    return self._relax_native_flex_forward(*args, **kwargs)


def flex_forward(self, *args, **kwargs):
    if Envs.RELAX_QWEN38_QSA_FUSED_ATTENTION:
        from .qsa_fused import _fused_flex_forward

        return _fused_flex_forward(self, *args, **kwargs)
    return block_flex(self, *args, **kwargs)
