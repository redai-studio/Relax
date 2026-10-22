# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Vendored community source; original notices follow.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process-local counters for which QSA block selector and kernel actually ran.

QSA has one production selector -- the exact, deterministic TileLang radix
Top-K -- and three PyTorch paths that must never run underneath it: the
``torch.topk`` route oracle in :func:`select_qsa_token_ids`, the streaming
merge selector used by the ``none`` backend, and the dense ``[B, S, P]``
mask/Top-K support builder.  "The TileLang backend was requested" is not
evidence that the TileLang selector ran, so tests assert on these counters
instead of on configuration.

The same argument applies one level down to the two remaining QSA operators.
``tilelang_sparse_gqa_fwd`` / ``tilelang_sparse_gqa_bwd`` and ``tilelang_kl``
record that the fused kernels ran, while ``torch_gqa`` and ``torch_kl`` record
their PyTorch oracles.  An integration that silently degraded to the oracles
would otherwise still produce a finite loss and a plausible gradient.

The counters are also how activation-recompute tests check that the recomputed
forward really re-ran exact Top-K rather than reusing a cached route.
"""

from __future__ import annotations

from typing import Dict


_COUNTER_NAMES = (
    "exact_topk",
    "exact_topk_rows",
    "torch_topk",
    "streaming_topk",
    "dense_mask",
    "tilelang_sparse_gqa_fwd",
    "tilelang_sparse_gqa_bwd",
    "torch_gqa",
    "tilelang_kl",
    "torch_kl",
)

_COUNTERS: Dict[str, int] = {name: 0 for name in _COUNTER_NAMES}


def count_qsa_selector_call(name: str, *, rows: int = 0) -> None:
    """Record one invocation of a named QSA block selector or kernel."""
    if name not in _COUNTERS:
        raise KeyError(f"unknown QSA selector counter {name!r}")
    _COUNTERS[name] += 1
    if name == "exact_topk":
        _COUNTERS["exact_topk_rows"] += int(rows)


def reset_qsa_selector_counters() -> None:
    """Zero every selector counter."""
    for name in _COUNTERS:
        _COUNTERS[name] = 0


def get_qsa_selector_counters() -> Dict[str, int]:
    """Return a snapshot of the selector counters."""
    return dict(_COUNTERS)


__all__ = ["count_qsa_selector_call", "get_qsa_selector_counters", "reset_qsa_selector_counters"]
