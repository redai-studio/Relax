# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Retain all real-document queries; restore the public THD physical shape."""

from dataclasses import dataclass, fields, replace
from functools import wraps

import torch.nn.functional as F

from relax.models.qwen3_8.ops.qsa_mixed_ops.selection import MixedQSASelection


@dataclass
class RealDocsSelection(MixedQSASelection):
    _q38_valid_end: int | None = None


def attach(selection, packed):
    bounds = getattr(selection, "_relax_qsa_mixed_bounds", None)
    valid = getattr(packed, "valid_seqlens_q_cpu", None)
    n = getattr(packed, "num_valid_tokens", None)
    if (
        not bounds
        or valid is None
        or len(bounds) != len(valid) + 1
        or not isinstance(n, int)
        or not 0 < n < bounds[-1]
        or bounds[-2] != n
        or valid[-1] != 0
        or tuple(valid[:-1]) != tuple(b - a for a, b in zip(bounds[:-2], bounds[1:-1]))
    ):
        return selection
    return RealDocsSelection(
        **{f.name: getattr(selection, f.name) for f in fields(MixedQSASelection)}, _q38_valid_end=n
    )


def adapt(native):
    from relax.utils.logging_utils import get_logger

    reported = set()

    @wraps(native)
    def attention(q, k, v, selection, scale, is_thd):
        n = getattr(selection, "_q38_valid_end", None)
        if n is None or not is_thd:
            return native(q, k, v, selection, scale, is_thd)
        total = q.shape[1]
        if q.shape[0] != 1 or not 0 < n < total or selection._relax_qsa_mixed_bounds[-2:] != (n, total):
            raise RuntimeError("THD padding boundary changed after QSA selection")
        ids = selection.selected_ids.reshape(total, -1)[:n]
        comp = replace(
            selection,
            positions=selection.positions[:, :n],
            selected_ids=ids,
            doc_ids=selection.doc_ids[:, :n] if selection.doc_ids is not None else None,
            _relax_qsa_mixed_bounds=selection._relax_qsa_mixed_bounds[:-1],
            _q38_valid_end=None,
        )
        if not reported:
            get_logger(__name__).info(
                "QSA computes all %d real packed queries, omitting %d physical tail positions", n, total - n
            )
            reported.add(True)
        y = native(q[:, :n], k[:, :n], v[:, :n], comp, scale, is_thd)
        return F.pad(y, (0, 0, 0, 0, 0, total - n))

    return attention


def install():
    from relax.models.qwen3_8.ops import qsa_mixed_ops as mixed

    selector = mixed.forward

    @wraps(selector)
    def forward(self, hidden_states, rotary_pos_emb, packed_seq_params=None, output_format="bits"):
        s = selector(self, hidden_states, rotary_pos_emb, packed_seq_params, output_format)
        return attach(s, packed_seq_params)

    mixed.forward = forward
    mixed.attention_for_selection = adapt(mixed.attention_for_selection)
