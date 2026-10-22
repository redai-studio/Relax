# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch


@pytest.mark.parametrize(
    "limit,length,expected",
    [
        (None, 2052, "auto"),
        (None, 16384, "auto"),
        (None, 16385, "grouped"),
        ("0", 2051, "auto"),
        ("0", 2052, "grouped"),
        ("4096", 4096, "auto"),
        ("4096", 4097, "grouped"),
    ],
)
def test_mixed_threshold_keeps_dense_short_and_routes_sparse(monkeypatch, limit, length, expected):
    from relax.models.qwen3_8.ops.qsa_mixed_ops import mixed

    if limit is None:
        monkeypatch.delenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", raising=False)
    else:
        monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", limit)
    seen = []

    def call(kind):
        def forward(q, *_args):
            seen.append(kind)
            return q

        return forward

    monkeypatch.setattr(mixed, "attention_auto", call("auto"))
    monkeypatch.setattr(mixed, "long_attention", call("grouped"))
    q = torch.empty(length, 12, 256, device="meta")
    kv = torch.empty(length, 1, 256, device="meta")
    indices = torch.empty(length, 2052, dtype=torch.int32, device="meta")
    assert mixed.attention(q, kv, kv, indices, 256**-0.5) is q
    assert seen == [expected]


@pytest.mark.parametrize("limit", [-1, 16385])
def test_mixed_threshold_rejects_unsupported_flex_layout(monkeypatch, limit):
    from relax.models.qwen3_8.ops.qsa_mixed_ops import mixed

    monkeypatch.setenv("RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN", str(limit))
    q = torch.empty(2052, 12, 256, device="meta")
    kv = torch.empty(2052, 1, 256, device="meta")
    indices = torch.empty(2052, 2052, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="maximum length"):
        mixed.attention(q, kv, kv, indices, 256**-0.5)
