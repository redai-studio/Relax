# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Safety checks for Gemma-4's packed-only core attention."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


# The unit-test image intentionally carries an older Megatron tree without the
# Gemma-4 bridge classes. Load the leaf module directly so these attention tests
# do not execute relax.models.gemma4.__init__ and fail on an unrelated optional
# integration dependency.
_MODULE_PATH = Path(__file__).parents[3] / "relax/models/gemma4/attention.py"
_SPEC = importlib.util.spec_from_file_location("_relax_gemma4_attention_test_target", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Gemma4CoreAttention = _MODULE.Gemma4CoreAttention


def _attention():
    config = SimpleNamespace(
        attention_dropout=0.0,
        context_parallel_size=1,
        softmax_scale=1.0,
        window_size=(1023, 0),
        window_attn_skip_freq=6,
    )
    return Gemma4CoreAttention(config, layer_number=1)


def test_thd_without_sequence_boundaries_fails_loudly():
    attention = _attention()
    q = torch.zeros(8, 2, 4)
    with pytest.raises(NotImplementedError, match="requires packed THD inputs with cu_seqlens"):
        attention(q, q, q)


def test_sbhd_fails_instead_of_silently_dropping_sliding_window():
    attention = _attention()
    q = torch.zeros(8, 1, 2, 4)
    with pytest.raises(NotImplementedError, match="supports packed THD attention only"):
        attention(q, q, q)


def test_flash_path_reuses_packed_max_seqlen(monkeypatch):
    captured = {}

    def fake_flash_attn_varlen_func(query, key, value, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(query)

    monkeypatch.setitem(
        sys.modules,
        "flash_attn",
        SimpleNamespace(flash_attn_varlen_func=fake_flash_attn_varlen_func),
    )
    attention = _attention()
    q = torch.zeros(8, 2, 4)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 8]), max_seqlen_q=123)

    attention(q, q, q, packed_seq_params=packed)

    assert captured["max_seqlen_q"] == 123
    assert captured["max_seqlen_k"] == 123


def test_sdpa_path_reuses_cpu_sequence_boundaries():
    class BoundarySpy:
        def detach(self):
            raise AssertionError("cached CPU boundaries should avoid accelerator synchronization")

    attention = _attention()
    q = torch.zeros(8, 2, 257)
    packed = SimpleNamespace(
        cu_seqlens_q=BoundarySpy(),
        cu_seqlens_q_cpu=[0, 3, 8],
        max_seqlen_q=5,
    )

    output = attention(q, q, q, packed_seq_params=packed)

    assert output.shape == (8, 514)
