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


def test_import_does_not_compile_flex_or_change_dynamo_defaults(monkeypatch):
    original_limit = torch._dynamo.config.recompile_limit

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("Importing the default attention path must not compile Flex")

    monkeypatch.setattr(torch, "compile", unexpected_compile)
    spec = importlib.util.spec_from_file_location("_gemma4_no_flex_import", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert torch._dynamo.config.recompile_limit == original_limit


def test_flex_compile_budget_is_scoped_to_invocation(monkeypatch):
    original_limit = torch._dynamo.config.recompile_limit
    calls = []

    def fake_compile(function, **options):
        calls.append(options)

        def compiled():
            assert torch._dynamo.config.recompile_limit >= 64
            raise RuntimeError("test compilation failure")

        return compiled

    _MODULE._get_compiled_flex_attention.cache_clear()
    monkeypatch.setattr(torch, "compile", fake_compile)
    function = _MODULE._get_compiled_flex_attention()
    assert torch._dynamo.config.recompile_limit == original_limit
    with pytest.raises(RuntimeError, match="test compilation failure"):
        function()
    assert torch._dynamo.config.recompile_limit == original_limit
    assert calls == [{"fullgraph": True, "dynamic": True}]
    _MODULE._get_compiled_flex_attention.cache_clear()


@pytest.mark.parametrize("query_length,query_offset", [(1, 0), (127, 0), (129, 128), (145, 256)])
def test_causal_flex_mask_matches_sdpa(query_length, query_offset):
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(31)
    key_length = query_length + query_offset
    query = torch.randn(1, 2, query_length, 32) * 0.1
    key = torch.randn(1, 1, key_length, 32) * 0.1
    value = torch.randn_like(key)
    block_mask = _MODULE._causal_block_mask(torch.device("cpu"), query_length, key_length, query_offset)
    output = flex_attention(query, key, value, block_mask=block_mask, enable_gqa=True)
    mask = torch.arange(key_length)[None, :] <= torch.arange(query_length)[:, None] + query_offset
    expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, enable_gqa=True)
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)


def test_flex_global_dispatch_is_opt_in(monkeypatch):
    config = SimpleNamespace(
        attention_dropout=0.0,
        context_parallel_size=1,
        softmax_scale=1.0,
        window_size=(1023, 0),
        window_attn_skip_freq=6,
    )
    monkeypatch.delenv("RELAX_GEMMA4_FORCE_FLEX_GLOBAL", raising=False)
    assert not Gemma4CoreAttention(config, layer_number=6).force_flex_global
    monkeypatch.setenv("RELAX_GEMMA4_FORCE_FLEX_GLOBAL", "1")
    assert not Gemma4CoreAttention(config, layer_number=1).force_flex_global
    attention = Gemma4CoreAttention(config, layer_number=6)
    called = []
    monkeypatch.setattr(attention, "_thd_flex_per_subseq", lambda q, k, v, b: called.append(b) or q.flatten(1))
    query = torch.zeros(8, 2, 512)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 8]), cu_seqlens_q_cpu=[0, 3, 8], max_seqlen_q=5)
    output = attention(query, query, query, packed_seq_params=packed)
    assert output.shape == (8, 1024)
    assert called == [[0, 3, 8]]


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention"),
        ),
        pytest.param(
            "cuda",
            torch.bfloat16,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention"),
        ),
    ],
)
def test_flex_packed_chunk_boundaries_preserve_outputs_and_gradients(monkeypatch, device, dtype):
    torch.manual_seed(32)
    monkeypatch.setattr(_MODULE, "_FLEX_QUERY_CHUNK_SIZE", 128)
    attention = _attention()
    lengths = [129, 145]
    boundaries = [0, 129, sum(lengths)]
    query = (torch.randn(sum(lengths), 2, 512, device=device, dtype=dtype) * 0.1).requires_grad_(device == "cuda")
    key = (torch.randn(sum(lengths), 1, 512, device=device, dtype=dtype) * 0.1).requires_grad_(device == "cuda")
    value = torch.randn_like(key, requires_grad=device == "cuda")
    output = attention._thd_flex_per_subseq(query, key, value, boundaries)
    references = []
    for start, end in zip(boundaries, boundaries[1:]):
        expected = torch.nn.functional.scaled_dot_product_attention(
            query[start:end].transpose(0, 1).unsqueeze(0),
            key[start:end].transpose(0, 1).unsqueeze(0),
            value[start:end].transpose(0, 1).unsqueeze(0),
            scale=1.0,
            is_causal=True,
            enable_gqa=True,
        )
        references.append(expected.squeeze(0).transpose(0, 1).flatten(1))
    reference = torch.cat(references)
    atol, rtol = (3e-3, 2e-2) if dtype == torch.bfloat16 else (3e-5, 1e-4)
    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
    if device == "cuda":
        upstream = torch.randn_like(output)
        actual_grads = torch.autograd.grad(output, (query, key, value), upstream)
        reference_grads = torch.autograd.grad(reference, (query, key, value), upstream)
        for actual, expected in zip(actual_grads, reference_grads, strict=True):
            torch.testing.assert_close(actual, expected, atol=max(atol, 5e-5), rtol=max(rtol, 3e-4))
    with torch.no_grad():
        changed_key = key.detach().clone()
        changed_value = value.detach().clone()
        changed_key[129:] += 10
        changed_value[129:] += 10
        changed = attention._thd_flex_per_subseq(query.detach(), changed_key, changed_value, boundaries)
    torch.testing.assert_close(output[:129], changed[:129])


def test_flex_rejects_training_dropout():
    attention = _attention()
    attention.dropout_p = 0.1
    query = torch.zeros(8, 2, 512)
    with pytest.raises(NotImplementedError, match="dropout to be zero"):
        attention._thd_flex_per_subseq(query, query, query, [0, 8])
