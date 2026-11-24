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
_MODULE_PATH = Path(__file__).parents[4] / "relax/models/gemma4/attention.py"
_SPEC = importlib.util.spec_from_file_location("_relax_gemma4_attention_test_target", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
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


def _packed_sdpa(query, key, value, boundaries):
    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        output = torch.nn.functional.scaled_dot_product_attention(
            query[start:end].transpose(0, 1).unsqueeze(0),
            key[start:end].transpose(0, 1).unsqueeze(0),
            value[start:end].transpose(0, 1).unsqueeze(0),
            scale=1.0,
            is_causal=True,
            enable_gqa=True,
        )
        outputs.append(output.squeeze(0).transpose(0, 1).flatten(1))
    return torch.cat(outputs)


def test_flex_packed_chunk_boundaries_preserve_cpu_outputs(monkeypatch):
    torch.manual_seed(32)
    monkeypatch.setattr(_MODULE, "_FLEX_QUERY_CHUNK_SIZE", 128)
    attention = _attention()
    lengths = [129, 145]
    boundaries = [0, 129, sum(lengths)]
    query = torch.randn(sum(lengths), 2, 512, dtype=torch.float32) * 0.1
    key = torch.randn(sum(lengths), 1, 512, dtype=torch.float32) * 0.1
    value = torch.randn_like(key)
    output = attention._thd_flex_per_subseq(query, key, value, boundaries)
    reference = _packed_sdpa(query, key, value, boundaries)
    atol, rtol = (3e-5, 1e-4)
    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
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


def _hf_vision_mask(groups, window_left):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_vision_model

    config = Gemma4TextConfig(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=window_left + 1,
        use_bidirectional_attention="vision",
    )
    config._attn_implementation = "eager"
    masks = create_masks_for_vision_model(
        config=config,
        inputs_embeds=torch.zeros(1, groups.numel(), 4, device=groups.device),
        attention_mask=None,
        past_key_values=None,
        position_ids=torch.arange(groups.numel(), device=groups.device)[None, :],
        block_sequence_ids=groups[None, :],
    )
    return {name: mask[0, 0] == 0 for name, mask in masks.items()}


@pytest.mark.parametrize("window_left", [31, 1023])
def test_vision_mask_matches_transformers_with_future_image_keys(window_left):
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(17)
    groups = torch.full((389,), -1, dtype=torch.long)
    groups[30:65] = 0
    groups[100:310] = 1
    offset, length = 128, 137
    query = torch.randn(1, 2, length, 4) * 0.1
    key = torch.randn(1, 1, groups.numel(), 4) * 0.1
    value = torch.randn_like(key)
    masks = _hf_vision_mask(groups, window_left)
    assert masks["sliding_attention"][128, 300]
    assert not masks["full_attention"][128, 300]
    assert bool(masks["sliding_attention"][264, 100]) is (window_left >= 164)
    output = flex_attention(
        query,
        key,
        value,
        scale=1.0,
        enable_gqa=True,
        block_mask=_MODULE._vision_block_mask(groups, length, offset, window_left),
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=1.0,
        enable_gqa=True,
        attn_mask=masks["sliding_attention"][offset : offset + length],
    )
    torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)


def test_vision_flex_preserves_packed_isolation_across_query_chunks(monkeypatch):
    monkeypatch.setattr(_MODULE, "_FLEX_QUERY_CHUNK_SIZE", 128)
    attention = _attention()
    attention.window_size = (31, 0)
    torch.manual_seed(19)
    query = torch.randn(260, 2, 4) * 0.1
    key = torch.randn(260, 1, 4) * 0.1
    value = torch.randn_like(key)
    groups = torch.full((260,), -1, dtype=torch.long)
    groups[120:150] = 0
    groups[160:220] = 1
    boundaries = [0, 157, 260]
    output = attention._thd_vision_flex(query, key, value, boundaries, groups)
    references = []
    for start, end in zip(boundaries, boundaries[1:]):
        reference = torch.nn.functional.scaled_dot_product_attention(
            query[start:end].transpose(0, 1)[None],
            key[start:end].transpose(0, 1)[None],
            value[start:end].transpose(0, 1)[None],
            scale=1.0,
            enable_gqa=True,
            attn_mask=_hf_vision_mask(groups[start:end], 31)["sliding_attention"],
        )
        references.append(reference.squeeze(0).transpose(0, 1).flatten(1))
    torch.testing.assert_close(output, torch.cat(references), atol=2e-5, rtol=2e-5)
    changed = value.clone()
    changed[157:] += 100
    changed_output = attention._thd_vision_flex(query, key, changed, boundaries, groups)
    torch.testing.assert_close(output[:157], changed_output[:157])


def test_image_metadata_keeps_global_layers_causal(monkeypatch):
    monkeypatch.setenv("RELAX_GEMMA4_FORCE_FLEX_GLOBAL", "1")
    attention = Gemma4CoreAttention(_attention().config, layer_number=6)
    torch.manual_seed(20)
    query = torch.randn(8, 2, 512) * 0.1
    key = torch.randn(8, 1, 512) * 0.1
    value = torch.randn_like(key)
    packed = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 8]),
        cu_seqlens_q_cpu=[0, 8],
        max_seqlen_q=8,
        gemma4_vision_group_ids=torch.zeros(8, dtype=torch.long),
    )
    output = attention(query, key, value, packed_seq_params=packed)
    torch.testing.assert_close(output, _packed_sdpa(query, key, value, [0, 8]), atol=3e-5, rtol=1e-4)
