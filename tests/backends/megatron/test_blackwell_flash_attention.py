# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise TE's actual Blackwell attention selection and CUDA gradients."""

import importlib
import json

import pytest


@pytest.fixture
def attention_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("Requires a Blackwell GPU")
    attention = pytest.importorskip("transformer_engine.pytorch.attention")
    pytest.importorskip("flash_attn.cute")
    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    backend_module = importlib.import_module(
        "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention"
    )
    return torch, attention.DotProductAttention, backend_module


@pytest.mark.parametrize(
    "dimension,qheads,kvheads,lengths,causal,training,expected_major",
    [(72, 16, 16, [16, 127, 129], False, True, 2), (256, 16, 4, [17, 129], True, True, 4)],
    ids=["vision-packed", "language-gqa"],
)
def test_blackwell_attention_forward_backward(
    attention_runtime, record_property, dimension, qheads, kvheads, lengths, causal, training, expected_major
):
    torch, dot_product_attention, backend_module = attention_runtime
    torch.manual_seed(20261021)
    total = sum(lengths)
    q, k, v = [
        torch.randn(total, heads, dimension, device="cuda", dtype=torch.bfloat16, requires_grad=training)
        for heads in (qheads, kvheads, kvheads)
    ]
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    mask_type = "padding_causal" if causal else "padding"
    layer = dot_product_attention(
        num_attention_heads=qheads,
        kv_channels=dimension,
        num_gqa_groups=kvheads,
        qkv_format="thd",
        attn_mask_type=mask_type,
        attention_dropout=0.0,
    ).cuda()
    layer.train(training)
    with torch.set_grad_enabled(training):
        output = layer(
            q,
            k,
            v,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
            pad_between_seqs=False,
        ).reshape(total, qheads, dimension)
    backend = backend_module._attention_backends
    assert backend["use_flash_attention"] and not backend["use_fused_attention"]
    version = backend["flash_attention_backend"]
    assert version.major == expected_major, version
    qref, kref, vref = [value.detach().float().requires_grad_(training) for value in (q, k, v)]
    references = []
    for start, end in zip(offsets, offsets[1:]):
        qs = qref[start:end].transpose(0, 1)
        ks = kref[start:end].repeat_interleave(qheads // kvheads, dim=1).transpose(0, 1)
        vs = vref[start:end].repeat_interleave(qheads // kvheads, dim=1).transpose(0, 1)
        scores = (qs @ ks.transpose(-1, -2)) * dimension**-0.5
        if causal:
            mask = torch.ones(end - start, end - start, device="cuda", dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        references.append((scores.softmax(-1) @ vs).transpose(0, 1))
    reference = torch.cat(references)
    comparisons = [("output", output, reference)]
    if training:
        upstream = torch.randn_like(output)
        output.backward(upstream)
        reference.backward(upstream.float())
        comparisons.extend([("dq", q.grad, qref.grad), ("dk", k.grad, kref.grad), ("dv", v.grad, vref.grad)])
    torch.cuda.synchronize()
    errors = {}
    for key, actual, expected in comparisons:
        assert actual is not None and torch.isfinite(actual).all()
        atol, rtol = (0.005, 0.03) if key == "output" else (2 * torch.finfo(torch.bfloat16).eps, 0.05)
        torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)
        difference = actual.detach().float() - expected.detach()
        relative_l2 = difference.norm() / expected.detach().norm().clamp_min(1e-12)
        assert relative_l2 < 0.02, (key, relative_l2)
        errors[key] = {"max_abs": float(difference.abs().max()), "relative_l2": float(relative_l2)}
    record_property("flash_attention_version", str(version))
    record_property("max_abs_errors", json.dumps(errors))


@pytest.mark.parametrize("training", [True])
def test_qwen_context_parallel_keeps_native_fa2_selection(attention_runtime, record_property, training):
    _, _, _ = attention_runtime
    utils = importlib.import_module("transformer_engine.pytorch.attention.dot_product_attention.utils")
    params = utils.AttentionParams(
        qkv_layout="thd_thd_thd",
        num_heads=8,
        num_gqa_groups=1,
        head_dim_qk=256,
        head_dim_v=256,
        max_seqlen_q=32752,
        max_seqlen_kv=32752,
        attn_mask_type="padding_causal",
        window_size=(-1, 0),
        core_attention_bias_shape=None,
        core_attention_bias_requires_grad=False,
        context_parallel=True,
        cp_comm_type="p2p",
        cp_size=4,
        pad_between_seqs=False,
        is_training=training,
    )
    use_flash, version, use_fused, _, use_unfused, _ = utils.get_attention_backend(params)
    assert use_flash and version.major == 2 and not use_fused and not use_unfused, version
    record_property("flash_attention_version", str(version))
