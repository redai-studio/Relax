# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import inspect

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_qwen38_decode_matches_materialized_bf16_qk(monkeypatch: pytest.MonkeyPatch) -> None:
    l2norm = pytest.importorskip("fla.modules.l2norm").l2norm
    module = pytest.importorskip("sglang.kernels.ops.attention.fla.fused_recurrent")
    decode = module.fused_recurrent_gated_delta_rule_packed_decode
    if "RELAX_GDN_MATCH_TRAINING_QK_NORM" not in inspect.getsource(decode):
        pytest.skip("Requires the Qwen3.8 SGLang numerical parity patch")

    batch, heads, value_heads, key_dim, value_dim = 8, 2, 4, 128, 128
    torch.manual_seed(123)
    q = torch.randn(batch, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, value_heads, value_dim, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(batch, value_heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(value_heads, device="cuda")
    bias = torch.randn_like(a_log)
    initial = torch.randn(batch, value_heads, value_dim, key_dim, device="cuda") * 0.1
    packed = torch.cat([q.flatten(1), k.flatten(1), v.flatten(1)], dim=-1)

    # Megatron materializes FLA-normalized Q/K in BF16 before the recurrence.
    query = l2norm(q).float().repeat_interleave(value_heads // heads, dim=1) * key_dim**-0.5
    key = l2norm(k).float().repeat_interleave(value_heads // heads, dim=1)
    decay = -a_log.exp() * F.softplus(a.float() + bias)
    beta = b.float().sigmoid().to(b.dtype).float()
    state = initial * decay.exp()[:, :, None, None]
    residual = (v.float() - (state * key[:, :, None, :]).sum(-1)) * beta[:, :, None]
    state = state + residual[:, :, :, None] * key[:, :, None, :]
    expected = (state * query[:, :, None, :]).sum(-1).to(v.dtype)

    errors = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("RELAX_GDN_MATCH_TRAINING_QK_NORM", enabled)
        output = torch.empty(batch, 1, value_heads, value_dim, device="cuda", dtype=torch.bfloat16)
        decode(
            packed,
            a,
            b,
            a_log,
            bias,
            key_dim**-0.5,
            initial.clone(),
            output,
            torch.arange(batch, device="cuda"),
            use_qk_l2norm_in_kernel=True,
        )
        errors.append((output[:, 0].float() - expected.float()).abs().mean().item())
    assert errors[1] < 1e-7
    assert errors[1] < errors[0] * 0.01
