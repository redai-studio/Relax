# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CUDA forward/backward accuracy for Gemma4 packed FlexAttention."""

import pytest
import torch

from tests.cpu.models.gemma4.test_attention import _MODULE, _attention, _packed_sdpa


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlexAttention")


def _eager_packed_flex(query, key, value, boundaries):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def causal(_batch, _head, query_index, key_index):
        return query_index >= key_index

    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        output = flex_attention(
            query[start:end].transpose(0, 1).unsqueeze(0),
            key[start:end].transpose(0, 1).unsqueeze(0),
            value[start:end].transpose(0, 1).unsqueeze(0),
            block_mask=create_block_mask(causal, None, None, end - start, end - start, device=query.device),
            scale=1.0,
            enable_gqa=True,
        )
        outputs.append(output.squeeze(0).transpose(0, 1).flatten(1))
    return torch.cat(outputs)


def _assert_bf16_flex_gradients(output, inputs, upstream, boundaries):
    # Match PyTorch's FlexAttention accuracy tests: compare compiled and eager
    # BF16 errors against FP64. Math SDPA retains FP32 intermediates, whereas
    # Flex backward uses the rounded saved output to compute its softmax delta.
    # A direct elementwise comparison to SDPA is too strict near zero gradients.
    eager_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    gold_inputs = [tensor.detach().double().requires_grad_() for tensor in inputs]
    eager = _eager_packed_flex(*eager_inputs, boundaries)
    gold = _packed_sdpa(*gold_inputs, boundaries)
    actual_grads = torch.autograd.grad(output, inputs, upstream)
    eager_grads = torch.autograd.grad(eager, eager_inputs, upstream)
    gold_grads = torch.autograd.grad(gold, gold_inputs, upstream.double())
    for name, actual, reference, golden in zip(("query", "key", "value"), actual_grads, eager_grads, gold_grads):
        assert torch.isfinite(actual).all(), name
        assert torch.isfinite(reference).all(), f"eager {name}"
        assert torch.isfinite(golden).all(), f"FP64 {name}"
        # Check every packed sample separately so one sample cannot hide an
        # error in another. Upstream allows 2.1x the eager BF16 mean error.
        for start, end in zip(boundaries, boundaries[1:]):
            actual_error = (actual[start:end].double() - golden[start:end]).abs().mean()
            reference_error = (reference[start:end].double() - golden[start:end]).abs().mean()
            assert actual_error <= 2.1 * reference_error + 1e-7, (
                f"{name}[{start}:{end}]: compiled error {actual_error}, eager error {reference_error}"
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flex_packed_chunk_boundaries_preserve_outputs_and_gradients(monkeypatch, dtype):
    torch.manual_seed(32)
    monkeypatch.setattr(_MODULE, "_FLEX_QUERY_CHUNK_SIZE", 128)
    attention = _attention()
    lengths = [129, 145]
    boundaries = [0, 129, sum(lengths)]
    query = (torch.randn(sum(lengths), 2, 512, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    key = (torch.randn(sum(lengths), 1, 512, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    value = torch.randn_like(key, requires_grad=True)
    output = attention._thd_flex_per_subseq(query, key, value, boundaries)
    reference = _packed_sdpa(query, key, value, boundaries)
    atol, rtol = (3e-3, 2e-2) if dtype == torch.bfloat16 else (3e-5, 1e-4)
    torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
    upstream = torch.randn_like(output)
    if dtype == torch.bfloat16:
        _assert_bf16_flex_gradients(output, (query, key, value), upstream, boundaries)
    else:
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
