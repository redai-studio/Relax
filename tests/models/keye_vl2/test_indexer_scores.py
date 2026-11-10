# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Fixed-tile Keye indexer requires CUDA")


def _rounding_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query = torch.zeros(128, 16, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(4096, 1, 64, device="cuda", dtype=torch.bfloat16)
    weights = torch.zeros(128, 16, device="cuda", dtype=torch.bfloat16)
    query[120, 4] = torch.tensor(_ROUNDING_QUERY, device="cuda", dtype=torch.bfloat16)
    key[503, 0] = torch.tensor(_ROUNDING_KEY, device="cuda", dtype=torch.bfloat16)
    weights[120, 4] = 1
    return query, key, weights


def test_indexer_scores_midpoint_dot_is_independent_of_key_width() -> None:
    """A real BF16 midpoint case changed top-k after long-context re-
    prefill."""
    from relax.models.keye_vl2.indexer import indexer_scores

    query, key, weights = _rounding_inputs()
    dot = (query[120, 4].double() * key[503, 0].double()).sum().to(query.dtype)
    torch.testing.assert_close(dot, dot.new_tensor(36.5), atol=0, rtol=0)
    for width in (504, 2048, 2688, 2743, 4096):
        scores = indexer_scores(query, key[:width], weights)
        expected = torch.zeros_like(scores)
        expected[120, 503] = dot * 0.125
        torch.testing.assert_close(scores, expected, atol=0, rtol=0)


def test_indexer_weights_gradient_uses_the_actual_fixed_dot() -> None:
    from relax.models.keye_vl2.indexer import indexer_scores

    query, key, weights = _rounding_inputs()
    weights.requires_grad_()
    indexer_scores(query, key[:2743], weights)[120, 503].backward()
    expected = torch.zeros_like(weights)
    expected[120, 4] = 36.5 * 0.125
    torch.testing.assert_close(weights.grad, expected, atol=0, rtol=0)


@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize(
    "needs", [(True, True, True), (True, False, False), (False, True, False), (False, False, True)]
)
def test_indexer_vjp_matches_independent_fp64_dot_with_bf16_boundaries(seed: int, needs: tuple[bool, ...]) -> None:
    from relax.models.keye_vl2.indexer import indexer_scores

    torch.manual_seed(seed)
    inputs = [
        torch.randn(7, 16, 128, device="cuda", dtype=torch.bfloat16)[..., ::2],
        torch.randn(129, 1, 128, device="cuda", dtype=torch.bfloat16)[..., ::2],
        torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)[:, ::2],
    ]
    actual_inputs = [tensor.requires_grad_(need) for tensor, need in zip(inputs, needs, strict=True)]
    reference_inputs = [
        tensor.detach().clone().requires_grad_(need) for tensor, need in zip(inputs, needs, strict=True)
    ]
    actual = indexer_scores(*actual_inputs)
    q, k, w = reference_inputs
    # No production kernel/backward in the oracle. Keep all BF16 activation
    # boundaries, but form dot products and the head sum independently in FP64.
    dots = torch.einsum("qhd,kd->hqk", q.double(), k.squeeze(1).double()).to(q.dtype)
    weighted = (dots * 0.125).relu() * w.transpose(0, 1).unsqueeze(-1)
    expected = weighted.double().sum(0).to(q.dtype)
    upstream = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, [x for x in actual_inputs if x.requires_grad], upstream)
    reference_grads = torch.autograd.grad(expected, [x for x in reference_inputs if x.requires_grad], upstream)
    for value, reference in zip((actual, *actual_grads), (expected, *reference_grads), strict=True):
        error = (value.double() - reference.double()).norm() / reference.double().norm().clamp_min(1e-30)
        assert error < 0.006, f"relative L2 error {error.item()}"


def test_indexer_zero_relu_has_zero_gradients() -> None:
    from relax.models.keye_vl2.indexer import indexer_scores

    q = torch.zeros(3, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(73, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gradients = torch.autograd.grad(indexer_scores(q, k, w).sum(), (q, k, w))
    for gradient in gradients:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)


# Minimal input vectors from the observed 36.375 BF16 rounding boundary.
_ROUNDING_QUERY = [
    -0.12890625,
    -0.00238037109375,
    1.1328125,
    -0.8359375,
    -0.55078125,
    -0.412109375,
    0.455078125,
    -0.7265625,
    -0.341796875,
    -1.7265625,
    -1.5703125,
    1.296875,
    -1.765625,
    1.4609375,
    -1.34375,
    -1.8046875,
    0.0673828125,
    0.8046875,
    -2.078125,
    0.82421875,
    0.11328125,
    -0.296875,
    -1.1796875,
    1.0859375,
    1.390625,
    1.40625,
    1.7109375,
    -0.625,
    0.75,
    0.58984375,
    0.8828125,
    0.89453125,
    1.9453125,
    -0.64453125,
    0.66015625,
    -0.302734375,
    -0.00604248046875,
    0.439453125,
    -1.953125,
    -1.2265625,
    -0.396484375,
    -1.796875,
    -0.357421875,
    -1.21875,
    -0.271484375,
    -1.53125,
    1.4140625,
    0.421875,
    -2.609375,
    1.59375,
    -0.5390625,
    0.88671875,
    -0.0302734375,
    -0.439453125,
    0.73828125,
    -0.11328125,
    0.443359375,
    1.09375,
    -0.052490234375,
    -0.0693359375,
    -0.5625,
    0.765625,
    0.55078125,
    1.421875,
]
_ROUNDING_KEY = [
    1.15625,
    -0.36328125,
    1.5859375,
    0.09130859375,
    0.9453125,
    1.5234375,
    -0.7890625,
    0.70703125,
    0.51953125,
    -1.640625,
    -0.447265625,
    0.97265625,
    -1.3046875,
    0.25,
    0.09375,
    1.140625,
    0.83203125,
    1.59375,
    -1.59375,
    0.4140625,
    0.08447265625,
    0.57421875,
    1.9296875,
    -0.439453125,
    0.283203125,
    2.484375,
    0.99609375,
    0.2412109375,
    0.099609375,
    2.109375,
    0.38671875,
    0.546875,
    1.2265625,
    0.95703125,
    1.15625,
    -0.16015625,
    0.2890625,
    -0.76171875,
    -1.625,
    -1.8046875,
    -1.765625,
    -2.25,
    1.328125,
    -1.7421875,
    -0.0986328125,
    -0.40234375,
    1.3515625,
    -1.390625,
    -0.455078125,
    -0.2080078125,
    -1.8125,
    1.3515625,
    0.04052734375,
    -1.03125,
    1.15625,
    0.322265625,
    0.8359375,
    0.11328125,
    -0.8515625,
    -1.421875,
    -0.44921875,
    -0.345703125,
    0.640625,
    0.55078125,
]
