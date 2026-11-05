# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""KDA backward must match a recurrent FP64 oracle, including small gate
gradients.

Run with RELAX_TEST_K3_PARALLEL_GPU=1 on one idle GPU. The unpatched FLA 0.4.2
kernel loses gate accuracy in FP32 dots that default to TF32, even without TP.
"""

import os

import pytest
import torch
from torch import Tensor


pytestmark = pytest.mark.skipif(
    os.environ.get("RELAX_TEST_K3_PARALLEL_GPU") != "1" or not torch.cuda.is_available(),
    reason="Requires explicit RELAX_TEST_K3_PARALLEL_GPU=1 and one idle GPU with FLA.",
)


def _inputs(length: int) -> tuple[list[Tensor], Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(317)

    def normal(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, device="cuda")

    shape = (1, length, 4, 128)
    q, k = [torch.nn.functional.normalize(normal(*shape), dim=-1).bfloat16() for _ in range(2)]
    v = normal(*shape).bfloat16() * 0.02
    g = (normal(*shape) * 0.001).bfloat16()
    beta = torch.rand(1, length, 4, generator=generator, device="cuda").bfloat16()
    a_log, bias = normal(4) * 0.02, normal(4 * 128) * 0.02
    dout = normal(*shape).bfloat16() * 0.02
    return [q, k, v, g, beta, a_log, bias], dout


def _reference(inputs: list[Tensor]) -> Tensor:
    """Differentiate the defining recurrence, without FLA's chunk
    factorization."""
    q, k, v, g, beta, a_log, bias = inputs
    heads, width = q.shape[-2:]
    gate = -5.0 * torch.sigmoid(a_log.view(heads, 1).exp() * (g + bias.view(heads, width)))
    state = q.new_zeros(q.shape[0], heads, width, v.shape[-1])
    outputs = []
    for token in range(q.shape[1]):
        state = state * gate[:, token].exp().unsqueeze(-1)
        residual = v[:, token] - (k[:, token].unsqueeze(-1) * state).sum(-2)
        state = state + (beta[:, token].unsqueeze(-1) * k[:, token]).unsqueeze(-1) * residual.unsqueeze(-2)
        outputs.append((q[:, token].unsqueeze(-1) * state).sum(-2) * width**-0.5)
    return torch.stack(outputs, dim=1)


def _relative_close(actual: Tensor, expected: Tensor, name: str) -> None:
    error = (actual.double() - expected.double()).norm()
    norm = expected.double().norm()
    assert error <= 0.01 * norm + 1e-10, f"{name}: error={float(error):.6g}, norm={float(norm):.6g}"


def test_kimi_k3_kda_gate_gradients_match_recurrence_across_head_shards_and_padding() -> None:
    from fla.ops.kda import chunk_kda

    inputs, dout = _inputs(109)
    reference_inputs = [x.double().requires_grad_() for x in inputs]
    expected_output = _reference(reference_inputs)
    (expected_output * dout.double()).sum().backward()
    names = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
    for heads, padding in ((slice(None), 0), (slice(None), 19), (slice(0, 2), 0), (slice(2, 4), 0)):
        leaves = [x[:, :, heads].clone().requires_grad_() for x in inputs[:5]]
        leaves += [
            inputs[5][heads].clone().requires_grad_(),
            inputs[6].view(4, 128)[heads].flatten().clone().requires_grad_(),
        ]
        q, k, v, g, beta, a_log, bias = leaves
        if padding:
            q, k, v, g = [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, padding)) for x in (q, k, v, g)]
            beta = torch.nn.functional.pad(beta, (0, 0, 0, padding))
        output, _ = chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            A_log=a_log,
            dt_bias=bias,
            use_gate_in_kernel=True,
            safe_gate=True,
            lower_bound=-5.0,
            transpose_state_layout=True,
        )
        (output[:, :109].float() * dout[:, :, heads].float()).sum().backward()
        _relative_close(output[:, :109], expected_output[:, :, heads], "output")
        for name, leaf, reference in zip(names, leaves, reference_inputs, strict=True):
            expected = reference.grad
            if name == "A_log":
                expected = expected[heads]
            elif name == "dt_bias":
                expected = expected.view(4, 128)[heads].flatten()
            else:
                expected = expected[:, :, heads]
            _relative_close(leaf.grad, expected, f"{name}, heads={heads}, padding={padding}")


def test_kimi_k3_kda_first_token_has_no_gate_gradient_with_padding() -> None:
    from fla.ops.kda import chunk_kda

    inputs, dout = _inputs(1)
    q, k, v, g, beta, a_log, bias = [x.requires_grad_() for x in inputs]
    padded = [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 19)) for x in (q, k, v, g)]
    output, _ = chunk_kda(
        *padded,
        torch.nn.functional.pad(beta, (0, 0, 0, 19)),
        A_log=a_log,
        dt_bias=bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        transpose_state_layout=True,
    )
    (output[:, :1].float() * dout.float()).sum().backward()
    # With zero initial state, the first output cannot depend on its forget gate.
    for parameter in (g, a_log, bias):
        assert parameter.grad.double().norm() < 1e-10
