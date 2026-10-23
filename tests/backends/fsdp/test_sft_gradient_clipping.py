# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.fsdp.sft import clip_sharded_gradients


@pytest.mark.parametrize("max_norm", [0.0, 0.1, 100.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sft_foreach_clipping_matches_individual_scaling(monkeypatch, max_norm, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA clipping parity requires a GPU worker")
    generator = torch.Generator(device=device).manual_seed(42)
    grads = [
        torch.randn(17, generator=generator, dtype=dtype, device=device),
        torch.randn(5, 3, generator=generator, dtype=dtype, device=device).t(),
        torch.empty(0, dtype=dtype, device=device),
    ]
    expected = [g.clone() for g in grads]
    squares = torch.zeros((), dtype=torch.float32, device=device)
    for grad in expected:
        squares.add_(grad.float().square().sum())
    norm = squares.sqrt()
    if max_norm > 0:
        scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
        for grad in expected:
            grad.mul_(scale)
    group = object()
    calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group: calls.append(group))
    parameters = [
        SimpleNamespace(device=grad.device, grad=SimpleNamespace(to_local=lambda grad=grad: grad)) for grad in grads
    ]
    parameters.append(SimpleNamespace(device=torch.device(device), grad=None))
    result = clip_sharded_gradients(parameters, max_norm, group)
    assert calls == [group]
    assert torch.equal(result, norm)
    assert all(torch.equal(actual, reference) for actual, reference in zip(grads, expected))


def test_sft_clipping_without_gradients_keeps_collective_and_noop(monkeypatch):
    group = object()
    calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group: calls.append(group))
    monkeypatch.setattr(torch, "_foreach_mul_", lambda *args: pytest.fail("Empty gradients must not launch foreach"))
    result = clip_sharded_gradients([SimpleNamespace(device=torch.device("cpu"), grad=None)], 1.0, group)
    assert calls == [group]
    assert result == 0
