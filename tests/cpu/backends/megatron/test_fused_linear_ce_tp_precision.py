# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise production autograd with CPU tensors; mock CUDA epilogues and TP
transport."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def fused_ce(monkeypatch):
    # Loading the complete module avoids unrelated Megatron package startup.
    name = "relax.backends.megatron.fused_linear_ce"
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/fused_linear_ce.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", "0")
    monkeypatch.setattr(module, "_log_enabled", lambda *_: None)
    ready = []
    utils = ModuleType("megatron.core.utils")
    utils.ensure_params_ready = lambda parameters: ready.extend(parameters)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    ops = ModuleType("relax.backends.megatron.fused_linear_ce_ops")

    def forward(x, weight, **kwargs):
        return {"lse": torch.zeros(x.shape[0]), "D": x @ weight}

    def backward(x, weight, **kwargs):
        return {"D": torch.ones(x.shape[0], weight.shape[1], dtype=x.dtype)}

    ops.linear_ce_forward = forward
    ops.linear_ce_backward = backward
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    return module, ready


@pytest.mark.parametrize("size,policy", [(1, False), (1, True), (4, None), (4, False), (4, True)])
@pytest.mark.parametrize("hidden_dtype", [torch.bfloat16, torch.float32])
def test_tp_dgrad_precision_and_bridge_rounding(fused_ce, monkeypatch, size, policy, hidden_dtype):
    module, ready = fused_ce
    group = object()
    monkeypatch.setattr(module.dist, "get_world_size", lambda received: size if received is group else 0)
    monkeypatch.setattr(module.dist, "get_rank", lambda received: 0 if received is group else -1)
    reductions = []
    # A peer contribution whose FP32 sum has extra precision that must be
    # rounded at the native BF16 head boundary, even for FP32 bridge inputs.
    peer = torch.tensor([[1 / 256, 1 / 128]], dtype=torch.float32)

    def reduce(tensor, op, group):
        assert group is expected_group
        if tensor.shape == (1, 2):
            assert op == module.dist.ReduceOp.SUM
            reductions.append(tensor.dtype)
            tensor.copy_(tensor.float() + peer)

    expected_group = group
    monkeypatch.setattr(module.dist, "all_reduce", reduce)
    hidden = torch.ones(1, 2, dtype=hidden_dtype, requires_grad=True)
    weight = torch.nn.Parameter(torch.full((2, 2), 0.5, dtype=torch.bfloat16), requires_grad=False)
    kwargs = {} if policy is None else {"tp_dgrad_fp32": policy}
    loss, _ = module.linear_cross_entropy(hidden, weight, torch.tensor([0]), group, **kwargs)
    (dx,) = torch.autograd.grad(loss.sum(), hidden)
    expected = torch.ones_like(hidden)
    if size > 1:
        expected = (expected.float() + peer).bfloat16().to(hidden_dtype)
    torch.testing.assert_close(dx, expected, rtol=0, atol=0)
    assert dx.dtype == hidden_dtype
    assert reductions == ([] if size == 1 else [torch.float32 if policy else torch.bfloat16])
    assert len(ready) == 1 and ready[0] is weight


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_public_entrypoint_forwards_precision_request(fused_ce, monkeypatch, temperature):
    module, _ = fused_ce
    seen = []
    monkeypatch.setattr(module._LinearCE, "apply", lambda *args: seen.append(args) or (None, None))
    hidden = torch.ones(1, 2)
    weight = torch.ones(2, 2)
    labels = torch.tensor([0])
    assert module.SUPPORTS_FP32_TP_DGRAD
    module.linear_cross_entropy(hidden, weight, labels, None, temperature=temperature, tp_dgrad_fp32=True)
    assert seen[0][4] == 1.0 / temperature
    assert seen[0][-1] is True


@pytest.mark.parametrize("policy", [False, True])
def test_blackwell_backport_retains_original_call_contract(fused_ce, monkeypatch, policy):
    from types import SimpleNamespace

    module, _ = fused_ce
    monkeypatch.setenv("RELAX_SFT_FUSED_LINEAR_CE", "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    calls = []
    kernel = ModuleType("relax.backends.megatron.kernels.linear_cross_entropy")
    result = object()
    kernel.linear_cross_entropy = lambda *args, **kwargs: calls.append((args, kwargs)) or result
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    hidden = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    weight, labels, group = object(), object(), object()
    assert module.linear_cross_entropy(hidden, weight, labels, group, tp_dgrad_fp32=policy) is result
    assert calls == [((hidden, weight, labels, group), {"accumulate": False, "return_logits": False})]
