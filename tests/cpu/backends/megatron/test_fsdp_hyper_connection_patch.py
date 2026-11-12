# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions for legacy gated-residual parameter ownership and math."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.cpu.backends.megatron.test_ple_packed_cp_gradients import _added_source


@pytest.fixture
def native_mcore():
    """Execute the actual image-built gated-residual classes on CPU."""
    module = ModuleType("native_hyper_connection")

    class MegatronModule(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    module.__dict__.update(
        torch=torch,
        nn=nn,
        F=F,
        Tensor=torch.Tensor,
        MegatronModule=MegatronModule,
        nvtx_decorator=lambda **kwargs: lambda fn: fn,
    )
    source = _added_source("megatron/core/transformer/hyper_connection.py")
    exec(
        compile("from __future__ import annotations\n" + source, "docker hyper_connection patch", "exec"),
        module.__dict__,
    )
    return module


def make_native(module, kind, dtype=torch.float32):
    config = SimpleNamespace(
        hidden_size=8,
        num_residual_streams=4,
        mhc_gated_residual_rank=5,
        layernorm_epsilon=1e-6,
        params_dtype=dtype,
        sequence_parallel=True,
        perform_initialization=True,
        init_method_std=0.02,
    )
    kwargs = {"eps": 1e-5} if kind == "GatedResidualOutputMixer" else {}
    return getattr(module, kind)(config, **kwargs)


@pytest.fixture
def legacy_mcore(monkeypatch):
    """Small CPU reproducer of the legacy MCore gated-residual
    implementation."""
    upstream = ModuleType("megatron.core.transformer.hyper_connection")

    def group_rmsnorm(x, weight, n, eps):
        streams = x.float().unflatten(-1, (n, -1))
        normed = streams * torch.rsqrt(streams.pow(2).mean(-1, keepdim=True) + eps)
        return (normed.flatten(-2) * (1.0 + weight.float())).to(x.dtype)

    class GatedResidualHyperConnection(nn.Module):
        def __init__(self, dtype=torch.float32, use_combine=True):
            super().__init__()
            self.n, self.hidden_size, self.norm_eps = 4, 8, 1e-6
            width, rank = self.n * self.hidden_size, 5
            self.hc_norm = nn.Module()
            self.hc_norm.weight = nn.Parameter(torch.randn(width, dtype=dtype))
            self.input_mix_weight_down = nn.Linear(width, rank, bias=False, dtype=dtype)
            self.input_mix_weight_up = nn.Linear(rank, width, bias=False, dtype=dtype)
            self.block_inject_weight = nn.Linear(width, self.n, bias=False, dtype=dtype) if use_combine else None
            for param in self.parameters():
                param.sequence_parallel = True

        def _mix(self, hidden_states):
            # Frozen legacy arithmetic: the regression is the direct weight read.
            s, b, _ = hidden_states.shape
            n, C = self.n, self.hidden_size
            xn = upstream.gated_residual_group_rmsnorm(hidden_states, self.hc_norm.weight, n, self.norm_eps)
            down = F.silu(self.input_mix_weight_down(xn).float() / n)
            gate = torch.sigmoid(self.input_mix_weight_up(down.to(xn.dtype)).float())
            block_input = (gate.view(s, b, n, C) * xn.float().view(s, b, n, C)).mean(dim=2)
            block_input = block_input.to(hidden_states.dtype)
            if self.block_inject_weight is None:
                return block_input, None
            inject = 2.0 * torch.sigmoid(self.block_inject_weight(xn).float() / n)
            return block_input, inject

        def forward(self, hidden_states):
            block_input, inject = self._mix(hidden_states)
            return block_input, None, inject

    class GatedResidualOutputMixer(GatedResidualHyperConnection):
        def __init__(self, dtype=torch.float32):
            super().__init__(dtype=dtype, use_combine=False)
            self.norm_eps = 1e-5

        def forward(self, hidden_states):
            return self._mix(hidden_states)[0]

    upstream.gated_residual_group_rmsnorm = group_rmsnorm
    upstream.GatedResidualHyperConnection = GatedResidualHyperConnection
    upstream.GatedResidualOutputMixer = GatedResidualOutputMixer
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    return upstream


@pytest.mark.parametrize("kind", ["GatedResidualHyperConnection", "GatedResidualOutputMixer"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("input_requires_grad", [False, True])
def test_patch_preserves_outputs_and_all_gradients(legacy_mcore, native_mcore, kind, dtype, input_requires_grad):
    torch.manual_seed(42)
    cls = getattr(legacy_mcore, kind)
    reference = cls(dtype=dtype)
    repaired = make_native(native_mcore, kind, dtype)
    repaired.load_state_dict(reference.state_dict(), strict=True)
    inputs = torch.randn(3, 2, 32, dtype=dtype, requires_grad=input_requires_grad)
    repaired_inputs = inputs.detach().clone().requires_grad_(input_requires_grad)

    def outputs_and_backward(model, x):
        outputs = model._mix(x) if kind == "GatedResidualHyperConnection" else model(x)
        tensors = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(t for t in outputs if t is not None)
        sum(t.float().square().sum() for t in tensors).backward()
        return tensors

    expected = outputs_and_backward(reference, inputs)
    actual = outputs_and_backward(repaired, repaired_inputs)
    assert len(expected) == len(actual)
    assert actual[0].dtype == dtype
    if len(actual) > 1:
        assert actual[1].dtype == torch.float32  # Injection gates stay in FP32.
    for old, new in zip(expected, actual, strict=True):
        assert old.dtype == new.dtype
        assert torch.equal(old, new)
    if input_requires_grad:
        assert torch.equal(inputs.grad, repaired_inputs.grad)
    for name, param in reference.named_parameters():
        assert torch.equal(param.grad, repaired.get_parameter(name).grad), name


@pytest.mark.parametrize("kind", ["GatedResidualHyperConnection", "GatedResidualOutputMixer"])
def test_norm_hook_precedes_weight_read(legacy_mcore, native_mcore, kind):
    """FSDP's non-unit hooks gather only a module's direct parameters."""
    gathered = set()

    def instrument(module):
        norm_impl = module.gated_residual_group_rmsnorm

        def require_gathered_weight(x, weight, n, eps):
            assert id(weight) in gathered, "norm parameter read before gather hook"
            return norm_impl(x, weight, n, eps)

        module.gated_residual_group_rmsnorm = require_gathered_weight

    def attach_gather_hooks(model):
        def gather(module, _inputs):
            gathered.update(id(p) for p in module.parameters(recurse=False))

        for module in model.modules():
            module.register_forward_pre_hook(gather)

    instrument(legacy_mcore)
    instrument(native_mcore)
    broken = getattr(legacy_mcore, kind)()
    attach_gather_hooks(broken)
    with pytest.raises(AssertionError, match="read before gather"):
        broken(torch.randn(3, 2, 32))

    gathered.clear()
    repaired = make_native(native_mcore, kind)
    attach_gather_hooks(repaired)
    repaired(torch.randn(3, 2, 32))
    assert id(repaired.hc_norm.weight) in gathered


def test_patch_preserves_parameter_tree_and_output_norm_epsilon(legacy_mcore, native_mcore):
    reference = legacy_mcore.GatedResidualOutputMixer(dtype=torch.bfloat16)
    model = make_native(native_mcore, "GatedResidualOutputMixer", torch.bfloat16).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    assert model.hc_norm.eps == model.norm_eps == 1e-5
    assert list(model.state_dict()) == list(reference.state_dict())
    assert not model.hc_norm.training
    for name, param in model.named_parameters():
        assert torch.equal(param, reference.get_parameter(name))
        assert param.sequence_parallel


@pytest.mark.parametrize("hook", ["forward_pre", "forward", "full_backward_pre", "full_backward"])
def test_fused_mix_respects_norm_hooks(native_mcore, hook):
    spec = importlib.util.spec_from_file_location(
        "hc_fused_mix_for_fsdp_test",
        Path(__file__).resolve().parents[4] / "relax/models/qwen3_8/ops/hc_fused_mix.py",
    )
    fusion = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fusion)
    model = make_native(native_mcore, "GatedResidualOutputMixer", torch.bfloat16)
    getattr(model.hc_norm, f"register_{hook}_hook")(lambda *args: None)
    # Only device eligibility is simulated; rejection must occur before tensor math.
    inputs = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16, ndim=3)
    assert not fusion._supported(model, inputs, None)
