# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions for legacy gated-residual parameter ownership and math."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from torch import nn


# Avoid importing the backend package, which installs GPU-specific patches.
_SPEC = importlib.util.spec_from_file_location(
    "fsdp_hyper_connection_patch_under_test",
    Path(__file__).resolve().parents[4] / "relax/backends/megatron/fsdp_hyper_connection_patch.py",
)
patch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patch)


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
def test_patch_preserves_outputs_and_all_gradients(legacy_mcore, kind, dtype, input_requires_grad):
    torch.manual_seed(42)
    cls = getattr(legacy_mcore, kind)
    reference, repaired = cls(dtype=dtype), cls(dtype=dtype)
    repaired.load_state_dict(reference.state_dict(), strict=True)
    assert patch.patch_gated_residual_norms_for_fsdp(repaired) == 1
    inputs = torch.randn(3, 2, 32, dtype=dtype, requires_grad=input_requires_grad)
    repaired_inputs = inputs.detach().clone().requires_grad_(input_requires_grad)

    def outputs_and_backward(model, x):
        outputs = model(x)
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
@pytest.mark.parametrize("fused_dispatch", [False, True])
def test_norm_hook_precedes_weight_read(legacy_mcore, kind, fused_dispatch, monkeypatch):
    """Reproduce the non-unit FSDP rule: hooks gather only direct
    parameters."""
    if fused_dispatch:
        spec = importlib.util.spec_from_file_location(
            "hc_fused_mix_for_fsdp_test",
            Path(__file__).resolve().parents[4] / "relax/models/qwen3_8/ops/hc_fused_mix.py",
        )
        fusion = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fusion)
        core = ModuleType("megatron.core")
        core.utils = ModuleType("megatron.core.utils")
        monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setenv("RELAX_QWEN38_HC_FUSED_MIX", "1")
        original_mix = legacy_mcore.GatedResidualHyperConnection._mix

        # Mirror the image-built static dispatch in the current MCore _mix.
        def mix(self, hidden_states):
            from relax.utils.env import Envs

            if Envs.RELAX_QWEN38_HC_FUSED_MIX and fusion.supported(self, hidden_states):
                return fusion.fused_mix(self, hidden_states)
            return original_mix(self, hidden_states)

        monkeypatch.setattr(legacy_mcore.GatedResidualHyperConnection, "_mix", mix)
    cls = getattr(legacy_mcore, kind)
    gathered = set()
    norm_impl = legacy_mcore.gated_residual_group_rmsnorm

    def require_gathered_weight(x, weight, n, eps):
        assert id(weight) in gathered, "norm parameter read before gather hook"
        return norm_impl(x, weight, n, eps)

    legacy_mcore.gated_residual_group_rmsnorm = require_gathered_weight

    def attach_gather_hooks(model):
        def gather(module, _inputs):
            gathered.update(id(p) for p in module.parameters(recurse=False))

        for module in model.modules():
            module.register_forward_pre_hook(gather)

    broken = cls()
    attach_gather_hooks(broken)
    with pytest.raises(AssertionError, match="read before gather"):
        broken(torch.randn(3, 2, 32))

    gathered.clear()
    repaired = cls()
    patch.patch_gated_residual_norms_for_fsdp(repaired)
    attach_gather_hooks(repaired)
    repaired(torch.randn(3, 2, 32))
    assert id(repaired.hc_norm.weight) in gathered


def test_patch_preserves_parameter_tree_and_is_idempotent(legacy_mcore):
    model = nn.ModuleDict(
        {
            "decoder": legacy_mcore.GatedResidualOutputMixer(dtype=torch.bfloat16),
            "mtp": nn.ModuleList([legacy_mcore.GatedResidualHyperConnection()]),
            "unrelated": nn.Linear(3, 3),
        }
    ).eval()
    model["decoder"].hc_norm.weight.requires_grad_(False)
    params = dict(model.named_parameters())
    state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    unrelated_forward = model["unrelated"].forward

    assert patch.patch_gated_residual_norms_for_fsdp(model) == 2
    assert patch.patch_gated_residual_norms_for_fsdp(model) == 0
    assert model["unrelated"].forward == unrelated_forward
    assert not model["decoder"].hc_norm.training
    assert not model["decoder"].hc_norm.weight.requires_grad
    assert model["decoder"].hc_norm.eps == model["decoder"].norm_eps
    assert list(model.state_dict()) == list(state)
    for name, param in model.named_parameters():
        assert param is params[name]
        assert torch.equal(param, state[name])
        if "hc_norm" in name:
            assert param.sequence_parallel
    model.load_state_dict(state, strict=True)


def test_patch_preserves_new_upstream_norm_and_custom_mix(legacy_mcore):
    model = legacy_mcore.GatedResidualOutputMixer()
    native_norm = nn.Identity()
    model.hc_norm = native_norm
    native_mix = model._mix
    assert patch.patch_gated_residual_norms_for_fsdp(model) == 0
    assert model.hc_norm is native_norm
    assert model._mix == native_mix

    class CustomMixer(legacy_mcore.GatedResidualOutputMixer):
        def _mix(self, hidden_states):
            return hidden_states, None

    custom = CustomMixer()
    assert patch.patch_gated_residual_norms_for_fsdp(custom) == 0
    inputs = torch.randn(3, 2, 32)
    assert custom(inputs) is inputs


def test_patch_is_noop_for_mcore_without_gated_residual(legacy_mcore, monkeypatch):
    monkeypatch.delattr(legacy_mcore, "GatedResidualHyperConnection")
    assert patch.patch_gated_residual_norms_for_fsdp(nn.Linear(3, 3)) == 0


def test_patch_only_suppresses_missing_hyper_connection(monkeypatch):
    def missing_module(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(patch, "import_module", missing_module)
    assert patch.patch_gated_residual_norms_for_fsdp(nn.Linear(3, 3)) == 0

    def missing_dependency(name):
        raise ModuleNotFoundError(name="required_dependency")

    monkeypatch.setattr(patch, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError):
        patch.patch_gated_residual_norms_for_fsdp(nn.Linear(3, 3))
