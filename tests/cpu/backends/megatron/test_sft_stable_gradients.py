# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Policy/ownership contracts on CPU, with MCore/CUDA boundaries isolated.

The native GPU GEMM/GAF and real collective acceptance run separately. These
tests exercise the public context, actual CPU autograd and checkpointing.
"""

import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from relax.backends.megatron import sft_precision as precision


def test_disabled_policy_does_not_touch_old_torch_cuda_api(monkeypatch):
    class NoAPI:
        def __getattr__(self, name):
            raise AssertionError(f"Disabled policy touched CUDA API: {name}")

    monkeypatch.delenv("RELAX_SFT_STABLE_GRADIENTS", raising=False)
    monkeypatch.setattr(torch.backends, "cuda", NoAPI())
    args = Namespace()
    assert precision.configure_sft_stable_gradients(args) is False
    assert args._relax_sft_stable_gradients_config == {}
    assert precision.fused_head_precision_kwargs(SimpleNamespace(), ModuleType("old"), None) == {}
    value = torch.ones(1)
    with precision.native_head_precision(SimpleNamespace(), value) as mapped:
        assert mapped is value
    precision.set_sft_stable_head(None, True)


def test_enabled_policy_records_effective_process_blas_configuration(monkeypatch):
    class Matmul:
        allow_bf16_reduced_precision_reduction_split_k = True
        _reduced = True

        @property
        def allow_bf16_reduced_precision_reduction(self):
            return self._reduced

        @allow_bf16_reduced_precision_reduction.setter
        def allow_bf16_reduced_precision_reduction(self, value):
            self._reduced, self.allow_bf16_reduced_precision_reduction_split_k = value

    calls = []

    def preferred(value=None):
        if value is not None:
            calls.append(value)
        return "Cublaslt"

    monkeypatch.setenv("RELAX_SFT_STABLE_GRADIENTS", "1")
    monkeypatch.setattr(torch.backends, "cuda", SimpleNamespace(matmul=Matmul(), preferred_blas_library=preferred))
    args = Namespace(loss_type="sft", sft_chunked_logits=True, bf16=True)
    assert precision.configure_sft_stable_gradients(args)
    assert calls == ["cublaslt"]
    assert args._relax_sft_stable_gradients_config == {
        "preferred_blas_library": "Cublaslt",
        "allow_bf16_reduced_precision_reduction": False,
        "allow_bf16_reduced_precision_reduction_split_k": False,
    }


@pytest.fixture
def heads(monkeypatch):
    class Group:
        def size(self):
            return 4

    class Column(torch.nn.Module):
        def __init__(self, frozen=False):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(7, 5, dtype=torch.bfloat16), requires_grad=not frozen)
            self.bias = None
            self.gather_output = False
            self.sequence_parallel = False
            self.disable_grad_reduce = False
            self.allreduce_dgrad = True
            self.gradient_accumulation_fusion = False
            self.tp_group = Group()
            self.config = SimpleNamespace(defer_embedding_wgrad_compute=False)

        def forward(self, hidden):
            assert self.disable_grad_reduce and not self.allreduce_dgrad
            return F.linear(hidden, self.weight)

    class Native(Column):
        pass

    calls, ready = [], []

    class Copy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, group):
            ctx.group = group
            return hidden

        @staticmethod
        def backward(ctx, grad):
            calls.append((grad.dtype, ctx.group))
            return grad * ctx.group.size(), None

    for name in ("megatron", "megatron.core", "megatron.core.tensor_parallel", "megatron.core.transformer"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    modules = {
        "megatron.core.tensor_parallel.layers": dict(ColumnParallelLinear=Column),
        "megatron.core.transformer.linear_cross_entropy": dict(LinearCrossEntropyModule=Native),
        "megatron.core.tensor_parallel.mappings": dict(
            copy_to_tensor_model_parallel_region=lambda hidden, group: Copy.apply(hidden, group)
        ),
        "megatron.core.utils": dict(ensure_params_ready=lambda params: ready.extend(params)),
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    # External CUDA metadata boundary only: tensor math/autograd stay on CPU.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return SimpleNamespace(Column=Column, Native=Native, calls=calls, ready=ready)


@pytest.mark.parametrize("frozen", [False, True])
def test_native_precision_keeps_parameter_and_one_fp32_tp_sum_with_checkpoint(heads, frozen):
    layer = heads.Native(frozen)
    precision.set_sft_stable_head(layer, True)
    original = layer.weight
    hidden = torch.randn(3, 1, 5, dtype=torch.bfloat16, requires_grad=True)

    def forward(value):
        with precision.native_head_precision(layer, value) as mapped:
            return layer(mapped)

    output = checkpoint(forward, hidden, use_reentrant=True)
    assert not layer.disable_grad_reduce and layer.allreduce_dgrad
    output.float().sum().backward()
    assert layer.weight is original
    assert (layer.weight.grad is None) == frozen
    assert heads.calls == [(torch.float32, layer.tp_group)]
    assert heads.ready == [original, original]
    expected = (torch.ones(3, 7, dtype=torch.bfloat16) @ original.detach()).mul(4).reshape(hidden.shape)
    torch.testing.assert_close(hidden.grad, expected, atol=0, rtol=0)


def test_native_precision_restores_attributes_after_forward_failure(heads):
    layer = heads.Column()
    precision.set_sft_stable_head(layer, True)
    with pytest.raises(RuntimeError, match="forward failed"):
        with precision.native_head_precision(layer, torch.ones(2, 1, 5, dtype=torch.bfloat16)):
            raise RuntimeError("forward failed")
    assert not layer.disable_grad_reduce and layer.allreduce_dgrad


@pytest.mark.parametrize(
    "guard", ["hook", "expert", "gather", "defer", "buffer", "subclass", "main_grad", "labels_ce"]
)
def test_stable_head_rejects_unsupported_contracts(heads, guard):
    layer = heads.Column()
    if guard == "hook":
        layer.register_forward_pre_hook(lambda _module, args: args)
    elif guard == "expert":
        layer.is_expert = True
    elif guard == "gather":
        layer.gather_output = True
    elif guard == "defer":
        layer.config.defer_embedding_wgrad_compute = True
    elif guard == "buffer":
        layer.grad_output_buffer = []
    elif guard == "labels_ce":
        layer.config.cross_entropy_loss_fusion = True
        layer.config.cross_entropy_fusion_impl = "linear"
    elif guard == "subclass":

        class Adapter(heads.Column):
            pass

        layer = Adapter()
    else:
        layer.gradient_accumulation_fusion = True
        layer.weight.main_grad = torch.zeros_like(layer.weight)
    with pytest.raises(ValueError, match="SFT stable gradients"):
        precision.set_sft_stable_head(layer, True)


def test_stable_precision_rejects_external_weight_sp_and_graph_capture(heads, monkeypatch):
    layer = heads.Column()
    precision.set_sft_stable_head(layer, True)
    hidden = torch.ones(2, 1, 5, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="external"):
        with precision.native_head_precision(layer, hidden, layer.weight.clone()):
            pass
    layer.sequence_parallel = True
    with pytest.raises(ValueError, match="SP gather"):
        with precision.native_head_precision(layer, hidden):
            pass
    layer.sequence_parallel = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(ValueError, match="CUDA graphs disabled"):
        with precision.native_head_precision(layer, hidden):
            pass


def test_fused_precision_uses_capability_without_changing_legacy_kwargs(heads):
    layer = heads.Column()
    old = ModuleType("old_fused")
    assert precision.fused_head_precision_kwargs(layer, old, layer.weight) == {}
    precision.set_sft_stable_head(layer, True)
    assert precision.fused_head_precision_kwargs(layer, old, layer.weight) is None
    current = ModuleType("current_fused")
    current.SUPPORTS_FP32_TP_DGRAD = True
    assert precision.fused_head_precision_kwargs(layer, current, layer.weight) == {"tp_dgrad_fp32": True}
