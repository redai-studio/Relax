# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA"
)


def test_partial_update_preserves_sibling_scale_and_storage(monkeypatch):
    from relax.backends.sglang.deepseekv41.routed_fp8 import FusedMoE, RoutedFp8MoE

    # Isolate the converted values delivered to the native TP writer at EP=TP=1.
    # W1 and W3 occupy different halves of both the fused weight and its scale.
    def native_write(self, param, source, name, shard, expert_id):
        target = param[expert_id]
        if shard != "w2":
            gate, up = target.chunk(2, dim=0)
            target = gate if shard == "w1" else up
        target.copy_(source)

    monkeypatch.setattr(FusedMoE, "_weight_loader_impl", native_write)
    layer = RoutedFp8MoE.__new__(RoutedFp8MoE)
    torch.nn.Module.__init__(layer)
    layer.hidden_size = layer.intermediate_size_per_partition = 128
    layer.moe_tp_size = 1
    layer.use_presharded_weights = False
    for name, rows in (("w13", 2), ("w2", 1)):
        weight = torch.zeros((1, rows * 128, 128), dtype=torch.float8_e4m3fn, device="cuda")
        scale = torch.zeros((1, rows, 1), dtype=torch.float32, device="cuda")
        layer.register_parameter(name + "_weight", torch.nn.Parameter(weight, requires_grad=False))
        layer.register_parameter(name + "_weight_scale_inv", torch.nn.Parameter(scale, requires_grad=False))
    pointers = {name: p.data_ptr() for name, p in layer.named_parameters()}

    def load(shard, value):
        param = layer.w2_weight if shard == "w2" else layer.w13_weight
        source = torch.full((128, 128), value, dtype=torch.bfloat16)
        layer._weight_loader_impl(param, source, "experts." + shard + ".weight", shard, 0)

    load("w3", 4)
    load("w2", 2)
    sibling = layer.w13_weight[0, 128:].view(torch.uint8).clone()
    down = layer.w2_weight.view(torch.uint8).clone()
    other_scales = (layer.w13_weight_scale_inv[0, 1].item(), layer.w2_weight_scale_inv.item())
    for value in (1, -1024):
        load("w1", value)
        gate = layer.w13_weight[0, :128].float() * layer.w13_weight_scale_inv[0, 0]
        assert (gate == value).all()
        assert torch.equal(layer.w13_weight[0, 128:].view(torch.uint8), sibling)
        assert torch.equal(layer.w2_weight.view(torch.uint8), down)
        assert (layer.w13_weight_scale_inv[0, 1].item(), layer.w2_weight_scale_inv.item()) == other_scales
        assert {name: p.data_ptr() for name, p in layer.named_parameters()} == pointers


def test_opt_in_routes_v41_only(monkeypatch):
    from sglang.srt.models import deepseek_v2

    from relax.backends.sglang import deepseekv41
    from relax.backends.sglang.deepseekv41.routed_fp8 import FusedMoE, RoutedFp8MoE

    class ModelMoE:
        def __init__(self, config):
            self.expert_type = deepseek_v2.get_moe_impl_class(None)

    monkeypatch.setattr(deepseek_v2, "DeepseekV2MoE", ModelMoE)
    monkeypatch.setattr(deepseek_v2, "get_moe_impl_class", lambda config: FusedMoE)
    config = SimpleNamespace(model_type="deepseek_v41")
    monkeypatch.delenv("RELAX_DSV41_ROUTED_FP8", raising=False)
    importlib.reload(deepseekv41)
    assert ModelMoE(config).expert_type is FusedMoE
    monkeypatch.setenv("RELAX_DSV41_ROUTED_FP8", "1")
    importlib.reload(deepseekv41)
    assert ModelMoE(config).expert_type is RoutedFp8MoE
    assert ModelMoE(SimpleNamespace(model_type="deepseek_v3")).expert_type is FusedMoE
    assert deepseek_v2.get_moe_impl_class(None) is FusedMoE
