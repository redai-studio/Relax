# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the actual GLM dispatcher, fused expert copy and all-layer MLA
refresh on CPU."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from relax.backends.sglang.delta_capability import requires_full_input, sparse_loader_identity
from relax.backends.sglang.delta_loader import DeltaReceiver, load_delta_weights
from relax.distributed.weight_sync.codec import encode
from relax.utils.sparse_tensor import FullTensorRequired


def make_model():
    pytest.importorskip("sglang.srt.models.glm4_moe_lite")
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
    from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM

    model = Glm4MoeLiteForCausalLM.__new__(Glm4MoeLiteForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        architectures=["Glm4MoeLiteForCausalLM"],
        n_routed_experts=4,
        first_k_dense_replace=1,
        num_hidden_layers=3,
        moe_layer_freq=1,
        num_nextn_predict_layers=1,
        q_lora_rank=None,
    )
    model.quant_config = None
    model.num_fused_shared_experts = 0
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module(), torch.nn.Module()])
    model.model.start_layer, model.model.end_layer = 1, 3
    for layer_id in (1, 2):
        layer = model.model.layers[layer_id]
        attn = layer.self_attn = torch.nn.Module()
        attn.kv_b_proj = torch.nn.Linear(3, 8, bias=False, dtype=torch.bfloat16)
        attn.qk_nope_head_dim = attn.v_head_dim = 2
        attn.w_kc, attn.w_vc = None, None
        layer.mlp = torch.nn.Module()
        expert = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(expert)
        layer.mlp.experts = expert
        expert.layer_id = layer_id
        expert.moe_ep_rank, expert.moe_ep_size = 0, 2
        expert._expert_storage_rank = expert.moe_tp_rank = 0
        expert.moe_tp_size = 1
        expert._num_local_routed = expert.num_local_experts = 2
        expert._num_global_routed = 4
        expert._has_fused_shared = expert.use_triton_kernels = expert.use_flashinfer_trtllm_moe = False
        expert.use_presharded_weights = False
        expert.num_fused_shared_experts = 0
        expert.quant_config = None
        expert.quant_method = UnquantizedFusedMoEMethod()
        expert.moe_runner_config = SimpleNamespace(is_gated=True)
        expert.w13_weight = torch.nn.Parameter(torch.zeros(2, 8, 3, dtype=torch.bfloat16), requires_grad=False)
        expert.w2_weight = torch.nn.Parameter(torch.zeros(2, 3, 4, dtype=torch.bfloat16), requires_grad=False)
        for p in (expert.w13_weight, expert.w2_weight):
            p.weight_loader = expert.weight_loader
    return model


def emulate_static_cuda_policy(monkeypatch):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    policy = SimpleNamespace(
        enable_eplb=False,
        init_expert_location="trivial",
        ep_join_mode=None,
        is_ep_scale_joiner=False,
        moe_a2a_backend="none",
        ep_num_redundant_experts=0,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(
            get_exec=lambda: SimpleNamespace(moe=policy),
            get_parallel=lambda: SimpleNamespace(max_ep_size=None),
        ),
    )
    # Production admits CUDA expert Parameters only. Simulate solely the
    # backend probe: named_parameters(), physical storage and the loader all
    # continue to operate on the actual CPU Parameters.
    monkeypatch.setattr(FusedMoE, "parameters", lambda self: iter((SimpleNamespace(device=torch.device("cuda")),)))
    return policy


def test_stock_glm_dispatcher_only_admits_independent_routed_experts(monkeypatch):
    from relax.backends.sglang.delta_model_loaders import stock_rank_skips_input

    model = make_model()
    assert sparse_loader_identity(model) is None
    policy = emulate_static_cuda_policy(monkeypatch)
    assert sparse_loader_identity(model) is not None
    expert = "model.layers.1.mlp.experts.0.gate_proj.weight"
    assert not requires_full_input(model, expert)
    for name in (
        expert.replace("layers.1", "layers.3"),
        expert.replace(".0.gate", ".4.gate"),
        expert.replace("layers.1", "layers.0"),
        expert.replace("experts.0", "shared_experts"),
        expert.replace("experts.0.gate_proj.weight", "experts.gate_up_proj"),
        "model.layers.2.self_attn.kv_b_proj.weight",
    ):
        assert requires_full_input(model, name)
    model.config.first_k_dense_replace = 0
    assert stock_rank_skips_input(model, expert.replace("layers.1", "layers.0"))
    assert not stock_rank_skips_input(model, expert)
    assert not stock_rank_skips_input(model, expert.replace("layers.1", "layers.3"))
    model.num_fused_shared_experts = 1
    assert sparse_loader_identity(model) is None
    model.num_fused_shared_experts = 0
    policy.ep_num_redundant_experts = 1
    assert sparse_loader_identity(model) is None
    policy.ep_num_redundant_experts = 0
    policy.enable_eplb = True
    assert sparse_loader_identity(model) is None


@torch.no_grad()
def test_glm_full_delta_anchor_keeps_native_group_hooks_and_unobserved_experts_full(monkeypatch):
    actual, reference = make_model(), make_model()
    reference.load_state_dict(actual.state_dict())
    policy = emulate_static_cuda_policy(monkeypatch)
    from sglang.srt.models.deepseek_common import deepseek_weight_loader

    refresh = []
    bind = deepseek_weight_loader.bind_or_assign

    def record(target, source):
        refresh.append(source.shape)
        return bind(target, source)

    monkeypatch.setattr(deepseek_weight_loader, "bind_or_assign", record)

    receiver = DeltaReceiver(actual)
    prefix = "model.layers.1.mlp.experts."
    names = [
        prefix + "0.gate_proj.weight",
        prefix + "1.up_proj.weight",
        "model.layers.2.self_attn.kv_b_proj.weight",
        prefix + "3.down_proj.weight",
    ]
    weights = [
        torch.arange(n, dtype=torch.int16).view(torch.bfloat16).reshape(shape)
        for n, shape in [(12, (4, 3)), (12, (4, 3)), (24, (8, 3)), (12, (3, 4))]
    ]
    for version, seed, base in (("0", True, None), ("1", False, "0"), ("3", True, "1")):
        control = dict(
            version=version, base=base, seed=seed, receiver_gpu_budget_bytes=65536, staging_budget_bytes=4096
        )
        if not seed:
            policy.enable_eplb = True
            with pytest.raises(ValueError, match="acknowledged loader geometry"):
                receiver.begin(control)
            assert receiver.version == "0" and receiver.active is None
            policy.enable_eplb = False
        receiver.begin(control)
        data = []
        for i, (name, weight) in enumerate(zip(names, weights, strict=True)):
            weight.view(torch.int16).view(-1)[1] = 0x7FC1 if version == "1" else (3 + i)
            sparse = not seed and i < 2
            positions = torch.tensor([1], dtype=torch.int32) if sparse else None
            values = weight.reshape(-1)[1:2] if sparse else weight.reshape(-1)
            data.append((name, weight.shape, weight.dtype, positions, values))
        meta, raw = encode(data, max_bytes=4096)
        # The first group contains only direct expert updates in D1.
        meta["loader_ends"] = [2, 4]
        status = torch.empty(4, dtype=torch.uint8)
        refresh.clear()
        receiver.consume(dict(frame=0, version=version, metadata=meta), raw, status)
        assert len(refresh) == 2 * 2 * 2  # 2 native invocations x 2 local layers x K/V
        assert status.tolist() == [0, 0, 1, 1]
        receiver.finalize(dict(version=version, frames=1), torch.full((1,), 255, dtype=torch.uint8))
        assert not receiver.force_full_seed and receiver.full_only == set(names[2:])
        if not seed:
            assert receiver.target_shard_metrics["entries"] == 2
        for lo, hi in ((0, 2), (2, 4)):
            reference.load_weights(zip(names[lo:hi], weights[lo:hi], strict=True))
        for (n, a), (m, b) in zip(actual.named_parameters(), reference.named_parameters(), strict=True):
            assert n == m and torch.equal(a.view(torch.int16), b.view(torch.int16))
        for i in (1, 2):
            for attr in ("w_kc", "w_vc"):
                a, b = (getattr(model.model.layers[i].self_attn, attr) for model in (actual, reference))
                assert torch.equal(a.view(torch.int16), b.view(torch.int16))

    receiver.begin(dict(version="4", base="3", seed=False, receiver_gpu_budget_bytes=65536, staging_budget_bytes=4096))
    # Losing an acknowledged direct binding must reject the frame before a native
    # SparseTensor fallback can touch the live expert or its caches.
    receiver._broadcast_plan = None
    meta, raw = encode(
        [data[0][:3] + (torch.tensor([0], dtype=torch.int32), weights[0].reshape(-1)[:1])], max_bytes=4096
    )
    actual._relax_delta_receiver = receiver
    refresh.clear()
    with pytest.raises(FullTensorRequired, match="acknowledged copy region"):
        load_delta_weights(
            actual,
            [
                (json.dumps(dict(op="consume", frame=0, version="4", metadata=meta)), raw),
                ("__delta_capability__", torch.empty(1, dtype=torch.uint8)),
            ],
        )
    assert receiver.version == "3" and receiver.failed and not refresh
