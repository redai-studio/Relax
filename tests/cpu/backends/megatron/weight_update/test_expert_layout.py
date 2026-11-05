# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.expert_layout import DirectExpertLayout, direct_expert_chunk
from relax.backends.megatron.weight_update.ipc import dispatch_typed_buckets


# Receiver-rank order from SGLang's [MoE DP, EP, MoE TP] process groups.
# Each entry is (expert rank, expert tensor-parallel rank).
TOPOLOGIES = [
    pytest.param(1, 1, 1, [(0, 0)], id="single"),
    pytest.param(2, 1, 1, [(0, 0), (0, 1)], id="tp2"),
    pytest.param(2, 2, 1, [(0, 0), (1, 0)], id="ep2"),
    pytest.param(4, 4, 1, [(0, 0), (1, 0), (2, 0), (3, 0)], id="ep4"),
    pytest.param(4, 2, 1, [(0, 0), (0, 1), (1, 0), (1, 1)], id="ep2-tp2"),
    pytest.param(4, 2, 2, [(0, 0), (1, 0), (0, 0), (1, 0)], id="ep2-dp2"),
    pytest.param(4, 1, 2, [(0, 0), (0, 1), (0, 0), (0, 1)], id="tp2-dp2"),
    pytest.param(8, 2, 2, [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0), (1, 1)], id="ep2-tp2-dp2"),
]


def receiver(**overrides):
    return (
        dict(
            tp_size=2,
            ep_size=1,
            pp_size=1,
            moe_dp_size=1,
            moe_runner_backend="triton",
            moe_a2a_backend="none",
            is_cuda=True,
            dtype="bfloat16",
            disable_shared_experts_fusion=True,
            num_experts=4,
            hidden_size=6,
            moe_intermediate_size=8,
        )
        | overrides
    )


@pytest.mark.parametrize("tp,ep,dp,ranks", TOPOLOGIES)
@pytest.mark.parametrize("backend", ["triton", "deep_gemm", "flashinfer_cutlass"])
@pytest.mark.parametrize("prefix", ["model.layers", "model.language_model.layers"])
@pytest.mark.parametrize("geometry", [(4, 6, 8), (8, 10, 12)])
def test_direct_shards_preserve_expert_and_gate_up_order(tp, ep, dp, ranks, backend, prefix, geometry):
    e, h, i = geometry
    layout = DirectExpertLayout.from_receiver(
        receiver(
            tp_size=tp,
            ep_size=ep,
            moe_dp_size=dp,
            num_experts=e,
            hidden_size=h,
            moe_intermediate_size=i,
            moe_runner_backend=backend,
        )
    )
    gate = torch.arange(e * i * h).reshape(e, i, h).to(torch.bfloat16)
    up = gate + 512
    down = torch.arange(e * h * i).reshape(e, h, i).to(torch.bfloat16)
    packed = torch.cat((gate, up), dim=1)
    for rank, (expert_rank, expert_tp_rank) in enumerate(ranks):
        [(name, actual)] = direct_expert_chunk([(f"{prefix}.9.mlp.experts.gate_up_proj", packed)], layout, rank)
        halves = tuple(t.chunk(ep)[expert_rank].chunk(tp // ep // dp, dim=1)[expert_tp_rank] for t in (gate, up))
        expected = torch.cat(halves[::-1] if backend == "flashinfer_cutlass" else halves, dim=1)
        assert name == "model.layers.9.mlp.experts.w13_weight"
        assert torch.equal(actual, expected) and actual.is_contiguous()
        snapshot = actual.clone()
        layout.shard(packed + 4, "gate_up_proj", rank)
        assert torch.equal(actual, snapshot)
        [(name, actual)] = direct_expert_chunk([(f"{prefix}.9.mlp.experts.down_proj", down)], layout, rank)
        assert name == "model.layers.9.mlp.experts.w2_weight"
        expected = down.chunk(ep)[expert_rank].chunk(tp // ep // dp, dim=2)[expert_tp_rank]
        assert torch.equal(actual, expected) and actual.is_contiguous()
    assert direct_expert_chunk([("model.norm.weight", gate)], layout, 0) is None
    assert direct_expert_chunk([], layout, 0) is None


@pytest.mark.parametrize("tp,ep,dp,ranks", TOPOLOGIES)
@pytest.mark.parametrize("backend", ["triton", "deep_gemm", "flashinfer_cutlass"])
def test_direct_matches_installed_sglang_fused_moe_loader(monkeypatch, tp, ep, dp, ranks, backend):
    """Use the receiver's real per-expert loaders as the independent oracle."""
    layer_module = pytest.importorskip("sglang.srt.layers.moe.fused_moe_triton.layer", exc_type=ImportError)
    unquant = pytest.importorskip("sglang.srt.layers.quantization.unquant", exc_type=ImportError)
    moe = pytest.importorskip("sglang.srt.layers.moe.utils", exc_type=ImportError)
    monkeypatch.setattr(unquant, "get_moe_runner_backend", lambda: moe.MoeRunnerBackend(backend))
    quant_method = unquant.UnquantizedFusedMoEMethod(use_deep_gemm=backend == "deep_gemm")
    e, h, i = 4, 6, 8
    layout = DirectExpertLayout.from_receiver(
        receiver(tp_size=tp, ep_size=ep, moe_dp_size=dp, moe_runner_backend=backend)
    )
    moe_tp = tp // ep // dp
    gate = torch.arange(e * i * h).reshape(e, i, h).to(torch.bfloat16)
    up, down = gate + 512, gate.transpose(1, 2).contiguous() + 1024
    receiver_layer = SimpleNamespace(
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_method=quant_method,
        quant_config=None,
        use_padded_loading=False,
        use_presharded_weights=False,
        use_triton_kernels=False,
        moe_tp_size=moe_tp,
        _num_local_routed=e // ep,
        _has_fused_shared=False,
    )
    for rank, (expert_rank, expert_tp_rank) in enumerate(ranks):
        receiver_layer._expert_storage_rank = expert_rank
        w13, w2 = (
            torch.empty(e // ep, 2 * i // moe_tp, h, dtype=torch.bfloat16),
            torch.empty(e // ep, h, i // moe_tp, dtype=torch.bfloat16),
        )
        for expert in range(e):
            local_expert = layer_module.FusedMoE._map_global_expert_id_to_local_expert_id(receiver_layer, expert)
            if local_expert == -1:
                continue
            layer_module.FusedMoE._load_w13(receiver_layer, w13[local_expert], 0, "w1", gate[expert], expert_tp_rank)
            layer_module.FusedMoE._load_w13(receiver_layer, w13[local_expert], 0, "w3", up[expert], expert_tp_rank)
            layer_module.FusedMoE._load_w2(receiver_layer, w2[local_expert], 1, "w2", down[expert], expert_tp_rank)
        assert torch.equal(w13, layout.shard(torch.cat((gate, up), dim=1), "gate_up_proj", rank))
        assert torch.equal(w2, layout.shard(down, "down_proj", rank))


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_resolved_receiver_geometry_has_no_model_family_requirement(tp):
    assert DirectExpertLayout.from_receiver(receiver(tp_size=tp)) == DirectExpertLayout(4, 6, 8, tp)


@pytest.mark.parametrize(
    "override",
    [
        dict(tp_size=3),
        dict(tp_size=0),
        dict(ep_size=3),
        dict(ep_size=None),
        dict(ep_size=0),
        dict(tp_size=3, ep_size=3),
        dict(pp_size=2),
        dict(moe_dp_size=3),
        dict(moe_dp_size=None),
        dict(moe_dp_size=False),
        dict(moe_runner_backend="auto"),
        dict(moe_runner_backend="triton_kernel"),
        dict(moe_runner_backend="flashinfer_trtllm"),
        dict(moe_runner_backend="flashinfer_trtllm_routed"),
        dict(moe_runner_backend="flashinfer_mxfp4"),
        dict(dtype="float16"),
        dict(dtype="auto"),
        dict(disable_shared_experts_fusion=False),
        dict(enforce_shared_experts_fusion=True),
        dict(enable_eplb=True),
        dict(ep_num_redundant_experts=1),
        dict(init_expert_location="random"),
        dict(ep_join_mode="scale"),
        dict(elastic_ep_backend="mooncake"),
        dict(dwdp_size=2),
        dict(quantization="fp8"),
        dict(quantization_config={"quant_method": "fp8"}),
        dict(enable_lora=True),
        dict(kt_weight_path="/weights"),
        dict(speculative_algorithm="EAGLE"),
        dict(hidden_size=None),
        dict(json_model_override_args='{"hidden_size":4096}'),
    ],
)
def test_unsupported_receiver_layout_is_rejected(override):
    with pytest.raises(ValueError):
        DirectExpertLayout.from_receiver(receiver(**override))


@pytest.mark.parametrize("backend", ["triton", "deep_gemm", "flashinfer_cutlass"])
@pytest.mark.parametrize("override", [dict(is_cuda=False), dict(is_cuda=None)])
def test_direct_cuda_backends_reject_unknown_platform(backend, override):
    with pytest.raises(ValueError):
        DirectExpertLayout.from_receiver(receiver(moe_runner_backend=backend, **override))


@pytest.mark.parametrize("backend", ["triton", "deep_gemm", "flashinfer_cutlass"])
@pytest.mark.parametrize("a2a", ["none", "deepep", "flashinfer", "mooncake", "nixl", "mori", "pplx", "ascend_fuseep"])
def test_weight_layout_is_independent_of_activation_dispatcher(backend, a2a):
    # The running engine owns dispatcher/platform compatibility validation.
    # Equal resident weight layouts must use the same weight transfer plan.
    base = receiver(ep_size=2, moe_runner_backend=backend)
    assert DirectExpertLayout.from_receiver(base | {"moe_a2a_backend": a2a}) == DirectExpertLayout.from_receiver(base)


def test_unsharded_native_order_keeps_local_expert_storage():
    layout = DirectExpertLayout.from_receiver(receiver(tp_size=4, ep_size=2, moe_dp_size=2))
    tensor = torch.zeros(4, 16, 6, dtype=torch.bfloat16)
    local = layout.shard(tensor, "gate_up_proj", 1)
    assert layout.moe_tp_size == 1 and local.shape == (2, 16, 6)
    assert local.data_ptr() == tensor[2:].data_ptr()
    assert layout.shard(tensor, "gate_up_proj", 3).data_ptr() == local.data_ptr()


def test_receiver_layout_agreement_includes_gate_up_order():
    triton, deep_gemm, cutlass = [
        DirectExpertLayout.from_receiver(receiver(moe_runner_backend=backend))
        for backend in ("triton", "deep_gemm", "flashinfer_cutlass")
    ]
    assert triton == deep_gemm
    assert cutlass != triton
    assert triton != DirectExpertLayout.from_receiver(receiver(ep_size=2))
    assert triton != DirectExpertLayout.from_receiver(receiver(moe_dp_size=2))


@pytest.mark.parametrize(
    "packets,formats",
    [
        ([[("direct", "d0")], [("direct", "d1")]], ["direct"]),
        ([[("flattened_bucket", "f0")], [("flattened_bucket", "f1")]], ["flattened_bucket"]),
        ([[], [("direct", "d1")]], ["direct"]),
        ([[("direct", "d0")], []], ["direct"]),
        ([[("direct", "d0")], [("flattened_bucket", "f1")]], ["direct", "flattened_bucket"]),
        (
            [[("flattened_bucket", "f0"), ("flattened_bucket", "g0")], [], [("direct", "d2")], []],
            ["flattened_bucket", "direct", "flattened_bucket"],
        ),
        ([[], []], []),
    ],
)
def test_format_dispatch_handles_empty_and_uneven_ranks(packets, formats):
    calls, held = [], []
    serializer = SimpleNamespace(
        serialize=lambda value, output_str: "empty-direct" if isinstance(value, list) else "empty-flat"
    )

    def remote(**kwargs):
        calls.append(kwargs)
        return len(calls)

    refs = dispatch_typed_buckets(
        packets,
        engine=SimpleNamespace(update_weights_from_tensor=SimpleNamespace(remote=remote)),
        version=12,
        serializer=serializer,
        empty_flattened=lambda: dict(flattened_tensor=torch.empty(0, dtype=torch.uint8), metadata=[]),
        held=held,
    )
    assert len(refs) == len(formats) and [call["load_format"] for call in calls] == formats
    assert all(
        len(call["serialized_named_tensors"]) == len(packets) and call["weight_version"] == "12" for call in calls
    )


def test_dispatch_validates_all_formats_before_sending_and_retains_partial_refs():
    calls, held, errors = [], [], []

    def remote(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("second dispatch failed")
        return "first-reader"

    kwargs = dict(
        engine=SimpleNamespace(update_weights_from_tensor=SimpleNamespace(remote=remote)),
        version=None,
        serializer=SimpleNamespace(serialize=lambda value, output_str: "empty"),
        empty_flattened=dict,
        held=held,
        dispatch_errors=errors,
    )
    with pytest.raises(ValueError):
        dispatch_typed_buckets([[("direct", "d"), ("invalid", "bad")]], **kwargs)
    assert calls == []
    refs = dispatch_typed_buckets([[("direct", "d"), ("flattened_bucket", "f")]], **kwargs)
    assert refs == ["first-reader"] and errors == ["second dispatch failed"]
    assert all("weight_version" not in call for call in calls)


def test_partial_dispatch_retains_readers_if_empty_serialization_fails():
    calls, errors, held = [], [], []

    def serialize(value, output_str):
        if isinstance(value, dict):
            raise RuntimeError("empty serialization failed")
        return "empty-direct"

    refs = dispatch_typed_buckets(
        [[("direct", "d")], [("flattened_bucket", "f")]],
        engine=SimpleNamespace(
            update_weights_from_tensor=SimpleNamespace(remote=lambda **kw: calls.append(kw) or "reader")
        ),
        version=1,
        serializer=SimpleNamespace(serialize=serialize),
        empty_flattened=dict,
        held=held,
        dispatch_errors=errors,
    )
    assert refs == ["reader"] and errors == ["empty serialization failed"]
    assert len(calls) == 1 and len(held) == 2
