# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.expert_layout import DirectExpertLayout, direct_expert_chunk
from relax.backends.megatron.weight_update.ipc import dispatch_typed_buckets


def receiver(**overrides):
    return (
        dict(
            tp_size=2,
            ep_size=1,
            pp_size=1,
            moe_dp_size=1,
            moe_runner_backend="triton",
            dtype="bfloat16",
            disable_shared_experts_fusion=True,
            num_experts=4,
            hidden_size=6,
            moe_intermediate_size=8,
        )
        | overrides
    )


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("prefix", ["model.layers", "model.language_model.layers"])
@pytest.mark.parametrize("geometry", [(4, 6, 8), (8, 10, 12)])
def test_direct_shards_preserve_expert_and_gate_up_order(tp, prefix, geometry):
    e, h, i = geometry
    layout = DirectExpertLayout(e, h, i, tp)
    gate = torch.arange(e * i * h).reshape(e, i, h).to(torch.bfloat16)
    up = gate + 512
    down = torch.arange(e * h * i).reshape(e, h, i).to(torch.bfloat16)
    packed = torch.cat((gate, up), dim=1)
    for rank in range(tp):
        [(name, actual)] = direct_expert_chunk([(f"{prefix}.9.mlp.experts.gate_up_proj", packed)], layout, rank)
        expected = torch.cat((gate.chunk(tp, dim=1)[rank], up.chunk(tp, dim=1)[rank]), dim=1)
        assert name == "model.layers.9.mlp.experts.w13_weight"
        assert torch.equal(actual, expected) and actual.is_contiguous()
        snapshot = actual.clone()
        layout.shard(packed + 4, "gate_up_proj", rank)
        assert torch.equal(actual, snapshot)
        [(name, actual)] = direct_expert_chunk([(f"{prefix}.9.mlp.experts.down_proj", down)], layout, rank)
        assert name == "model.layers.9.mlp.experts.w2_weight"
        assert torch.equal(actual, down.chunk(tp, dim=2)[rank]) and actual.is_contiguous()
    assert direct_expert_chunk([("model.norm.weight", gate)], layout, 0) is None
    assert direct_expert_chunk([], layout, 0) is None


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_direct_matches_installed_sglang_fused_moe_loader(tp):
    """Use the receiver's real per-expert loaders as the independent oracle."""
    layer_module = pytest.importorskip("sglang.srt.layers.moe.fused_moe_triton.layer")
    e, h, i = 4, 6, 8
    layout = DirectExpertLayout(e, h, i, tp)
    gate = torch.arange(e * i * h).reshape(e, i, h).to(torch.bfloat16)
    up, down = gate + 512, gate.transpose(1, 2).contiguous() + 1024
    receiver_layer = SimpleNamespace(
        moe_runner_config=SimpleNamespace(is_gated=True),
        quant_method=SimpleNamespace(),
        quant_config=None,
        use_padded_loading=False,
        use_presharded_weights=False,
        use_triton_kernels=False,
        moe_tp_size=tp,
    )
    for rank in range(tp):
        w13, w2 = (
            torch.empty(e, 2 * i // tp, h, dtype=torch.bfloat16),
            torch.empty(e, h, i // tp, dtype=torch.bfloat16),
        )
        for expert in range(e):
            layer_module.FusedMoE._load_w13(receiver_layer, w13[expert], 0, "w1", gate[expert], rank)
            layer_module.FusedMoE._load_w13(receiver_layer, w13[expert], 0, "w3", up[expert], rank)
            layer_module.FusedMoE._load_w2(receiver_layer, w2[expert], 1, "w2", down[expert], rank)
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
        dict(ep_size=2),
        dict(pp_size=2),
        dict(moe_dp_size=2),
        dict(moe_runner_backend="auto"),
        dict(moe_runner_backend="triton_kernel"),
        dict(moe_runner_backend="flashinfer_cutlass"),
        dict(dtype="float16"),
        dict(dtype="auto"),
        dict(disable_shared_experts_fusion=False),
        dict(enforce_shared_experts_fusion=True),
        dict(enable_eplb=True),
        dict(ep_num_redundant_experts=1),
        dict(init_expert_location="random"),
        dict(ep_join_mode="scale"),
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
