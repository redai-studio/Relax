# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
from types import SimpleNamespace

import pytest
import torch

from relax.backends.megatron.weight_update.expert_layout import DirectExpertLayout, expert_slice_chunk
from relax.backends.megatron.weight_update.ipc import dispatch_typed_buckets
from relax.backends.sglang.weight_loader import EXPERT_SLICE_LOADER, expert_slice_name, load_expert_slices
from tests.backends.megatron.weight_update.test_expert_layout import TOPOLOGIES


def model_with_experts(e, h, i):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].mlp = torch.nn.Module()
    layer = torch.nn.Module()
    model.model.layers[0].mlp.experts = layer
    layer.w13_weight = torch.nn.Parameter(torch.zeros(e, 2 * i, h, dtype=torch.bfloat16))
    layer.w2_weight = torch.nn.Parameter(torch.zeros(e, h, i, dtype=torch.bfloat16))
    return model, layer


@pytest.mark.parametrize("tp,ep,dp,ranks", TOPOLOGIES)
@pytest.mark.parametrize("up_first", [False, True])
@pytest.mark.parametrize("prefix", ["model.layers", "model.language_model.layers"])
def test_streamed_projection_matches_full_load_for_every_receiver(tp, ep, dp, ranks, up_first, prefix):
    e, h, i = 8, 10, 12
    layout = DirectExpertLayout(e, h, i, tp, up_first, ep, dp, supports_expert_slices=True)
    for rank, (expert_rank, expert_tp_rank) in enumerate(ranks):
        model, layer = model_with_experts(e // ep, h, i // layout.moe_tp_size)
        pointers = [p.data_ptr() for p in model.parameters()]
        previous = None
        for step in range(3):
            gate = (torch.arange(e * i * h).reshape(e, i, h) + step * 100).to(torch.bfloat16)
            up, down = gate + 2048, gate.transpose(1, 2).contiguous() + 4096
            packed = torch.cat((gate, up), dim=1)
            # Multiple separated owner ranges travel in one IPC request.
            for ranges in [[(0, 2), (4, 6)], [(2, 4), (6, 8)]]:
                for projection, full in [("gate_up_proj", packed), ("down_proj", down)]:
                    chunks = expert_slice_chunk(
                        [
                            (f"{prefix}.0.mlp.experts.{projection}[{start}:{end}]", full[start:end])
                            for start, end in ranges
                        ],
                        layout,
                        rank,
                    )
                    load_expert_slices(model, chunks)
            gate = gate.chunk(ep)[expert_rank].chunk(layout.moe_tp_size, dim=1)[expert_tp_rank]
            up = up.chunk(ep)[expert_rank].chunk(layout.moe_tp_size, dim=1)[expert_tp_rank]
            assert torch.equal(layer.w13_weight, torch.cat((up, gate) if up_first else (gate, up), dim=1))
            assert torch.equal(
                layer.w2_weight, down.chunk(ep)[expert_rank].chunk(layout.moe_tp_size, dim=2)[expert_tp_rank]
            )
            assert pointers == [p.data_ptr() for p in model.parameters()]
            if previous is not None:
                assert not torch.equal(previous, layer.w13_weight)
            previous = layer.w13_weight.clone()


@pytest.mark.parametrize(
    "mutation", ["offset", "fullshape", "trailing_shape", "dtype", "empty", "name", "protocol", "overlap"]
)
def test_loader_validates_all_slices_before_mutating_parameters(mutation):
    model, layer = model_with_experts(8, 10, 12)
    name = "model.layers.0.mlp.experts.w13_weight"
    good = (expert_slice_name(name, 0, layer.w13_weight.shape), torch.ones(2, 24, 10, dtype=torch.bfloat16))
    descriptor, tensor = expert_slice_name(name, 4, layer.w13_weight.shape), good[1]
    if mutation == "offset":
        descriptor = expert_slice_name(name, 7, layer.w13_weight.shape)
    elif mutation == "fullshape":
        descriptor = expert_slice_name(name, 4, (9, 24, 10))
    elif mutation == "trailing_shape":
        tensor = tensor.transpose(1, 2)
    elif mutation == "dtype":
        tensor = tensor.float()
    elif mutation == "empty":
        tensor = tensor[:0]
    elif mutation == "name":
        descriptor = expert_slice_name("model.layers.0.mlp.experts.other", 4, layer.w13_weight.shape)
    elif mutation == "protocol":
        descriptor = descriptor.replace(".v1:", ".v2:")
    else:
        descriptor = expert_slice_name(name, 1, layer.w13_weight.shape)
    with pytest.raises(ValueError):
        load_expert_slices(model, [good, (descriptor, tensor)])
    assert torch.count_nonzero(layer.w13_weight) == 0
    load_expert_slices(model, [])


def test_loader_resolves_replaced_parameters_after_offload():
    model, layer = model_with_experts(8, 10, 12)
    weights = [
        (expert_slice_name("model.layers.0.mlp.experts.w2_weight", 6, (8, 10, 12)), torch.ones(2, 10, 12).bfloat16())
    ]
    old = layer.w2_weight
    load_expert_slices(model, weights)
    layer.w2_weight = torch.nn.Parameter(torch.zeros_like(old))
    load_expert_slices(model, weights)
    assert torch.equal(old, layer.w2_weight) and old.data_ptr() != layer.w2_weight.data_ptr()


def test_slice_capability_is_required_before_export():
    layout = DirectExpertLayout(8, 10, 12, 2)
    with pytest.raises(ValueError, match="does not support"):
        expert_slice_chunk(
            [("model.layers.0.mlp.experts.down_proj[2:4]", torch.ones(2, 10, 12).bfloat16())], layout, 0
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_slice_and_hf_payload_is_rejected(reverse):
    layout = DirectExpertLayout(8, 10, 12, 2, supports_expert_slices=True)
    tensors = [
        ("model.layers.0.mlp.experts.down_proj[2:4]", torch.ones(2, 10, 12).bfloat16()),
        ("model.norm.weight", torch.ones(10)),
    ]
    with pytest.raises(ValueError, match="Mixed"):
        expert_slice_chunk(tensors[::-1] if reverse else tensors, layout, 0)


def test_slice_dispatch_uses_list_payload_for_empty_ranks():
    calls, held = [], []
    dispatch_typed_buckets(
        [[(EXPERT_SLICE_LOADER, "slice0")], [], [("direct", "native2")]],
        engine=SimpleNamespace(update_weights_from_tensor=SimpleNamespace(remote=lambda **kw: calls.append(kw))),
        version=3,
        serializer=SimpleNamespace(serialize=lambda value, output_str: json.dumps(value)),
        empty_flattened=lambda: pytest.fail("Native expert slices require empty lists"),
        held=held,
    )
    assert [call["load_format"] for call in calls] == [EXPERT_SLICE_LOADER, "direct"]
    assert calls[0]["serialized_named_tensors"] == ["slice0", "[]", "[]"]
    assert calls[1]["serialized_named_tensors"] == ["[]", "[]", "native2"]
    assert all(call["weight_version"] == "3" for call in calls)
