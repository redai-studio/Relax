# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_PATH = Path(__file__).parents[4] / "relax/backends/megatron/weight_update/qwen35_direct_experts.py"
_SPEC = importlib.util.spec_from_file_location("qwen35_direct_test", _PATH)
direct = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(direct)


@pytest.mark.parametrize("rank", [0, 1])
def test_shards_match_separate_gate_up_and_down_slices_and_keep_old_storage(rank):
    gate_up = torch.arange(3 * 8 * 6, dtype=torch.bfloat16).reshape(3, 8, 6)
    down = torch.arange(3 * 6 * 4, dtype=torch.bfloat16).reshape(3, 6, 4)
    first = direct._shard_expert(gate_up, "gate_up_proj", rank)
    second = direct._shard_expert(down, "down_proj", rank)
    rows = [0, 1, 4, 5] if rank == 0 else [2, 3, 6, 7]
    columns = [0, 1] if rank == 0 else [2, 3]
    torch.testing.assert_close(first, gate_up[:, rows], rtol=0, atol=0)
    torch.testing.assert_close(second, down[:, :, columns], rtol=0, atol=0)
    assert first.is_contiguous() and second.is_contiguous()
    snapshots = first.clone(), second.clone()
    direct._shard_expert(gate_up + 5, "gate_up_proj", rank)
    direct._shard_expert(down + 5, "down_proj", rank)
    torch.testing.assert_close(first, snapshots[0], rtol=0, atol=0)
    torch.testing.assert_close(second, snapshots[1], rtol=0, atol=0)


@pytest.mark.parametrize(
    "projection,shape,target",
    [
        ("gate_up_proj", (256, 1024, 2048), (256, 512, 2048)),
        ("down_proj", (256, 2048, 512), (256, 2048, 256)),
    ],
)
def test_complete_model_shapes_and_runtime_names_without_allocating_storage(projection, shape, target):
    tensor = torch.empty(shape, dtype=torch.bfloat16, device="meta")
    for rank in (0, 1):
        result = direct.direct_expert_chunk(
            [(f"model.language_model.layers.9.mlp.experts.{projection}", tensor)], rank
        )
        suffix = "w13_weight" if projection == "gate_up_proj" else "w2_weight"
        assert result[0][0] == f"model.layers.9.mlp.experts.{suffix}"
        assert tuple(result[0][1].shape) == target
    assert direct.direct_expert_chunk([("lm_head.weight", tensor)], 0) is None
    assert direct.direct_expert_chunk([], 0) is None


def _layout():
    return dict(
        tp_size=2,
        ep_size=1,
        pp_size=1,
        moe_dp_size=1,
        moe_runner_backend="triton",
        dtype="bfloat16",
        disable_shared_experts_fusion=True,
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location="trivial",
        ep_join_mode=None,
        quantization=None,
        enable_lora=None,
        speculative_algorithm=None,
        json_model_override_args="{}",
    )


@pytest.mark.parametrize(
    "override",
    [
        dict(tp_size=4),
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
        dict(enable_lora=True),
        dict(speculative_algorithm="EAGLE"),
        dict(json_model_override_args='{"hidden_size":4096}'),
    ],
)
def test_unsupported_runtime_layout_is_rejected(override):
    direct.validate_direct_receiver(_layout())
    with pytest.raises(ValueError):
        direct.validate_direct_receiver(_layout() | override)


@pytest.mark.parametrize(
    "packets,formats",
    [
        ([[("direct", "d0")], [("direct", "d1")]], ["direct"]),
        ([[("flattened_bucket", "f0")], [("flattened_bucket", "f1")]], ["flattened_bucket"]),
        ([[], [("direct", "d1")]], ["direct"]),
        ([[("direct", "d0")], []], ["direct"]),
        ([[("direct", "d0")], [("flattened_bucket", "f1")]], ["direct", "flattened_bucket"]),
        ([[], []], []),
    ],
)
def test_mixed_and_empty_ranks_receive_one_consistent_format_per_request(packets, formats):
    calls, held = [], []

    def serialize(value, output_str):
        assert output_str
        return "empty-direct" if isinstance(value, list) else "empty-flat"

    def remote(**kwargs):
        calls.append(kwargs)
        return len(calls)

    refs = direct.dispatch_typed_buckets(
        packets,
        engine=SimpleNamespace(update_weights_from_tensor=SimpleNamespace(remote=remote)),
        version=12,
        serializer=SimpleNamespace(serialize=serialize),
        empty_flattened=lambda: dict(flattened_tensor=torch.empty(0, dtype=torch.uint8), metadata=[]),
        held=held,
    )
    assert len(refs) == len(formats) and [c["load_format"] for c in calls] == formats
    for call in calls:
        assert call["weight_version"] == "12"
        for rank, value in enumerate(call["serialized_named_tensors"]):
            if packets[rank] and packets[rank][0][0] == call["load_format"]:
                assert value == packets[rank][0][1]
            else:
                assert value == ("empty-direct" if call["load_format"] == "direct" else "empty-flat")
    assert all(isinstance(value, list) or value["flattened_tensor"].numel() == 0 for value in held)
