# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reconstruct known global GDN tensors from real TP-local component shards."""

from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("megatron.core", reason="TP gather tests require Megatron Core", exc_type=ImportError)


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_gather_preserves_gdn_glu_and_replicated_layouts(tp, wrapped, async_mode, monkeypatch):
    from relax.backends.megatron.weight_update import common

    inputs, sources, expected, split_sizes = [], {}, [], {}
    specs = [
        ("self_attention.conv1d.weight", (2, 2, 4), (1, 4)),
        ("self_attention.in_proj.weight", (2, 2, 4, 4, 1, 1), (3,)),
        ("mlp.linear_fc1.weight", (3, 3), (2,)),
    ]
    for parameter_index, (suffix, sizes, rest) in enumerate(specs):
        name = "decoder.layers.0." + suffix
        components = []
        for component, size in enumerate(sizes):
            shape = (size * tp, *rest)
            values = torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
            components.append((values + 1000 * component + 10000 * parameter_index).to(torch.bfloat16))
        shards = [
            torch.cat([component.chunk(tp, dim=0)[rank] for component in components], dim=0) for rank in range(tp)
        ]
        param = shards[0].clone()
        param.tensor_model_parallel = True
        # Ascend can report non-row metadata for a GDN weight when TP=1.
        param.partition_dim = -1 if tp == 1 and "self_attention" in name else 0
        param.partition_stride = 2 if "linear_fc1" in name else 1
        full_name = name.replace(".weight", ".to_wrap.weight") if wrapped else name
        inputs.append((SimpleNamespace(name=full_name), param))
        sources[param.data_ptr()] = shards
        expected.append(torch.cat(components, dim=0))
        if "self_attention" in name:
            split_sizes[name] = sizes
    replicated = torch.tensor([7, 11], dtype=torch.bfloat16)
    replicated.tensor_model_parallel = False
    inputs.append((SimpleNamespace(name="replicated.weight"), replicated))
    expected.append(replicated)
    pending, waited = [], []

    def all_gather(outputs, param, *, group, async_op):
        assert group == "tp" and async_op == async_mode
        shards = sources[param.data_ptr()]
        pending.append(outputs)

        def wait():
            if async_mode:
                assert len(pending) == 3, "Every gather must launch before waiting"
            for out, source in zip(outputs, shards, strict=True):
                out.copy_(source)
            waited.append(True)

        if async_mode:
            return SimpleNamespace(wait=wait)
        wait()
        return None

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    monkeypatch.setattr(
        common,
        "mpu",
        SimpleNamespace(get_tensor_model_parallel_world_size=lambda: tp, get_tensor_model_parallel_group=lambda: "tp"),
    )
    if async_mode:
        result = common.all_gather_params_async(inputs, gdn_split_sizes=split_sizes)
    else:
        config = SimpleNamespace(
            linear_key_head_dim=2, linear_value_head_dim=4, linear_num_key_heads=tp, linear_num_value_heads=tp
        )
        monkeypatch.setattr(common, "get_hf_config", lambda path: SimpleNamespace(text_config=config))
        result = [
            common.all_gather_param(SimpleNamespace(hf_checkpoint="test"), info.name, param) for info, param in inputs
        ]
    assert len(waited) == (0 if tp == 1 else 3)
    assert len(result) == len(expected)
    assert all(
        torch.equal(actual.view(torch.uint8), target.view(torch.uint8))
        for actual, target in zip(result, expected, strict=True)
    )
    assert result[-1].data_ptr() == replicated.data_ptr()


@pytest.mark.parametrize("kind", ["replicated", "duplicated", "buffer"])
def test_non_sharded_gdn_named_tensor_needs_no_gdn_config(monkeypatch, kind):
    from relax.backends.megatron.weight_update import common

    tensor = torch.ones(4)
    if kind != "buffer":
        tensor.tensor_model_parallel = kind == "duplicated"
        tensor.parallel_mode = "duplicated" if kind == "duplicated" else None
    monkeypatch.setattr(common, "get_hf_config", lambda *a: pytest.fail("unneeded config lookup"))
    actual = common.all_gather_param(SimpleNamespace(), "decoder.layers.0.self_attention.in_proj.weight", tensor)
    assert actual.data_ptr() == tensor.data_ptr()


@pytest.mark.parametrize("expert", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_single_rank_gather_uses_existing_storage(monkeypatch, expert, async_mode):
    from relax.backends.megatron.weight_update import common

    tensor = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
    tensor.tensor_model_parallel = True
    tensor.partition_dim = 0
    tensor.partition_stride = 2
    name = "decoder.layers.0.mlp." + ("experts." if expert else "") + "linear_fc1.weight"

    def unexpected(*args, **kwargs):
        pytest.fail("A single-rank gather must not allocate or enter a device collective")

    monkeypatch.setattr(torch, "empty_like", unexpected)
    monkeypatch.setattr(torch, "cat", unexpected)
    monkeypatch.setattr(torch.distributed, "all_gather", unexpected)
    monkeypatch.setattr(
        common,
        "mpu",
        SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 2 if expert else 1,
            get_expert_tensor_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_group=unexpected,
            get_expert_tensor_parallel_group=unexpected,
        ),
    )
    agreements = []
    if async_mode:
        actual = common.all_gather_params_async([(SimpleNamespace(name=name), tensor)], agreement=agreements.append)[0]
    else:
        actual = common.all_gather_param(SimpleNamespace(), name, tensor, agreement=agreements.append)
    assert agreements == [None] * (2 if async_mode else 3)
    assert actual.data_ptr() == tensor.data_ptr()
    assert torch.equal(actual.view(torch.uint8), tensor.view(torch.uint8))
