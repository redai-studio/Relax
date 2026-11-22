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

    monkeypatch.setattr(common.dist, "all_gather", all_gather)
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
    if tp == 1:
        for converted, (_, source) in zip(result[:-1], inputs[:-1], strict=True):
            converted.zero_()
            assert source.count_nonzero() > 0


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


@pytest.mark.parametrize("gated", [True, False])
def test_column_parallel_fc1_matches_actual_mlp_weight_and_bias_layout(monkeypatch, gated):
    from relax.backends.megatron.weight_update import common

    mlp = torch.nn.Module()
    mlp.config = SimpleNamespace(gated_linear_unit=gated)
    mlp.linear_fc1 = torch.nn.Linear(2, 4, bias=True)
    layouts = common.fc1_gating_by_parameter([mlp])
    parts = {}
    inputs = []
    expected = []
    for name, param in mlp.linear_fc1.named_parameters():
        global_rows = torch.arange(8 * param[0].numel()).reshape(8, *param.shape[1:]).to(param)
        # In a nongated activation each rank owns a contiguous row range.
        pieces = global_rows.chunk(4 if gated else 2, dim=0)
        shards = [torch.cat((pieces[i], pieces[i + 2])) if gated else pieces[i] for i in range(2)]
        with torch.no_grad():
            param.copy_(shards[0])
        param.gated_fc1 = layouts[id(param)]
        param.tensor_model_parallel = True
        param.partition_dim = 0
        inputs.append((SimpleNamespace(name=f"decoder.layers.0.mlp.linear_fc1.{name}"), param))
        parts[param.data_ptr()] = shards
        expected.append(global_rows)

    def gather(outputs, value, **kwargs):
        for output, shard in zip(outputs, parts[value.data_ptr()], strict=True):
            output.copy_(shard)

    monkeypatch.setattr(common.dist, "all_gather", gather)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_group", lambda: object())
    result = common.all_gather_params_async(inputs)
    for actual, reference in zip(result, expected, strict=True):
        assert torch.equal(actual, reference)
