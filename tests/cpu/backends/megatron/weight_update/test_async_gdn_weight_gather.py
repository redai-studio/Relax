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
    assert len(waited) == 3
    assert len(result) == len(expected)
    assert all(
        torch.equal(actual.view(torch.uint8), target.view(torch.uint8))
        for actual, target in zip(result, expected, strict=True)
    )
    assert result[-1].data_ptr() == replicated.data_ptr()


@pytest.mark.parametrize(("tp", "reuse"), [(1, True), (1, False), (2, True)])
def test_tp1_can_reuse_an_ordinary_broadcast_buffer(monkeypatch, tp, reuse):
    from relax.backends.megatron.weight_update import common

    source = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
    source.tensor_model_parallel = True
    source.partition_dim = 0
    calls = []

    def gather(outputs, local, *, group, async_op):
        calls.append(group)
        for part in outputs:
            part.copy_(local)

    monkeypatch.setattr(common.dist, "all_gather", gather)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_group", lambda: "tp")
    output = common.all_gather_param(None, "embedding.word_embeddings.weight", source, reuse_tp1_buffer=reuse)
    expect_reuse = tp == 1 and reuse
    assert calls == ([] if expect_reuse else ["tp"])
    assert (output.data_ptr() == source.data_ptr()) == expect_reuse
    assert torch.equal(output, source.repeat(tp, 1))


@pytest.mark.parametrize(
    ("partition_dim", "backend", "min_bytes", "use_direct"),
    [(0, "nccl", 1, True), (0, "nccl", 128, False), (1, "nccl", 1, False), (0, "gloo", 1, False)],
)
def test_oversized_row_gather_writes_the_final_layout(monkeypatch, partition_dim, backend, min_bytes, use_direct):
    from relax.backends.megatron.weight_update import common

    local = torch.arange(6, dtype=torch.bfloat16).view(2, 3).T
    peer = local + 10
    local.tensor_model_parallel = True
    local.partition_dim = partition_dim
    calls = []

    def gather_direct(output, source, *, group, async_op):
        assert group == "tp" and async_op and source.is_contiguous()
        assert output.shape == (6, 2)
        calls.append(("direct", output))
        return SimpleNamespace(wait=lambda: output.copy_(torch.cat([local, peer], dim=0)))

    def gather_shards(outputs, source, *, group, async_op):
        assert group == "tp" and async_op
        calls.append(("list", outputs))

        def wait():
            outputs[0].copy_(source)
            outputs[1].copy_(peer)

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(common.dist, "all_gather_into_tensor", gather_direct)
    monkeypatch.setattr(common.dist, "all_gather", gather_shards)
    monkeypatch.setattr(common.dist, "get_backend", lambda group: backend)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_group", lambda: "tp")
    (result,) = common.all_gather_params_async(
        [(SimpleNamespace(name="weight"), local)], contiguous_output_min_bytes=min_bytes
    )
    assert torch.equal(result, torch.cat([local, peer], dim=partition_dim))
    assert [method for method, _ in calls] == ["direct" if use_direct else "list"]
    if use_direct:
        assert result is calls[0][1]


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
