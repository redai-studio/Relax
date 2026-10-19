# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reconstruct known global GDN tensors from real TP-local component shards."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("wrapped", [False, True])
def test_async_gather_preserves_gdn_glu_and_replicated_layouts(tp, wrapped):
    path = Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_update/common.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "all_gather_params_async"
    )
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
        param.partition_dim = 0
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
        assert group == "tp" and async_op
        shards = sources[param.data_ptr()]
        pending.append(outputs)

        def wait():
            assert len(pending) == 3, "Every gather must launch before waiting"
            for out, source in zip(outputs, shards, strict=True):
                out.copy_(source)
            waited.append(True)

        return SimpleNamespace(wait=wait)

    namespace = {
        "torch": torch,
        "dist": SimpleNamespace(all_gather=all_gather),
        "mpu": SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: tp, get_tensor_model_parallel_group=lambda: "tp"
        ),
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    result = namespace["all_gather_params_async"](inputs, gdn_split_sizes=split_sizes)
    assert len(waited) == 3
    assert len(result) == len(expected)
    assert all(
        torch.equal(actual.view(torch.uint8), target.view(torch.uint8))
        for actual, target in zip(result, expected, strict=True)
    )
    assert result[-1].data_ptr() == replicated.data_ptr()
