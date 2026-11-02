# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep streamed PLE tables outside ordinary batched TP weight gathers."""

import ast
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "layout",
    [
        ("dense", "dense"),
        ("ple", "dense"),
        ("dense", "ple", "dense"),
        ("dense", "ple"),
        ("ple", "ple"),
        ("ple", "dense", "ple", "dense"),
    ],
)
def test_bridge_iterator_preserves_ple_shards_with_async_gather(async_mode: bool, layout: tuple[str, ...]) -> None:
    source = Path(__file__).resolve().parents[4] / "relax/backends/megatron/weight_update/hf_weight_iterator_bridge.py"
    tree = ast.parse(source.read_text())
    owner = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HfWeightIteratorBridge"
    )
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_iter_hf_params")
    suffix = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_QWEN4EXP_PLE_TABLE_SUFFIX" for target in node.targets)
    )
    infos = [
        SimpleNamespace(
            name=f"decoder.layers.{index}."
            + ("per_layer_embedding.ple_embedding.ngram_embedding.weight" if kind == "ple" else "weight")
        )
        for index, kind in enumerate(layout)
    ]
    params = [torch.tensor([float(index)]) for index in range(len(layout))]
    dense_names = [info.name for info, kind in zip(infos, layout, strict=True) if kind == "dense"]
    gdn_split_sizes = {dense_names[0]: (1,)} if dense_names else {}
    async_calls, sync_calls, produced_shards = [], [], []

    def gather_async(pairs: list, *, gdn_split_sizes: dict) -> list[torch.Tensor]:
        assert gdn_split_sizes is actor._gdn_split_sizes
        assert [info.name for info, _ in pairs] == dense_names
        async_calls.append(dense_names)
        return [param + 100 for _, param in pairs]

    def gather_sync(args: SimpleNamespace, name: str, param: torch.Tensor) -> torch.Tensor:
        assert not async_mode and name in dense_names
        sync_calls.append(name)
        return param + 100

    def convert(name: str, param: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        assert name in dense_names
        return [(name, param)]

    def ple_shards(name: str, param: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        assert name not in dense_names
        for shard in range(2):
            produced_shards.append((name, shard))
            yield f"{name}.shard{shard}", param + shard

    converter = SimpleNamespace(
        init_tasks=lambda: None,
        broadcast_and_apply_configs=lambda: None,
        convert=convert,
        iter_ple_shards=ple_shards,
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(),
        _bridge_converter=converter,
        lora_merge_mode=False,
        _non_expert_buckets=[infos],
        _vanilla_key_map={},
        _async_weight_gather=async_mode,
        _gdn_split_sizes=gdn_split_sizes,
    )
    namespace = {
        "time": time,
        "dist": SimpleNamespace(get_rank=lambda: 1),
        "device_utils": SimpleNamespace(make_current_torch_device=lambda: torch.device("cpu")),
        "_load_and_broadcast": lambda *args, **kwargs: params,
        "all_gather_param": gather_sync,
        "all_gather_params_async": gather_async,
    }
    exec(compile(ast.Module(body=[suffix, method], type_ignores=[]), str(source), "exec"), namespace)
    stream = namespace["_iter_hf_params"](actor, {}, include_experts=False)
    assert produced_shards == []
    expected_shards = []
    for info, param, kind in zip(infos, params, layout, strict=True):
        for shard in range(2) if kind == "ple" else (None,):
            name, value = next(stream)
            if kind == "ple":
                expected_shards.append((info.name, shard))
                assert name == f"{info.name}.shard{shard}"
                torch.testing.assert_close(value, param + shard)
            else:
                assert name == info.name
                torch.testing.assert_close(value, param + 100)
            assert produced_shards == expected_shards, "PLE shards must be produced only when consumed"
    with pytest.raises(StopIteration):
        next(stream)
    assert async_calls == ([dense_names] if async_mode else [])
    assert sync_calls == ([] if async_mode else dense_names)
