# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep CPU int32 rollout replay identical to the previous int64 layout.

Execute the real actor method and CP helpers without importing Megatron, Ray,
or a model. CUDA transfer and pinned-memory behavior have separate tests.
"""

import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


_ROOT = Path(__file__).resolve().parents[3]
_NUM_LAYERS = 8
_TOPK = 2
_TOKEN_LENGTHS = [1, 6, 11, 17]
_MICROBATCHES = [[2, 0], [3, 1]]


def _load_function(relative_path: str, name: str, namespace: dict) -> Callable:
    source = _ROOT / relative_path
    tree = ast.parse(source.read_text())
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(functions) == 1
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


class _ReplayIterator:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.offset = 0
        self.resets = 0

    def reset(self) -> None:
        self.offset = 0
        self.resets += 1

    def get_next(self, keys: list[str]) -> dict:
        indices = _MICROBATCHES[self.offset]
        self.offset += 1
        return {key: [self.data[key][index] for index in indices] for key in keys}


class _RecordedLayer:
    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id
        self.entries = []

    def record(self, top_indices: torch.Tensor) -> None:
        # The actor must finish layout selection without any CUDA allocation.
        assert top_indices.device.type == "cpu"
        self.entries.append(top_indices.clone())


@pytest.fixture
def replay_method(monkeypatch):
    registry = SimpleNamespace(all_routing_replays=[])
    block_module = ModuleType("megatron.core.transformer.transformer_block")
    block_module.get_num_layers_to_build = lambda config, vp_stage: (
        config.layer_spans[vp_stage][1] - config.layer_spans[vp_stage][0]
    )
    layer_module = ModuleType("megatron.core.transformer.transformer_layer")
    layer_module.get_transformer_layer_offset = lambda config, vp_stage: config.layer_spans[vp_stage][0]
    replay_module = ModuleType("relax.utils.training.routing_replay")
    replay_module.RoutingReplay = registry
    for module in (block_module, layer_module, replay_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    parallel = SimpleNamespace()
    namespace = {"torch": torch, "mpu": parallel, "Callable": Callable, "F": torch.nn.functional}
    for name in ("slice_with_cp", "maybe_padded_total_lengths"):
        _load_function("relax/backends/megatron/cp_utils.py", name, namespace)
    fill = _load_function("relax/backends/megatron/actor.py", "fill_routing_replay", namespace)
    return fill, parallel, registry


def _run_fill(
    replay_method,
    *,
    dtype: torch.dtype,
    layout: str,
    cp_size: int,
    cp_rank: int,
    tp_size: int,
    tp_rank: int,
    pp_rank: int,
    sequence_parallel: bool,
    virtual_pipeline: bool,
    moe_layer_freq: int | list[int],
) -> list[_RecordedLayer]:
    fill, parallel, registry = replay_method
    parallel.get_context_parallel_world_size = lambda: cp_size
    parallel.get_context_parallel_rank = lambda: cp_rank
    parallel.get_tensor_model_parallel_world_size = lambda: tp_size
    parallel.get_tensor_model_parallel_rank = lambda: tp_rank

    if virtual_pipeline:
        spans = [(2 * pp_rank, 2 * pp_rank + 2), (2 * pp_rank + 4, 2 * pp_rank + 6)]
    else:
        spans = [(4 * pp_rank, 4 * pp_rank + 4)]
    config = SimpleNamespace(num_layers=_NUM_LAYERS, moe_layer_freq=moe_layer_freq, layer_spans=spans)
    models = [SimpleNamespace(module=SimpleNamespace(config=config)) for _ in spans]
    local_moe_layers = [
        layer_id
        for start, end in spans
        for layer_id in range(start, end)
        if (layer_id % moe_layer_freq == 0 if isinstance(moe_layer_freq, int) else bool(moe_layer_freq[layer_id]))
    ]
    records = [_RecordedLayer(layer_id) for layer_id in local_moe_layers]
    registry.all_routing_replays = records
    data = {
        "tokens": [torch.arange(length) for length in _TOKEN_LENGTHS],
        "rollout_routed_experts": [
            torch.arange((length - 1) * _NUM_LAYERS * _TOPK, dtype=dtype).reshape(length - 1, _NUM_LAYERS * _TOPK)
            + sample_id * 1000
            for sample_id, length in enumerate(_TOKEN_LENGTHS)
        ],
    }
    args = SimpleNamespace(
        num_layers=_NUM_LAYERS,
        moe_router_topk=_TOPK,
        num_experts=65536,
        qkv_format="thd",
        is_vl_model=layout == "vl",
        allgather_cp=layout == "allgather",
        sequence_parallel=sequence_parallel,
        data_pad_size_multiplier=4,
    )
    actor = SimpleNamespace(args=args, model=models)
    iterators = [_ReplayIterator(data) for _ in spans]

    fill(actor, iterators, [1, 1], data)

    assert "rollout_routed_experts" not in data
    assert all(iterator.offset == 0 and iterator.resets == 2 for iterator in iterators)
    assert [record.layer_id for record in records] == local_moe_layers
    assert all(len(record.entries) == len(_MICROBATCHES) for record in records)
    return records


@pytest.mark.parametrize(
    "layout,cp_size,tp_size,sequence_parallel",
    [
        pytest.param("zigzag", 1, 1, False, id="no-parallelism"),
        pytest.param("zigzag", 4, 2, False, id="zigzag-cp"),
        pytest.param("zigzag", 4, 2, True, id="zigzag-cp-tp-sp"),
        pytest.param("allgather", 4, 2, False, id="allgather-cp"),
        pytest.param("allgather", 4, 2, True, id="allgather-cp-tp-sp"),
        pytest.param("vl", 4, 2, False, id="vl-per-sample-padding"),
        pytest.param("vl", 4, 2, True, id="vl-cp-tp-sp"),
    ],
)
@pytest.mark.parametrize("virtual_pipeline", [False, True], ids=["pp", "pp-vpp"])
@pytest.mark.parametrize(
    "moe_layer_freq", [2, [0, 1, 0, 1, 1, 0, 0, 1]], ids=["integer-dense-frequency", "explicit-dense-frequency"]
)
def test_fill_routing_replay_cpu_int32_matches_legacy_int64_layout(
    replay_method,
    layout: str,
    cp_size: int,
    tp_size: int,
    sequence_parallel: bool,
    virtual_pipeline: bool,
    moe_layer_freq: int | list[int],
) -> None:
    # Every local CP/TP/PP coordinate sees the same source batch. The actor
    # must select exactly the same token rows and local MoE layers as before.
    for cp_rank in range(cp_size):
        for tp_rank in range(tp_size):
            for pp_rank in range(2):
                options = dict(
                    layout=layout,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    pp_rank=pp_rank,
                    sequence_parallel=sequence_parallel,
                    virtual_pipeline=virtual_pipeline,
                    moe_layer_freq=moe_layer_freq,
                )
                legacy = _run_fill(replay_method, dtype=torch.int64, **options)
                compact = _run_fill(replay_method, dtype=torch.int32, **options)

                assert [layer.layer_id for layer in compact] == [layer.layer_id for layer in legacy]
                for old_layer, new_layer in zip(legacy, compact, strict=True):
                    for expected, actual in zip(old_layer.entries, new_layer.entries, strict=True):
                        assert expected.dtype == torch.int64
                        assert actual.dtype == torch.int32
                        assert actual.device.type == "cpu"
                        assert torch.equal(actual.to(torch.int64), expected)
