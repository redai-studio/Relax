# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real CPU route packing without importing the Ray actor or initializing
process groups."""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


torch = pytest.importorskip("torch")
_ROOT = Path(__file__).resolve().parents[3]
_LAYERS, _TOPK, _EXPERTS, _PADDED_LENGTH = 4, 2, 256, 24


def _source_nodes(path: str, names: set[str]) -> list[ast.FunctionDef]:
    nodes = [
        node
        for node in ast.parse((_ROOT / path).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in nodes} == names
    return nodes


def _compile_nodes(nodes: list[ast.AST], path: str, namespace: dict[str, Any]) -> dict[str, Any]:
    module = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[])
    exec(compile(module, path, "exec"), namespace)
    return namespace


def _runtime(cp_size: int, cp_rank: int, tp_size: int = 1, tp_rank: int = 0) -> dict[str, Any]:
    mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: cp_size,
        get_context_parallel_rank=lambda: cp_rank,
        get_tensor_model_parallel_world_size=lambda: tp_size,
        get_tensor_model_parallel_rank=lambda: tp_rank,
    )
    cp_path = "relax/backends/megatron/cp_utils.py"
    namespace = _compile_nodes(
        _source_nodes(cp_path, {"slice_bshd_contiguous_cp", "slice_with_cp", "maybe_padded_total_lengths"}),
        cp_path,
        {"torch": torch, "F": torch.nn.functional, "Callable": Callable, "mpu": mpu},
    )
    actor_path = "relax/backends/megatron/actor.py"
    return _compile_nodes(_source_nodes(actor_path, {"_pack_bshd_routing_replay"}), actor_path, namespace)


def _pad_routes(routes: Any, count: int) -> Any:
    rows = torch.arange(count * _LAYERS * _TOPK, dtype=routes.dtype, device=routes.device)
    rows = rows.reshape(count, _LAYERS, _TOPK) % _EXPERTS
    return torch.cat((routes, rows), dim=0)


def _samples(batch_size: int, shift: int = 0) -> tuple[list[Any], list[Any]]:
    token_lengths = [6, 10][:batch_size]
    tokens = [torch.arange(length) for length in token_lengths]
    routes = [
        (torch.arange((length - 1) * _LAYERS * _TOPK).reshape(length - 1, _LAYERS, _TOPK) + 16 + i * 128 + shift)
        % _EXPERTS
        for i, length in enumerate(token_lengths)
    ]
    return routes, tokens


def _expected_layout(routes: list[Any], cp_size: int, cp_rank: int, allgather_cp: bool) -> Any:
    """Independent index construction: sequence-major/batch-minor, not sample
    concatenation."""
    if allgather_cp or cp_size == 1:
        local_length = _PADDED_LENGTH // cp_size
        indices = torch.arange(cp_rank * local_length, (cp_rank + 1) * local_length)
    else:
        chunk = _PADDED_LENGTH // (2 * cp_size)
        indices = torch.cat(
            (
                torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk),
                torch.arange((2 * cp_size - cp_rank - 1) * chunk, (2 * cp_size - cp_rank) * chunk),
            )
        )
    per_sample = []
    for sample in routes:
        # Include the non-loss last token and every physical padding token.
        padding = torch.arange((_PADDED_LENGTH - sample.shape[0]) * _LAYERS * _TOPK, dtype=sample.dtype)
        padding = padding.reshape(-1, _LAYERS, _TOPK) % _EXPERTS
        per_sample.append(torch.cat((sample, padding), dim=0)[indices])
    return torch.stack(per_sample, dim=1).flatten(0, 1)


@pytest.mark.parametrize(("cp_size", "cp_rank"), [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("allgather_cp", [False, True])
def test_bshd_routes_use_common_physical_length_and_forward_token_order(
    cp_size: int, cp_rank: int, batch_size: int, allgather_cp: bool
) -> None:
    runtime = _runtime(cp_size, cp_rank, tp_size=2)
    routes, tokens = _samples(batch_size)
    before = [row.clone() for row in routes]
    pad_counts = []

    def pad_func(value: Any, count: int) -> Any:
        pad_counts.append(count)
        return _pad_routes(value, count)

    packed = runtime["_pack_bshd_routing_replay"](
        routes,
        [_PADDED_LENGTH] * batch_size,
        allgather_cp=allgather_cp,
        cp_size=cp_size,
        cp_rank=cp_rank,
        pad_func=pad_func,
    )

    assert packed.shape == (_PADDED_LENGTH // cp_size * batch_size, _LAYERS, _TOPK)
    torch.testing.assert_close(packed, _expected_layout(routes, cp_size, cp_rank, allgather_cp))
    assert [count for count in pad_counts if count] == [_PADDED_LENGTH - row.shape[0] for row in routes]
    assert packed.min().item() >= 0 and packed.max().item() < _EXPERTS
    for old, new in zip(before, routes, strict=True):
        torch.testing.assert_close(old, new)
    # L=24 is deliberately larger than each sample's own TP/CP rounding.
    for token in tokens:
        alignment = 2 * cp_size * 2
        assert ((token.shape[0] + alignment - 1) // alignment) * alignment < _PADDED_LENGTH


@pytest.mark.parametrize(
    ("batch_size", "max_seq_lens", "message"),
    [
        (2, [24], "every sample"),
        (2, [24, 16], "common padded length"),
        (2, [0, 0], "common padded length"),
        (1, [5], "final unrecorded token"),
        (1, [4], "final unrecorded token"),
    ],
)
def test_bshd_routes_reject_invalid_metadata_or_missing_final_token_slot(
    batch_size: int, max_seq_lens: list[int], message: str
) -> None:
    runtime = _runtime(1, 0)
    routes, _ = _samples(batch_size)
    with pytest.raises(ValueError, match=message):
        runtime["_pack_bshd_routing_replay"](
            routes,
            max_seq_lens,
            allgather_cp=True,
            cp_size=1,
            cp_rank=0,
            pad_func=_pad_routes,
        )


class _BatchIterator:
    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self.batches = batches
        self.index = 0
        self.reset_count = 0
        self.requested_fields: list[list[str]] = []

    def reset(self) -> None:
        self.index = 0
        self.reset_count += 1

    def get_next(self, fields: list[str]) -> dict[str, Any]:
        self.requested_fields.append(list(fields))
        batch = self.batches[self.index]
        self.index += 1
        return {field: batch[field] for field in fields}


def _load_fill_method(runtime: dict[str, Any]) -> Callable[..., None]:
    actor_path = "relax/backends/megatron/actor.py"
    tree = ast.parse((_ROOT / actor_path).read_text())
    methods = [
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "fill_routing_replay"
    ]
    assert len(methods) == 1
    return _compile_nodes(methods, actor_path, runtime)["fill_routing_replay"]


def _install_replay_class(monkeypatch: pytest.MonkeyPatch) -> Any:
    path = "relax/utils/training/routing_replay.py"
    classes = [
        node
        for node in ast.parse((_ROOT / path).read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "RoutingReplay"
    ]
    assert len(classes) == 1
    replay_class = _compile_nodes(classes, path, {"torch": torch})["RoutingReplay"]
    # CPU layout tests do not allocate CUDA-backed pinned memory.
    monkeypatch.setattr(replay_class, "record", lambda replay, value: replay.top_indices_list.append(value.clone()))
    module = ModuleType("relax.utils.training.routing_replay")
    module.RoutingReplay = replay_class
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return replay_class


@pytest.mark.parametrize(
    ("tp_size", "tp_rank", "cp_size", "cp_rank", "batch_size", "sequence_parallel", "pp_rank"),
    [
        (1, 0, 1, 0, 1, False, 0),
        (1, 0, 2, 1, 2, False, 1),
        (2, 0, 1, 0, 2, False, 0),
        (2, 1, 1, 0, 1, True, 1),
        (2, 0, 2, 0, 2, True, 0),
        (2, 1, 2, 0, 2, True, 1),
        (2, 0, 2, 1, 2, True, 1),
        (2, 1, 2, 1, 2, True, 0),
    ],
)
def test_fill_routing_replay_records_pp_layers_after_sequence_major_sp_slice(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    tp_rank: int,
    cp_size: int,
    cp_rank: int,
    batch_size: int,
    sequence_parallel: bool,
    pp_rank: int,
) -> None:
    runtime = _runtime(cp_size, cp_rank, tp_size, tp_rank)
    replay_class = _install_replay_class(monkeypatch)
    layer_build = ModuleType("megatron.core.transformer.transformer_block")
    layer_build.get_num_layers_to_build = lambda config, vp_stage: 2
    layer_offset = ModuleType("megatron.core.transformer.transformer_layer")
    layer_offset.get_transformer_layer_offset = lambda config, vp_stage: pp_rank * 2
    monkeypatch.setitem(sys.modules, layer_build.__name__, layer_build)
    monkeypatch.setitem(sys.modules, layer_offset.__name__, layer_offset)
    fill = _load_fill_method(runtime)
    batches, original_routes = [], []
    for shift in (0, 7):
        routes, tokens = _samples(batch_size, shift)
        original_routes.append(routes)
        batches.append(
            {
                "rollout_routed_experts": [row.flatten(1) for row in routes],
                "tokens": tokens,
                "max_seq_lens": [_PADDED_LENGTH] * batch_size,
            }
        )
    iterator = _BatchIterator(batches)
    other_iterator = _BatchIterator([])
    actor = SimpleNamespace(
        args=SimpleNamespace(
            num_layers=_LAYERS,
            num_experts=_EXPERTS,
            moe_router_topk=_TOPK,
            qkv_format="bshd",
            allgather_cp=True,
            sequence_parallel=sequence_parallel,
            data_pad_size_multiplier=2,
            is_vl_model=False,
        ),
        model=[SimpleNamespace(module=SimpleNamespace(config=SimpleNamespace(num_layers=_LAYERS, moe_layer_freq=1)))],
    )
    replays = [replay_class(), replay_class()]
    rollout = {"rollout_routed_experts": object(), "other_field": "keep"}

    fill(actor, [iterator, other_iterator], [1, 1], rollout)

    assert all(
        set(fields) == {"rollout_routed_experts", "tokens", "max_seq_lens"} for fields in iterator.requested_fields
    )
    assert len(iterator.requested_fields) == 2
    assert iterator.reset_count == other_iterator.reset_count == 2
    assert iterator.index == 0
    assert rollout == {"other_field": "keep"}
    for microbatch, routes in enumerate(original_routes):
        expected = _expected_layout(routes, cp_size, cp_rank, True)
        if sequence_parallel:
            expected = expected.chunk(tp_size, dim=0)[tp_rank]
        for local_layer, replay in enumerate(replays):
            assert len(replay.top_indices_list) == 2
            torch.testing.assert_close(replay.top_indices_list[microbatch], expected[:, pp_rank * 2 + local_layer])


def _expected_thd_layout(routes: list[Any], *, is_vl_model: bool, allgather_cp: bool, cp_rank: int) -> Any:
    def zigzag(sample: Any, padded_length: int) -> Any:
        sample = _pad_routes(sample, padded_length - sample.shape[0])
        chunk = padded_length // 4
        return torch.cat(
            (sample[cp_rank * chunk : (cp_rank + 1) * chunk], sample[(3 - cp_rank) * chunk : (4 - cp_rank) * chunk])
        )

    if is_vl_model:
        # The pre-existing bridge VL path aligns each full sample to TP*CP*2=8.
        local = [zigzag(sample, ((sample.shape[0] + 1 + 7) // 8) * 8) for sample in routes]
        return torch.cat(local, dim=0)
    samples = [_pad_routes(sample, 1) for sample in routes]
    if allgather_cp:
        # Non-VL THD concatenates samples first, globally pads to CP*TP*mult=8,
        # then takes a contiguous CP chunk. It must not gain BSHD interleaving.
        joined = torch.cat(samples, dim=0)
        joined = _pad_routes(joined, (-joined.shape[0]) % 8)
        return joined.chunk(2, dim=0)[cp_rank]
    local = [zigzag(sample, ((sample.shape[0] + 3) // 4) * 4) for sample in samples]
    joined = torch.cat(local, dim=0)
    return _pad_routes(joined, (-joined.shape[0]) % 4)


@pytest.mark.parametrize(
    ("is_vl_model", "allgather_cp", "cp_rank"),
    [(False, False, 0), (False, True, 1), (True, False, 0), (True, True, 1)],
)
def test_fill_routing_replay_preserves_thd_vl_and_non_vl_layout_without_bshd_metadata(
    monkeypatch: pytest.MonkeyPatch, is_vl_model: bool, allgather_cp: bool, cp_rank: int
) -> None:
    runtime = _runtime(2, cp_rank, tp_size=2, tp_rank=1)
    replay_class = _install_replay_class(monkeypatch)
    layer_build = ModuleType("megatron.core.transformer.transformer_block")
    layer_build.get_num_layers_to_build = lambda config, vp_stage: _LAYERS
    layer_offset = ModuleType("megatron.core.transformer.transformer_layer")
    layer_offset.get_transformer_layer_offset = lambda config, vp_stage: 0
    monkeypatch.setitem(sys.modules, layer_build.__name__, layer_build)
    monkeypatch.setitem(sys.modules, layer_offset.__name__, layer_offset)
    fill = _load_fill_method(runtime)
    routes, tokens = _samples(2)
    # Omitting max_seq_lens is intentional: legacy THD never requests it.
    iterator = _BatchIterator([{"rollout_routed_experts": [row.flatten(1) for row in routes], "tokens": tokens}])
    actor = SimpleNamespace(
        args=SimpleNamespace(
            num_layers=_LAYERS,
            num_experts=_EXPERTS,
            moe_router_topk=_TOPK,
            qkv_format="thd",
            allgather_cp=allgather_cp,
            sequence_parallel=True,
            data_pad_size_multiplier=2,
            is_vl_model=is_vl_model,
        ),
        model=[SimpleNamespace(module=SimpleNamespace(config=SimpleNamespace(num_layers=_LAYERS, moe_layer_freq=1)))],
    )
    replays = [replay_class() for _ in range(_LAYERS)]
    rollout = {"rollout_routed_experts": object()}

    fill(actor, [iterator], [1], rollout)

    assert iterator.requested_fields == [["rollout_routed_experts", "tokens"]]
    assert iterator.reset_count == 2 and iterator.index == 0
    assert rollout == {}
    expected = _expected_thd_layout(routes, is_vl_model=is_vl_model, allgather_cp=allgather_cp, cp_rank=cp_rank)
    expected = expected.chunk(2, dim=0)[1]
    for layer, replay in enumerate(replays):
        assert len(replay.top_indices_list) == 1
        torch.testing.assert_close(replay.top_indices_list[0], expected[:, layer])
