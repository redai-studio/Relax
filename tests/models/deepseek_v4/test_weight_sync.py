# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions for DeepSeek-V4 fusion pairs across DCS bucket
boundaries."""

import ast
import builtins
import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
DEVICE_DIRECT = ROOT / "relax/distributed/checkpoint_service/backends/device_direct.py"
WEIGHT_SYNC = ROOT / "relax/models/deepseek_v4/weight_sync.py"


def _load_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture()
def weight_sync_functions(monkeypatch):
    namespace = {"re": re, "partial": partial}
    pairing_path = ROOT / "relax/backends/megatron/weight_update/hf_weight_iterator_bridge.py"
    tree = ast.parse(pairing_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in ("_FUSED_PAIR_FRAGMENTS", "_PAIR_SENTINEL")
            for target in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(pairing_path), "exec"), namespace)
    _load_functions(pairing_path, {"_fused_pair_key"}, namespace)
    _load_functions(WEIGHT_SYNC, {"split_fused_weight_pairs", "get_dcs_bucket_splitter"}, namespace)
    _load_functions(DEVICE_DIRECT, {"_update_weight_from_distributed"}, namespace, "DeviceDirectBackend")
    original_import = builtins.__import__

    def import_pair_key(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "relax.backends.megatron.weight_update.hf_weight_iterator_bridge":
            return SimpleNamespace(_fused_pair_key=namespace["_fused_pair_key"])
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_pair_key)
    return namespace


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (".wq_a.", ".wkv."),
        (".q_a_proj.", ".kv_a_proj_with_mqa."),
        (".compressor.wkv.", ".compressor.wgate."),
    ],
)
def test_fused_pairs_keep_weight_and_scale_counterparts_distinct(weight_sync_functions, left, right):
    split = weight_sync_functions["split_fused_weight_pairs"]
    key = weight_sync_functions["_fused_pair_key"]
    prefix = "model.layers.41.self_attn"
    weights = [(prefix + left + tail, object()) for tail in ("weight", "weight_scale_inv")]
    ready, pending = split(weights, key)
    assert ready == [] and pending == weights
    weights += [(prefix + right + tail, object()) for tail in ("weight", "weight_scale_inv")]
    ready, pending = split(weights, key, require_complete=True)
    assert ready == weights and pending == []


def test_compressor_halves_cannot_pair_with_attention_halves(weight_sync_functions):
    key = weight_sync_functions["_fused_pair_key"]
    split = weight_sync_functions["split_fused_weight_pairs"]
    weights = [
        ("layers.0.attn.wq_a.weight", object()),
        ("layers.0.attn.compressor.wkv.weight", object()),
    ]
    assert split(weights, key) == ([], weights)
    with pytest.raises(ValueError, match="unpaired fused weights"):
        split(weights, key, require_complete=True)
    with pytest.raises(ValueError, match="Duplicate fused weight"):
        split([weights[0], weights[0]], key)


def _size_only_tensor(size):
    return SimpleNamespace(numel=lambda: size, element_size=lambda: 1)


def _sender_buckets(weight_sync_functions, entries, limit, pairing):
    sent = []
    converted = []
    original = []
    state = SimpleNamespace(
        _is_pp_src_rank=True,
        _rollout_bucket_splitter=(
            weight_sync_functions["get_dcs_bucket_splitter"]("deepseekv4config", use_bridge=True, has_model=True)
            if pairing
            else None
        ),
        _lora_skip_rollout_base=False,
        _lora_enabled=False,
        _lora_merge_mode=False,
        _use_bridge=True,
        args=SimpleNamespace(update_weight_buffer_size=limit),
        _megatron=SimpleNamespace(
            all_gather_param=lambda args, name, param: param,
        ),
        _bridge_converter=SimpleNamespace(convert=lambda name, param: dict(entries)[name][1]),
        _update_bucket_weights_from_distributed=lambda tensors, **kwargs: sent.append(list(tensors)),
    )
    buffer_size = 0
    for name, (size, _) in entries:
        buffer_size = weight_sync_functions["_update_weight_from_distributed"](
            state, name, _size_only_tensor(size), converted, original, buffer_size, rollout_only=True
        )
    if converted:
        if state._rollout_bucket_splitter is not None:
            state._rollout_bucket_splitter(converted, require_complete=True)
        sent.append(list(converted))
    return sent


def _qkv_weights():
    prefix = "model.layers.41.self_attn."
    q = [(prefix + "wq_a.weight", torch.ones(2, 2)), (prefix + "wq_a.weight_scale_inv", torch.ones(2, 1))]
    kv = [(prefix + "wkv.weight", torch.ones(1, 2)), (prefix + "wkv.weight_scale_inv", torch.ones(1, 1))]
    return q, kv


def test_sender_carries_only_unpaired_tensors_across_oversized_and_interleaved_weights(weight_sync_functions):
    q, kv = _qkv_weights()
    huge = _size_only_tensor(2 << 30)
    entries = [
        ("huge_before", (2 << 30, [("before", huge)])),
        ("query", (700 << 20, q)),
        ("huge_between", (2 << 30, [("between", huge)])),
        ("key_value", (700 << 20, kv)),
        ("last", (700 << 20, [("last", _size_only_tensor(700 << 20))])),
    ]
    buckets = _sender_buckets(weight_sync_functions, entries, 1 << 30, pairing=True)
    names = [[name for name, _ in bucket] for bucket in buckets]
    assert names == [["before"], ["between"], [name for name, _ in q + kv], ["last"]]
    # Unrelated multi-GB tensors remain separate; no whole-layer buffering.
    assert len(buckets[0]) == len(buckets[1]) == 1


def test_other_models_keep_existing_bucket_order_and_limit(weight_sync_functions):
    entries = [
        (name, (size, [(name, _size_only_tensor(size))])) for name, size in [("a", 4), ("b", 4), ("c", 16), ("d", 2)]
    ]
    buckets = _sender_buckets(weight_sync_functions, entries, 8, pairing=False)
    assert [[name for name, _ in bucket] for bucket in buckets] == [
        ["a", "b"],
        ["c"],
        ["d"],
    ]


@pytest.mark.parametrize(
    ("name", "enabled"),
    [("deepseekv4config", True), ("deepseek_v4", True), ("DeepSeek-V4-Flash", True), ("qwen3config", False)],
)
def test_pairing_gate_covers_config_class_and_explicit_model_names(weight_sync_functions, name, enabled):
    factory = weight_sync_functions["get_dcs_bucket_splitter"]
    assert (factory(name, use_bridge=True, has_model=True) is not None) is enabled
    assert factory(name, use_bridge=False, has_model=True) is None


@pytest.mark.parametrize(
    ("has_model", "model_name", "expected"),
    [(False, None, False), (False, "deepseekv4config", False), (True, "deepseekv4config", True), (True, None, False)],
)
def test_pairing_factory_requires_a_model_and_name(weight_sync_functions, has_model, model_name, expected):
    factory = weight_sync_functions["get_dcs_bucket_splitter"]
    splitter = factory(model_name, use_bridge=True, has_model=has_model)
    assert (splitter is not None) is expected
