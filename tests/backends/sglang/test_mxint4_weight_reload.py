# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Preserve MXINT4 checkpoint loaders and captured kernel storage on reload."""

import ast
from importlib import metadata

import pytest
from packaging.version import Version

from tests.backends.sglang.source_utils import sglang_source_path


torch = pytest.importorskip("torch")
try:
    _sglang_version = metadata.version("sglang")
except metadata.PackageNotFoundError:
    pytest.skip("SGLang is not installed", allow_module_level=True)
pytestmark = pytest.mark.skipif(Version(_sglang_version) < Version("0.5.21"), reason="SGLang 0.5.21 MXINT4")

_NAMES = ("w13_weight_packed", "w13_weight_scale", "w2_weight_packed", "w2_weight_scale")


def _scheme():
    path = sglang_source_path(
        "srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CompressedTensorsMxInt4MoE"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"create_weights", "restore_weights_before_loading", "process_weights_after_loading"}
    ]
    namespace = {"torch": torch, "set_weight_attrs": lambda param, attrs: param.__dict__.update(attrs)}
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    scheme_type = type("Scheme", (), {node.name: namespace[node.name] for node in methods})
    scheme = scheme_type()
    scheme.packed_factor = 8
    scheme.group_size = 32
    scheme.strategy = "group"
    scheme._cache_permute_indices = {"stale": torch.tensor([99])}
    scheme.calls = 0

    def prepare(w13, w2, s13, s2, num_experts):
        assert not scheme._cache_permute_indices
        scheme._cache_permute_indices["temporary"] = torch.arange(4)
        scheme.calls += 1
        return _converted(w13, w2, s13, s2)

    scheme.prepare_static_weights_for_kernel = prepare
    layer = torch.nn.Module()
    loader = object()
    scheme.create_weights(layer, 2, 256, 128, torch.bfloat16, weight_loader=loader)
    return scheme, layer, loader


def _converted(w13, w2, s13, s2):
    # Use different dtype/shape layouts; real GPU shuffling is tested separately.
    return (
        w13.clone().view(torch.uint8).flatten(1),
        s13.clone().flatten(1),
        w2.clone().view(torch.uint8).flatten(1),
        s2.clone().flatten(1),
    )


def _fill(layer, value: int) -> None:
    for name in _NAMES:
        getattr(layer, name).data.fill_(value)


def test_mxint4_three_updates_preserve_loader_and_runtime_identity() -> None:
    scheme, layer, loader = _scheme()
    serialized = dict(layer._mxint4_serialized_params)
    metadata_params = {name: getattr(layer, name) for name in ("w13_weight_shape", "w2_weight_shape")}
    runtime = None
    pointers = None
    for cycle in range(3):
        scheme.restore_weights_before_loading(layer)
        assert all(getattr(layer, name) is param for name, param in serialized.items())
        assert all(param.weight_loader is loader for param in serialized.values())
        _fill(layer, cycle + 1)
        # A second restore between buckets must leave already received values intact.
        scheme.restore_weights_before_loading(layer)
        assert torch.all(layer.w13_weight_packed == cycle + 1)
        expected = _converted(
            layer.w13_weight_packed, layer.w2_weight_packed, layer.w13_weight_scale, layer.w2_weight_scale
        )
        scheme.process_weights_after_loading(layer)
        if runtime is None:
            runtime = dict(layer._mxint4_runtime_params)
            pointers = {name: param.data_ptr() for name, param in runtime.items()}
        for name, value in zip(_NAMES, expected, strict=True):
            assert getattr(layer, name) is runtime[name]
            assert getattr(layer, name).data_ptr() == pointers[name]
            torch.testing.assert_close(getattr(layer, name), value, rtol=0, atol=0)
        assert not layer._mxint4_loading
        assert not scheme._cache_permute_indices
        scheme.process_weights_after_loading(layer)
        assert scheme.calls == cycle + 1
    for name, param in metadata_params.items():
        assert getattr(layer, name) is param
        assert param.weight_loader is loader


@pytest.mark.parametrize("initial", [True, False])
def test_mxint4_conversion_failure_keeps_serialized_state_for_retry(initial: bool) -> None:
    scheme, layer, _ = _scheme()
    if not initial:
        _fill(layer, 1)
        scheme.process_weights_after_loading(layer)
        scheme.restore_weights_before_loading(layer)
    _fill(layer, 2)
    runtime_before = {name: param.clone() for name, param in layer._mxint4_runtime_params.items()}
    prepare = scheme.prepare_static_weights_for_kernel

    def fail(*args, **kwargs):
        scheme._cache_permute_indices["failed"] = torch.arange(4)
        raise RuntimeError("conversion failed")

    scheme.prepare_static_weights_for_kernel = fail
    with pytest.raises(RuntimeError, match="conversion failed"):
        scheme.process_weights_after_loading(layer)
    assert layer._mxint4_loading
    assert not scheme._cache_permute_indices
    assert all(getattr(layer, name) is param for name, param in layer._mxint4_serialized_params.items())
    for name, value in runtime_before.items():
        torch.testing.assert_close(layer._mxint4_runtime_params[name], value, rtol=0, atol=0)
    scheme.prepare_static_weights_for_kernel = prepare
    scheme.process_weights_after_loading(layer)
    assert not layer._mxint4_loading


@pytest.mark.parametrize("change", ["shape", "dtype", "stride"])
def test_mxint4_validates_all_layouts_before_modifying_runtime(change: str) -> None:
    scheme, layer, _ = _scheme()
    _fill(layer, 1)
    scheme.process_weights_after_loading(layer)
    runtime = dict(layer._mxint4_runtime_params)
    before = {name: param.clone() for name, param in runtime.items()}
    scheme.restore_weights_before_loading(layer)
    _fill(layer, 3)
    prepare = scheme.prepare_static_weights_for_kernel

    def changed(*args, **kwargs):
        result = list(prepare(*args, **kwargs))
        if change == "shape":
            result[-1] = result[-1].reshape(1, -1)
        elif change == "dtype":
            result[-1] = result[-1].float()
        else:
            result[-1] = result[-1].T.contiguous().T
        return tuple(result)

    scheme.prepare_static_weights_for_kernel = changed
    with pytest.raises(RuntimeError, match="MXINT4 runtime layout changed"):
        scheme.process_weights_after_loading(layer)
    assert layer._mxint4_loading
    assert not scheme._cache_permute_indices
    for name, value in before.items():
        assert layer._mxint4_runtime_params[name] is runtime[name]
        torch.testing.assert_close(runtime[name], value, rtol=0, atol=0)
