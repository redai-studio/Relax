# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""MoE weight reloads and native/legacy weight update sessions."""

import ast
import sys
from contextlib import nullcontext
from importlib import metadata
from types import MethodType, ModuleType, SimpleNamespace

import pytest
from packaging.version import Version

from tests.backends.sglang.source_utils import sglang_source_path


torch = pytest.importorskip("torch")
try:
    _sglang_version = metadata.version("sglang")
except metadata.PackageNotFoundError:
    pytest.skip("SGLang is not installed", allow_module_level=True)
pytestmark = pytest.mark.skipif(
    Version(_sglang_version) < Version("0.5.21"), reason="SGLang 0.5.21 MoE weight updates"
)


def _restore(backend: str):
    path = sglang_source_path("srt/layers/quantization/unquant.py")
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UnquantizedFusedMoEMethod"
    )
    function = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load"
    )
    namespace = {
        "torch": torch,
        "get_moe_runner_backend": lambda: SimpleNamespace(
            is_flashinfer_trtllm_routed=lambda: backend == "flashinfer_trtllm_routed",
            is_flashinfer_trtllm=lambda: backend == "flashinfer_trtllm",
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name], SimpleNamespace(use_flashinfer_trtllm_moe=backend.startswith("flashinfer_trtllm"))


@pytest.mark.parametrize(
    "backend,weight_name,is_gated",
    [("flashinfer_trtllm", "w13_weight", True), ("flashinfer_trtllm_routed", "w2_weight", False)],
)
def test_bf16_moe_hot_copy_preserves_storage(backend: str, weight_name: str, is_gated: bool) -> None:
    restore, method = _restore(backend)
    layer = SimpleNamespace(
        num_local_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=256,
        moe_runner_config=SimpleNamespace(is_gated=is_gated),
    )
    if weight_name == "w13_weight":
        shape = (2, 256 * (2 if is_gated else 1), 128)
    else:
        shape = (2, 128, 256)
    expected = torch.arange(shape[0] * shape[1] * shape[2], dtype=torch.float32).reshape(shape).to(torch.bfloat16)
    blocked_shape = (shape[0], shape[1] * shape[2] // 64, 64)
    param = torch.nn.Parameter(torch.zeros_like(expected).reshape(blocked_shape), requires_grad=False)
    pointer = param.data_ptr()
    for value in (1, 3):
        restore(method, layer, param, f"model.layers.0.mlp.experts.{weight_name}")
        assert tuple(param.shape) == shape
        assert param.data_ptr() == pointer
        param.copy_(expected + value)
        # Later buckets must not reinterpret or overwrite the values already copied.
        restore(method, layer, param, f"model.layers.0.mlp.experts.{weight_name}")
        torch.testing.assert_close(param, expected + value, rtol=0, atol=0)
        assert param.data_ptr() == pointer
        param.data = param.data.reshape(blocked_shape)


def _lifecycle_method(relative_path: str, class_name: str, method_name: str):
    path = sglang_source_path(relative_path)
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {
        "torch": torch,
        "ContinueGenerationReqInput": object,
        "DisaggregationMode": SimpleNamespace(DECODE="decode"),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("explicit_postprocess", [False])
def test_bf16_moe_legacy_resume_repacks_once(explicit_postprocess: bool) -> None:
    finalize = _lifecycle_method(
        "srt/managers/scheduler_components/weight_updater.py",
        "SchedulerWeightUpdaterManager",
        "finalize_pending_bf16_weights",
    )
    layer = torch.nn.Module()
    layer._relax_trtllm_bf16_needs_repack = True
    layer.register_parameter("weight", torch.nn.Parameter(torch.arange(64.0).reshape(1, 8, 8), requires_grad=False))
    pointer = layer.weight.data_ptr()
    calls = []

    def pack(module):
        calls.append(module)
        module.weight.data = module.weight.reshape(1, 2, 4, 8)
        module._relax_trtllm_bf16_needs_repack = False

    layer.quant_method = SimpleNamespace(process_weights_after_loading=pack)
    untouched = torch.nn.Module()
    untouched.quant_method = SimpleNamespace(
        process_weights_after_loading=lambda module: pytest.fail("unmodified layer")
    )
    manager = SimpleNamespace(
        _select_runners=lambda: [
            ("target", SimpleNamespace(model=torch.nn.Sequential(layer))),
            ("draft", SimpleNamespace(model=torch.nn.Sequential(untouched))),
        ]
    )
    if explicit_postprocess:
        pack(layer)
    finalize(manager)
    finalize(manager)
    assert calls == [layer]
    assert tuple(layer.weight.shape) == (1, 2, 4, 8)
    assert layer.weight.data_ptr() == pointer
    torch.testing.assert_close(layer.weight.flatten(), torch.arange(64.0), rtol=0, atol=0)


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


@pytest.mark.parametrize("initial", [False])
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


@pytest.fixture
def manager(monkeypatch):
    msgspec = pytest.importorskip("msgspec")
    path = sglang_source_path("srt/managers/scheduler_components/weight_updater.py")
    tree = ast.parse(path.read_text())
    sessions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_WeightUpdateSession"]
    if not sessions:
        pytest.skip("Requires SGLang 0.5.21 weight update sessions")
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SchedulerWeightUpdaterManager"
    )
    names = {"begin_weight_update", "end_weight_update", "post_process_weights", "update_weights_from_tensor"}
    nodes = sessions + [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    events = []
    loader = ModuleType("sglang.srt.model_loader.loader")
    loader.post_load_weights = lambda model: events.append((model.role, "post_load"))
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    namespace = {
        "msgspec": msgspec,
        "logger": SimpleNamespace(error=lambda *_: None),
        "torch": SimpleNamespace(distributed=SimpleNamespace(barrier=lambda **_: None)),
        **{
            f"{name}ReqOutput": SimpleNamespace
            for name in ("BeginWeightUpdate", "EndWeightUpdate", "PostProcessWeights", "UpdateWeightsFromTensor")
        },
    }
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)

    runners = []
    for role in ("target", "draft"):
        model = SimpleNamespace(role=role, packed=False, fail=False)

        def restore(model=model):
            model.packed = False
            events.append((model.role, "restore"))

        def pack(model=model):
            assert not model.packed, f"{model.role} weights were packed twice"
            model.packed = True
            events.append((model.role, "pack"))

        def post(
            restore_weights_before_load=False, post_process_quantization=False, model=model, restore=restore, pack=pack
        ):
            if model.fail:
                return False, "injected failure"
            if restore_weights_before_load:
                restore()
            if post_process_quantization:
                pack()
            return True, "Success"

        def end(*, run_post_load, model=model, pack=pack):
            if run_post_load:
                loader.post_load_weights(model)
            pack()

        def load(*, named_tensors, load_format, model=model):
            model.packed = False
            events.append((model.role, "load"))
            return True, "Success"

        runner = SimpleNamespace(
            model=model,
            post_process_weights=post,
            weight_updater=SimpleNamespace(
                begin_weight_update=restore, end_weight_update=end, update_weights_from_tensor=load
            ),
        )
        runners.append((role, runner))

    obj = SimpleNamespace(
        _session=None,
        _select_runners=lambda selector="all": [
            (role, runner) for role, runner in runners if selector in ("all", role)
        ],
        tp_cpu_group=object(),
        tp_worker=SimpleNamespace(deserialize_own_rank=lambda value: value),
        draft_worker=object(),
        _observe_weight_load=lambda _: nullcontext(),
        flush_cache_after_weight_update=lambda _: None,
        record_weight_version_after_update=lambda _: None,
        events=events,
        runners=runners,
    )
    for name in names:
        setattr(obj, name, MethodType(namespace[name], obj))
    return obj


def _post(manager, *, restore=False, quant=True):
    return manager.post_process_weights(
        SimpleNamespace(restore_weights_before_load=restore, post_process_quantization=quant)
    )


def _load(manager):
    return manager.update_weights_from_tensor(
        SimpleNamespace(
            serialized_named_tensors=[],
            selector="all",
            disable_draft_model=None,
            load_format=None,
            weight_version=None,
        )
    )


@pytest.mark.parametrize("protocol", ["mixed"])
def test_weight_update_protocol_packs_each_cycle_once(manager, protocol):
    for _ in range(2):
        if protocol != "native":
            assert _post(manager, restore=True, quant=False).success
        if protocol != "legacy":
            assert manager.begin_weight_update(SimpleNamespace(selector="all")).success
        assert _load(manager).success
        if protocol != "native":
            assert _post(manager).success
        if protocol != "legacy":
            assert manager.end_weight_update(SimpleNamespace()).success
        assert manager._session is None
    assert manager.events.count(("target", "pack")) == 2
    assert manager.events.count(("draft", "pack")) == 2


@pytest.mark.parametrize("mutation", ["bucket"])
def test_new_weights_invalidate_legacy_postprocessing(manager, mutation):
    manager.begin_weight_update(SimpleNamespace(selector="all"))
    _load(manager)
    _post(manager)
    assert manager._session.post_processed
    if mutation == "bucket":
        _load(manager)
    else:
        _post(manager, restore=True, quant=False)
    assert not manager._session.post_processed
    manager.end_weight_update(SimpleNamespace())
    assert manager.events.count(("target", "pack")) == 2


def test_failed_postprocess_does_not_mark_session_complete(manager):
    manager.begin_weight_update(SimpleNamespace(selector="all"))
    manager.runners[0][1].model.fail = True
    assert not _post(manager).success
    assert not manager._session.post_processed
    assert not any(event == "pack" for _, event in manager.events)
