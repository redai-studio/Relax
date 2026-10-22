# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check BF16 MoE hot copies after TRT-LLM's block-layout postprocessing."""

import ast
from importlib import metadata
from types import SimpleNamespace

import pytest
from packaging.version import Version

from tests.backends.sglang.source_utils import sglang_source_path


torch = pytest.importorskip("torch")
try:
    _sglang_version = metadata.version("sglang")
except metadata.PackageNotFoundError:
    pytest.skip("SGLang is not installed", allow_module_level=True)
pytestmark = pytest.mark.skipif(Version(_sglang_version) < Version("0.5.21"), reason="SGLang 0.5.21 BF16 backends")


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


@pytest.mark.parametrize("backend", ["flashinfer_trtllm", "flashinfer_trtllm_routed"])
@pytest.mark.parametrize("weight_name", ["w13_weight", "w2_weight"])
@pytest.mark.parametrize("is_gated", [False, True])
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


def test_bf16_moe_restore_rejects_wrong_capacity() -> None:
    restore, method = _restore("flashinfer_trtllm")
    layer = SimpleNamespace(
        num_local_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=256,
        moe_runner_config=SimpleNamespace(is_gated=True),
    )
    param = torch.nn.Parameter(torch.zeros(64, dtype=torch.bfloat16), requires_grad=False)
    with pytest.raises(RuntimeError, match="Cannot restore.*weight shape"):
        restore(method, layer, param, "model.layers.0.mlp.experts.w13_weight")


def test_bf16_moe_restore_leaves_other_backends_unchanged() -> None:
    restore, method = _restore("triton")
    param = torch.nn.Parameter(torch.zeros(2, 8, 64, dtype=torch.bfloat16), requires_grad=False)
    pointer = param.data_ptr()
    restore(method, SimpleNamespace(), param, "model.layers.0.mlp.experts.w13_weight")
    assert tuple(param.shape) == (2, 8, 64)
    assert param.data_ptr() == pointer


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


@pytest.mark.parametrize("explicit_postprocess", [False, True])
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


@pytest.mark.parametrize("fail", [False, True])
def test_bf16_moe_resume_finalizes_before_unpausing(fail: bool) -> None:
    resume = _lifecycle_method("srt/managers/scheduler.py", "Scheduler", "continue_generation")
    scheduler = SimpleNamespace(_engine_paused=True, disaggregation_mode=None)
    calls = []

    def finalize():
        assert scheduler._engine_paused
        calls.append("finalize")
        if fail:
            raise RuntimeError("repack failed")

    scheduler.weight_updater = SimpleNamespace(finalize_pending_bf16_weights=finalize)
    request = SimpleNamespace(torch_empty_cache=False)
    if fail:
        with pytest.raises(RuntimeError, match="repack failed"):
            resume(scheduler, request)
        assert scheduler._engine_paused
    else:
        resume(scheduler, request)
        assert not scheduler._engine_paused
    assert calls == ["finalize"]
