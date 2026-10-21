# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Check fused routing capture wiring without recomputing the selected
experts."""

import ast
import sys
from contextlib import contextmanager, nullcontext
from importlib import metadata
from types import ModuleType, SimpleNamespace

import pytest
from packaging.version import Version

from tests.backends.sglang.source_utils import sglang_source_path


torch = pytest.importorskip("torch")
try:
    _version = metadata.version("sglang")
except metadata.PackageNotFoundError:
    pytest.skip("SGLang is not installed", allow_module_level=True)
pytestmark = pytest.mark.skipif(Version(_version) < Version("0.5.21"), reason="SGLang 0.5.21 fused R3")


def _load_function(path, name, namespace):
    function = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("backend", ["mxint4", "bf16"])
@pytest.mark.parametrize("mode", ["enabled", "disabled", "draft", "kernel_error"])
def test_fused_kernel_routes_reach_native_cache(monkeypatch, backend, mode):
    tokens, topk, layer_id = 3, 2, 1
    cache = SimpleNamespace(buffer=torch.full((tokens, 2, topk), -7, dtype=torch.int32))
    capture = _load_function(sglang_source_path("srt/state_capturer/base.py"), "capture", {"torch": torch})
    # The first capture method is BaseDeviceCache.capture; use the real strided
    # destination copy, including the int16 -> native int32 conversion.
    events = []

    def publish(*, layer_id, topk_indices):
        events.append("capture")
        capture(cache, layer_id, topk_indices)

    capturer = None if mode == "disabled" else SimpleNamespace(capture=publish)
    helper = _load_function(
        sglang_source_path("srt/state_capturer/routed_experts.py"),
        "capture_fused_routed_experts",
        {"torch": torch, "contextmanager": contextmanager, "get_global_experts_capturer": lambda: capturer},
    )

    def install(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    install("sglang.srt.state_capturer.routed_experts", capture_fused_routed_experts=helper)
    install("sglang.srt.layers.moe.token_dispatcher", StandardCombineInput=lambda **kwargs: SimpleNamespace(**kwargs))
    install(
        "sglang.srt.layers.moe.token_dispatcher.standard",
        StandardCombineInput=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    install("sglang.srt.layers.moe.topk", TopKOutputChecker=SimpleNamespace(format_is_bypassed=lambda _: True))
    routing_type = SimpleNamespace(Default=0, DeepSeekV3=2, TopK=1)
    install("sglang.srt.layers.moe.utils", RoutingMethodType=routing_type)
    expected = torch.tensor([[5, 2], [3, 7], [6, 1]], dtype=torch.int16)
    x = torch.randn(tokens, 4, dtype=torch.bfloat16)

    def kernel(**kwargs):
        events.append("kernel")
        output = kwargs["routing_replay_out"]
        if mode in {"disabled", "draft"}:
            assert output is None
        else:
            assert output.shape == (tokens, topk)
            assert output.dtype == torch.int16 and output.is_contiguous()
            output.copy_(expected)
        if mode == "kernel_error":
            raise RuntimeError("kernel failed")
        if "output" in kwargs:
            kwargs["output"].copy_(x + 1)
        return x + 1

    install("flashinfer.fused_moe", trtllm_bf16_moe=kernel)
    config = SimpleNamespace(
        top_k=topk,
        correction_bias=None,
        num_expert_group=1,
        topk_group=1,
        allow_routed_experts_capture=mode != "draft",
        renormalize=True,
    )
    dispatch = SimpleNamespace(hidden_states=x, topk_output=SimpleNamespace(router_logits=x, topk_config=config))
    runner = SimpleNamespace(
        is_gated=True,
        num_local_experts=8,
        num_experts=8,
        layer_id=layer_id,
        routed_scaling_factor=1.0,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        intermediate_size_per_partition=128,
        activation="silu",
        num_fused_shared_experts=0,
        routing_method_type=0,
    )
    namespace = {
        "torch": torch,
        "use_symmetric_memory": lambda *_, **__: nullcontext(),
        "get_parallel": lambda: SimpleNamespace(tp_group=None),
        "is_allocation_symmetric": lambda: False,
        "next_power_of_2": lambda value: value,
        "RoutingMethodType": routing_type,
        "get_activation_type": lambda *_, **__: 0,
        "trtllm_mxint4_block_scale_moe": kernel,
    }
    if backend == "mxint4":
        function = _load_function(
            sglang_source_path(
                "srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxint4_moe.py"
            ),
            "apply_weights",
            namespace,
        )
        layer = SimpleNamespace(
            layer_id=layer_id,
            routing_method_type=0,
            w13_weight_packed=None,
            w13_weight_scale=None,
            w2_weight_packed=None,
            w2_weight_scale=None,
        )

        def run():
            return function(SimpleNamespace(moe_runner_config=runner, moe_ep_rank=0), layer, dispatch)
    else:
        function = _load_function(
            sglang_source_path("srt/layers/moe/moe_runner/flashinfer_trtllm.py"),
            "fused_experts_none_to_flashinfer_trtllm_bf16",
            namespace,
        )
        quant = SimpleNamespace(gemm1_weights=None, gemm2_weights=None, global_num_experts=8, local_expert_offset=0)

        def run():
            return function(dispatch, quant, runner)

    if mode == "kernel_error":
        with pytest.raises(RuntimeError, match="kernel failed"):
            run()
        assert events == ["kernel"]
    else:
        result = run()
        torch.testing.assert_close(result.hidden_states, x + 1)
        assert events == (["kernel", "capture"] if mode == "enabled" else ["kernel"])
    torch.testing.assert_close(cache.buffer[:, 0], torch.full((tokens, topk), -7, dtype=torch.int32))
    torch.testing.assert_close(
        cache.buffer[:, layer_id],
        expected.to(torch.int32) if mode == "enabled" else torch.full((tokens, topk), -7, dtype=torch.int32),
    )
