# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Routed-expert capture, host-cache preservation and PD completion."""

import ast
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib import metadata
from types import MethodType, ModuleType, SimpleNamespace
from typing import NamedTuple

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


def _load_class(path, name, namespace):
    node = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == name
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("mode,dp_rank,overlap", [("decode_ragged", 1, True), ("prefill_after_decode", 0, False)])
def test_graph_routing_reaches_only_real_local_host_cache_rows(monkeypatch, mode, dp_rank, overlap):
    """Run the actual model dispatch, graph-size propagation, slicing and
    finalization.

    Graph launch/model arithmetic are CPU doubles; DP ranks have distinct
    routes and padded cache slot zero is a live sentinel that must not be
    overwritten.
    """
    model_path = sglang_source_path("srt/model_executor/model_runner.py")
    base_path = sglang_source_path("srt/state_capturer/base.py")
    graph = mode not in {"eager", "split"}
    decode_mode = mode.startswith("decode")
    stride = 8 if mode == "decode_ragged" else 12 if decode_mode else 16 if graph else 4
    real_counts = [6, 5] if mode == "decode_ragged" else [9, 6] if decode_mode else [3, 2]
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_cuda_graph=lambda: decode_mode,
            is_decode=lambda: decode_mode,
            is_split_prefill=lambda: mode == "split",
            is_extend=lambda **_: mode.startswith("prefill"),
            is_target_verify=lambda: False,
        ),
        global_num_tokens_cpu=[stride, stride],
        out_cache_loc=torch.tensor(list(range(5, 5 + real_counts[dp_rank])) + [0] * (stride - real_counts[dp_rank])),
        out_cache_loc_virtual=None,
        token_indices_to_pool=None,
        apply_deprecated_skip_attn_backend_init=lambda *_: None,
        post_forward_mlp_sync_batch=lambda *_: None,
        spec_info=SimpleNamespace(num_tokens_per_req=3, num_tokens_for_logprob_per_req=3) if decode_mode else None,
    )
    spec_path = sglang_source_path("srt/speculative/spec_info.py")
    spec_module = ModuleType("sglang.srt.speculative.spec_info")
    spec_module.spec_scale_global_num_tokens = _load_function(spec_path, "spec_scale_global_num_tokens", {})
    monkeypatch.setitem(sys.modules, spec_module.__name__, spec_module)
    init_metadata = _load_function(
        sglang_source_path("srt/model_executor/forward_batch_info.py"),
        "init_mlp_sync_metadata",
        {"torch": torch, "is_pin_memory_available": lambda _: False},
    )
    coordinated = mode in {"decode_coordinated", "decode_ragged"}
    counts = real_counts if coordinated else [3, 2]
    init_metadata(
        batch,
        SimpleNamespace(
            dp_spec_prefill_coordination_applied=coordinated,
            global_num_tokens=counts,
            global_num_tokens_for_logprob=counts,
            can_run_decode_cuda_graph=True,
            can_run_dp_draft_cuda_graph=True,
        ),
        "cpu",
    )
    # The original counts remain request counts for uncoordinated spec decode;
    # capture needs the expanded token counts even after buffer padding.
    assert batch.original_global_num_tokens_cpu == counts
    batch.global_num_tokens_cpu = [stride, stride]
    output_type = _load_class(model_path, "ModelRunnerOutput", {"dataclass": dataclass})
    capture_type = _load_class(base_path, "TopkCaptureOutput", {"dataclasses": SimpleNamespace(dataclass=dataclass)})
    dp_slice = _load_function(
        sglang_source_path("srt/layers/dp_attention.py"), "get_dp_local_slice_cpu", {"dp_slot_in": lambda _: dp_rank}
    )
    local_slice = _load_function(
        sglang_source_path("srt/state_capturer/routed_experts.py"),
        "_get_local_slice",
        {
            "is_dp_attention_enabled": lambda: True,
            "_is_scattered_a2a_backend": lambda: False,
            "get_dp_local_slice_cpu": dp_slice,
        },
    )
    end = _load_function(base_path, "on_forward_end", {"TopkCaptureOutput": capture_type})
    routes = torch.cat([torch.full((stride, 2, 2), value, dtype=torch.int32) for value in (11, 29)])
    capturer = SimpleNamespace(
        topk_size=2,
        device_cache=SimpleNamespace(buffer=routes),
        host_cache=SimpleNamespace(buffer=torch.full((32, 2, 2), -7, dtype=torch.int32)),
    )
    capturer._get_local_slice = MethodType(local_slice, capturer)
    capturer.on_forward_end = MethodType(end, capturer)
    indexer_calls = []
    namespace = {
        "contextlib": SimpleNamespace(nullcontext=nullcontext),
        "has_forward_context": lambda: True,
        "ModelRunnerOutput": output_type,
        "get_global_dwdp_manager": lambda: None,
        "EagerRunner": type("EagerRunner", (), {}),
        "_prefill_cuda_graph_allows_context_parallel": lambda *_: True,
        "device_timer_ctx": lambda *_: nullcontext(),
        "profile_range": lambda *_: nullcontext(),
        "build_step_span_name": lambda *_, **__: "test",
        "get_global_expert_distribution_recorder": lambda: SimpleNamespace(
            with_forward_pass=lambda *_: nullcontext({})
        ),
        "get_global_experts_capturer": lambda: capturer,
        "get_global_indexer_capturer": lambda: SimpleNamespace(on_forward_end=lambda **kw: indexer_calls.append(kw)),
        "get_schedule": lambda: SimpleNamespace(disable_overlap_schedule=not overlap),
        "get_exec": lambda: SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=None)),
        "dumper": SimpleNamespace(may_enable=False),
    }
    # Exercise PrefillCudaGraphRunner.execute itself; only backend launch and
    # padded batch creation are replaced, so stale/missing replay sizes fail.
    prefill = SimpleNamespace(
        _validate_capture_hidden_mode=lambda *_: None,
        backend=SimpleNamespace(replay_session=nullcontext),
        load_batch=lambda *_, **__: SimpleNamespace(input_ids=torch.zeros(stride)),
        raw_num_tokens=real_counts[dp_rank],
        _shape_key=lambda *_: SimpleNamespace(variant_label=None),
        model_runner=None,
        device_module=None,
        enable_cp_bcg_capture=False,
        _uses_eager_prefill_tail=lambda: False,
        _execute_tc_piecewise=lambda *_, **__: "logits",
        _finalize_execute_output=lambda output: output,
        can_run_graph=lambda _: True,
    )
    prefill.execute = MethodType(
        _load_function(
            sglang_source_path("srt/model_executor/runner/prefill_cuda_graph_runner.py"),
            "execute",
            {"maybe_publish_prefill_shared_read_done": lambda *_: None},
        ),
        prefill,
    )
    decode = SimpleNamespace(
        can_run_graph=lambda _: True,
        execute=lambda *_, **__: "logits",
        captured_req_width=3,
        ragged_verify_mode=mode == "decode_ragged",
        _replay_graph_key=SimpleNamespace(size=stride),
    )
    if mode != "prefill_first":
        decode.bs = 4
    runner = SimpleNamespace(
        device="cuda",
        decode_cuda_graph_runner=decode,
        prefill_cuda_graph_runner=prefill,
        hisparse_coordinator=None,
        _prepare_eager_forward_batch=lambda *_: None,
        _maybe_execute_deferred_mamba_cow_and_clear=lambda *_: None,
        forward_split_prefill=lambda *_, **__: "logits",
        _extend_forward_kwargs=lambda *_: {},
        eager_runner=SimpleNamespace(execute=lambda *_, **__: "logits"),
        pp_group=SimpleNamespace(is_last_rank=True),
        device_timer=None,
        forward_pass_id=0,
        msprobe_debugger=None,
        is_draft_worker=False,
        canary_manager=None,
        enable_elastic_ep=False,
        eplb_manager=None,
    )
    runner._forward_raw = MethodType(_load_function(model_path, "_forward_raw", namespace), runner)
    result = _load_function(model_path, "forward", namespace)(runner, batch)
    if overlap:
        result.routed_experts_output.map_device_tensors(lambda tensor: tensor.cpu())
        result.routed_experts_output.finalize()
    expected = torch.full_like(capturer.host_cache.buffer, -7)
    expected[batch.out_cache_loc[: real_counts[dp_rank]]] = 11 if dp_rank == 0 else 29
    torch.testing.assert_close(capturer.host_cache.buffer, expected)
    assert indexer_calls[0]["cuda_graph_batch"] == (stride if graph else None)
    assert result.can_run_graph == graph


@pytest.mark.parametrize("backend,mode", [("mxint4", "enabled"), ("bf16", "enabled")])
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


def load(name, file, namespace):
    path = sglang_source_path(file)
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    )
    node.decorator_list = []
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


def make_cache():
    cls = load("RoutedExpertsHostCache", "srt/state_capturer/routed_experts_host_cache.py", {"torch": torch})
    source = torch.arange(16 * 2 * 3, dtype=torch.int32).reshape(16, 2, 3)
    return cls(8, source), source


@pytest.mark.parametrize("host,device", [([5, 2], [7, 3])])
def test_routes_survive_host_backup_and_different_device_slots(host, device):
    cache, source = make_cache()
    host, device = torch.tensor(host), torch.tensor(device)
    expected = source[device].clone()
    cache.backup(source, host, device)
    source.fill_(-999)  # KV slots have since been reused by another request.
    destination = torch.tensor([9, 2])
    cache.restore(source, host, destination)
    torch.testing.assert_close(source[destination], expected)
    assert cache.covers(host)
    assert not cache.covers(torch.tensor([0, 6]))
    assert (source[0] == -999).all()
    cache.invalidate(host[:1])
    assert not cache.covers(host)
    cache.backup(source, host, destination)
    cache.reset()
    assert not cache.covers(host)


@pytest.mark.parametrize("backend", ["cpu_tensor"])
def test_retraction_restores_routes_after_request_mapping_changes(backend):
    class Backup(NamedTuple):
        routed_experts: object = None

    _, source = make_cache()
    resources = SimpleNamespace(experts_capturer=SimpleNamespace(host_cache=SimpleNamespace(buffer=source)))
    namespace = {
        "get_resources": lambda: resources,
        "dsv41_dspark_needs_rebootstrap": lambda _: False,
        "cast": lambda _, value: value,
    }
    backup = load("backup_kv_cache", "srt/mem_cache/common.py", namespace)
    restore = load("restore_kv_cache", "srt/mem_cache/common.py", namespace)
    req = SimpleNamespace(seqlen=3, kv=SimpleNamespace(req_pool_idx=0, retraction_backup=None))
    pool = SimpleNamespace(req_to_token=torch.tensor([[3, 7, 0]]))
    req.offload_kv_cache = lambda *_: setattr(req.kv, "retraction_backup", Backup())
    req.load_kv_cache = lambda *_: setattr(req.kv, "retraction_backup", None)
    tree = SimpleNamespace(backup_kv_cache=lambda _: Backup(), restore_kv_cache=lambda *_: None)
    expected = source[torch.tensor([3, 7])].clone()
    assert backup(req, tree, pool, None, backend)
    source.fill_(-999)
    pool.req_to_token[0, :2] = torch.tensor([9, 2])
    restore(req, tree, pool, None, backend)
    torch.testing.assert_close(source[torch.tensor([9, 2])], expected)
    assert req.kv.retraction_backup is None


@pytest.mark.parametrize("mode,expected", [("prefill", True), ("decode", False)])
def test_storage_only_prefixes_recompute_when_routes_are_required(mode, expected):
    check = load(
        "_needs_cached_routed_experts",
        "srt/mem_cache/unified_radix_cache.py",
        {
            "get_resources": lambda: SimpleNamespace(experts_capturer=object()),
            "get_disagg": lambda: SimpleNamespace(disaggregation_mode=mode),
        },
    )
    assert check() is expected
