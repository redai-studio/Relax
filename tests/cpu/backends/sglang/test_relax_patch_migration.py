# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Behavioral checks for Relax's private patch on the official SGLang
release."""

import ast
import os
from importlib import metadata
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from packaging.version import Version

from tests.backends.sglang.source_utils import sglang_source_path


torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytestmark = pytest.mark.skipif(Version(metadata.version("sglang")) < Version("0.5.21"), reason="SGLang 0.5.21")


def _load(path, class_name, name, **namespace):
    source = sglang_source_path(path)
    tree = ast.parse(source.read_text())
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("kernel_heads", [12])
def test_deepseek_sink_refresh_keeps_existing_views(kernel_heads):
    path = "srt/models/deepseek_v4.py"
    local = _load(path, "MqaAttentionBase", "_local_attn_sink")
    refresh = _load(path, "MqaAttentionBase", "refresh_attn_sink_cache")
    attention = SimpleNamespace(
        attn_tp_size=4,
        attn_tp_rank=1,
        n_local_heads=12,
        n_heads=48,
        attn_sink=torch.arange(48, dtype=torch.float32),
        _attn_sink_local=None,
    )
    view = local(attention, kernel_heads)
    cached = attention._attn_sink_local
    pointer = cached.data_ptr()
    for delta in (100, 200):
        attention.attn_sink.add_(delta)
        refresh(attention)
        assert attention._attn_sink_local is cached and cached.data_ptr() == pointer
        torch.testing.assert_close(view[:12], attention.attn_sink[12:24])
        assert torch.count_nonzero(cached[12:]) == 0


def _deferred_tree(backend, *, enabled=True):
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
    from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.components import (
        FullComponent,
        MambaComponent,
    )
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.runtime_context import get_context

    with get_context().override_server_args(_mamba_cache_chunk_size=256, mamba_max_states_per_path=-1):
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=Mock(spec=HybridReqToTokenPool),
            token_to_kv_pool_allocator=None,
            page_size=64,
            enable_mamba_extra_buffer=True,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
        )
        if backend == "rust":
            core = RustUnifiedTreeCore(params)
        else:
            cache = SimpleNamespace(enable_session_radix_cache=False, token_to_kv_pool_allocator=None)
            components = {
                ComponentType.FULL: FullComponent(cache, params),
                ComponentType.MAMBA: MambaComponent(cache, params),
            }
            core = UnifiedTreeCore(params, components)
            cache.tree_core = core
    if enabled:
        core.set_hicache_enabled()
    core.write_through_threshold = 2
    return core


def _chunked_checkpoint(core, length=256, *, prefix_len=0, tokens=None, mamba_slot=None, value_offset=1000):
    from array import array

    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    key = RadixKey(array("q", range(length) if tokens is None else tokens), extra_key="adapter", cache_salt="tenant")
    step = core.begin_insert(
        InsertParams(
            key=key,
            value=torch.arange(value_offset, value_offset + length, dtype=torch.int64),
            mamba_value=torch.tensor([length if mamba_slot is None else mamba_slot], dtype=torch.int64),
            prev_prefix_len=prefix_len,
            chunked=True,
        )
    )
    actions = list(step.actions)
    while step.result is None:
        step = core.resume_insert()
        actions.extend(step.actions)
    actions.extend(core.end_insert())
    assert actions == []
    return core.match_full_device_prefix(key)[1]


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_real_deferred_checkpoint_respects_namespace_and_cache_salt(backend):
    from array import array

    from sglang.srt.mem_cache.radix_cache import RadixKey

    core = _deferred_tree(backend)
    node = _chunked_checkpoint(core)
    assert not core.is_root(node)
    for extra_key, cache_salt, expected in [
        ("adapter", "tenant", 256),
        ("other-adapter", "tenant", 0),
        ("adapter", "other-tenant", 0),
    ]:
        key = RadixKey(array("q", range(256)), extra_key=extra_key, cache_salt=cache_salt)
        assert core.match_full_device_prefix(key)[0] == expected


@pytest.mark.parametrize("backend,condition", [("rust", "finished")])
def test_real_tree_final_short_tail_uses_backend_neutral_identity_and_clears_once(backend, condition):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams, InsertResult
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV

    core = _deferred_tree(backend)
    core.write_through_threshold = 1
    node = _chunked_checkpoint(core)
    receipt = core.inc_lock_ref(node).to_dec_params()
    root = core.root_node_handle("adapter")
    marker = (root, node, 256)
    if condition == "stale_root":
        core.reset()
        node = _chunked_checkpoint(core)
        receipt = core.inc_lock_ref(node).to_dec_params()
    elif condition == "stale_node":
        marker = (root, root, 256)
    elif condition == "stale_length":
        marker = (root, node, 192)
    request = SimpleNamespace(
        _hicache_deferred_chunked_backup=marker,
        origin_input_ids=list(range(256)),
        output_ids=[1],
        kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=256),
        last_node=node,
        priority=0,
        session_id=None,
        kv_rotation_base=None,
        extra_key="adapter",
        cache_salt="tenant",
        finished_reason=FINISH_ABORT() if condition == "abort" else None if condition == "unfinished" else object(),
    )
    actions = []
    events = []

    def apply_actions(pending):
        assert not events and request.kv.cache_protected_len == 256
        actions.extend(pending)

    cache = SimpleNamespace(
        disable=False,
        cache_controller=object(),
        enable_mamba_extra_buffer=True,
        enable_session_radix_cache=False,
        page_size=64,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(512).reshape(1, 512)),
        tree_core=core,
        _components_tuple=[
            SimpleNamespace(
                component_type="mamba",
                prepare_for_caching_req=lambda **kwargs: 0,
                floor_cache_len=lambda size: size,
                cleanup_after_caching_req=lambda *args, **kwargs: events.append("cleanup"),
            )
        ],
        insert=lambda params: InsertResult(prefix_len=0, last_device_node=core.root_node_handle("adapter")),
        _apply_cache_actions=apply_actions,
    )
    operation = _load(
        "srt/mem_cache/unified_radix_cache.py",
        "UnifiedRadixCache",
        "insert_req",
        RadixKey=RadixKey,
        InsertParams=InsertParams,
        BASE_COMPONENT_TYPE="full",
        torch=torch,
    )
    try:
        operation(cache, request, up_to=257)
        assert request._hicache_deferred_chunked_backup is None
        assert len(actions) == (1 if condition == "finished" else 0)
        if actions:
            assert isinstance(actions[0], BackupKV) and actions[0].node_ids == [node]
        operation(cache, request, up_to=257)
        assert len(actions) == (1 if condition == "finished" else 0)
        assert events == ["cleanup", "cleanup"]
    finally:
        core.dec_lock_ref(node, receipt)


@pytest.mark.parametrize("kind,output_str", [("named", False)])
def test_relax_weight_payloads_use_native_safe_pickle_types(monkeypatch, record_property, kind, output_str):
    from sglang.srt.utils.common import MultiprocessingSerializer, SafeUnpickler
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    named = [("bf16", torch.arange(4, dtype=torch.bfloat16)), ("fp32", torch.arange(4.0)), ("ids", torch.arange(2))]
    if kind == "mixed_bucket":
        bucket = FlattenedTensorBucket(named_tensors=named)
        payload = {"flattened_tensor": bucket.get_flattened_tensor(), "metadata": bucket.get_metadata()}
    elif kind == "lora":
        payload = dict(named)
    elif kind == "empty":
        payload = {"flattened_tensor": torch.empty(0, dtype=torch.uint8), "metadata": []}
    else:
        payload = named
    globals_seen = []
    original = SafeUnpickler.find_class

    def find_class(self, module, name):
        globals_seen.append((module, name))
        return original(self, module, name)

    monkeypatch.setattr(SafeUnpickler, "find_class", find_class)
    strategy = torch.multiprocessing.get_sharing_strategy()
    try:
        if kind == "lora":
            torch.multiprocessing.set_sharing_strategy("file_system")
        restored = MultiprocessingSerializer.deserialize(
            MultiprocessingSerializer.serialize(payload, output_str=output_str)
        )
    finally:
        torch.multiprocessing.set_sharing_strategy(strategy)
    assert not any(module.startswith(("Relax.", "slime.", "relax.")) for module, _ in globals_seen)
    record_property("pickle_globals", globals_seen)
    if kind in ("mixed_bucket", "empty"):
        if kind == "mixed_bucket":
            assert ("sglang.srt.weight_sync.tensor_bucket", "FlattenedTensorMetadata") in globals_seen
            assert all(
                type(item).__module__ == "sglang.srt.weight_sync.tensor_bucket" for item in restored["metadata"]
            )
        restored = FlattenedTensorBucket(**restored).reconstruct_tensors()
    elif kind == "lora":
        restored = list(restored.items())
    expected = [] if kind == "empty" else named
    assert [name for name, _ in restored] == [name for name, _ in expected]
    for (_, actual), (_, original_tensor) in zip(restored, expected, strict=True):
        assert actual.dtype == original_tensor.dtype and actual.shape == original_tensor.shape
        torch.testing.assert_close(actual, original_tensor, rtol=0, atol=0)


def _run_gc_startup(*, skip, elastic=False, warmup_ok=True, wait_weights=False, with_callback=True):
    calls = []
    state = SimpleNamespace(ready=False, tokenizer_manager=SimpleNamespace(server_status="Initial"))

    class RequestException(Exception):
        pass

    def post(url, **kwargs):
        calls.append(("freeze_gc", url, kwargs))
        if not state.ready:
            raise RequestException("HTTP listener is not bound during skipped warmup")
        return SimpleNamespace(raise_for_status=lambda: calls.append("HTTP200"))

    def warmup(server_args):
        calls.append("warmup")
        state.ready = warmup_ok
        if warmup_ok:
            state.tokenizer_manager.server_status = "Up"
        return warmup_ok

    namespace = {
        "get_serving": lambda: SimpleNamespace(skip_server_warmup=skip, admin_api_key="test-admin", api_key=None),
        "get_model": lambda: SimpleNamespace(
            checkpoint_engine_wait_weights_before_ready=wait_weights, delete_ckpt_after_loading=False
        ),
        "get_exec": lambda: SimpleNamespace(moe=SimpleNamespace(is_ep_scale_joiner=elastic, ep_join_mode="joiner")),
        "get_observability": lambda: SimpleNamespace(debug_tensor_dump_input_file=False),
        "_global_state": state,
        "ServerStatus": SimpleNamespace(Up="Up"),
        "_execute_server_warmup": warmup,
        "_wait_weights_ready": lambda: calls.append("wait_weights"),
        "requests": SimpleNamespace(post=post, exceptions=SimpleNamespace(RequestException=RequestException)),
        "ssl_verify_of": lambda _: True,
        "logger": SimpleNamespace(
            warning=lambda *args, **kwargs: calls.append("GC_warning"),
            info=lambda *args, **kwargs: calls.append("ready"),
            debug=lambda *args, **kwargs: None,
        ),
    }
    path = "srt/entrypoints/http_server.py"
    namespace["_freeze_gc_after_server_warmup"] = _load(path, None, "_freeze_gc_after_server_warmup", **namespace)
    operation = _load(path, None, "_wait_and_warmup", **namespace)
    callback = (lambda: calls.append("callback")) if with_callback else None
    operation(SimpleNamespace(url=lambda: "http://server.invalid"), callback, warmup)
    return calls, state.tokenizer_manager.server_status


@pytest.mark.parametrize("elastic", [False])
def test_skip_server_warmup_never_freezes_gc_before_http_bind(elastic):
    assert _run_gc_startup(skip=True, elastic=elastic) == (["ready", "callback"], "Up")


@pytest.fixture
def legacy_load_service():
    fastapi = pytest.importorskip("fastapi")
    if "SGLANG_TEST_SOURCE_ROOT" not in os.environ:
        try:
            version = metadata.version("sglang")
        except metadata.PackageNotFoundError:
            pytest.skip("Requires installed SGLang or SGLANG_TEST_SOURCE_ROOT")
        if Version(version) < Version("0.5.21"):
            pytest.skip("SGLang 0.5.21 migration")
    source = sglang_source_path("srt/entrypoints/http_server.py")
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_load")
    app = fastapi.FastAPI()
    manager = SimpleNamespace(get_loads=AsyncMock(return_value=[]))
    clock = Mock(return_value=123.5)
    namespace = {
        "app": app,
        "_global_state": SimpleNamespace(tokenizer_manager=manager),
        "time": SimpleNamespace(perf_counter=clock),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return app, manager, clock


def _snapshot(rank, running, waiting, total, used):
    return SimpleNamespace(
        dp_rank=rank,
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_total_tokens=total,
        num_used_tokens=used,
    )


def test_legacy_load_preserves_router_array_and_dp_token_accounting(legacy_load_service):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    app, manager, clock = legacy_load_service
    manager.get_loads.return_value = [_snapshot(2, 2, 3, 70, 20), _snapshot(7, 0, 4, 100, 60)]
    with TestClient(app) as client:
        response = client.get("/get_load")
    assert response.status_code == 200
    assert response.json() == [
        {
            "dp_rank": 2,
            "num_reqs": 5,
            "num_waiting_reqs": 3,
            "num_tokens": 70,
            "num_pending_tokens": 50,
            "ts_tic": 123.5,
        },
        {
            "dp_rank": 7,
            "num_reqs": 4,
            "num_waiting_reqs": 4,
            "num_tokens": 100,
            "num_pending_tokens": 40,
            "ts_tic": 123.5,
        },
    ]
    # The pinned router sums this field directly from the top-level JSON array.
    assert sum(item["num_tokens"] for item in response.json()) == 170
    manager.get_loads.assert_awaited_once_with(include=["core"])
    clock.assert_called_once_with()


def test_legacy_load_does_not_report_fake_idle_on_backend_error(legacy_load_service):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    app, manager, _ = legacy_load_service
    manager.get_loads.side_effect = RuntimeError("load snapshots unavailable")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/get_load")
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
