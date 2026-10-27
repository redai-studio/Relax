# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Behavioral checks for Relax's private patch on the official SGLang
release."""

import ast
from contextlib import contextmanager, nullcontext
from importlib import metadata
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize("kernel_heads", [12, 64])
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


@pytest.mark.parametrize("tp_size", [1, 4])
def test_deepseek_sink_refresh_leaves_native_lazy_allocation(tp_size):
    path = "srt/models/deepseek_v4.py"
    attention = SimpleNamespace(
        attn_tp_size=tp_size,
        attn_tp_rank=0,
        n_local_heads=12,
        n_heads=48,
        attn_sink=torch.arange(48, dtype=torch.float32),
        _attn_sink_local=None,
    )
    _load(path, "MqaAttentionBase", "refresh_attn_sink_cache")(attention)
    assert attention._attn_sink_local is None
    latest = _load(path, "MqaAttentionBase", "_local_attn_sink")(attention, 12)
    torch.testing.assert_close(latest[:12], attention.attn_sink[:12])


def test_deepseek_post_load_refreshes_every_local_layer():
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(compress_ratio=0, refresh_attn_sink_cache=Mock()),
            refresh_mhc_norm_weight_cache=Mock(),
        )
        for _ in range(2)
    ]
    model = SimpleNamespace(wo_a_fp8=False, model=SimpleNamespace(start_layer=0, end_layer=2, layers=layers))
    _load("srt/models/deepseek_v4.py", "DeepseekV4ForCausalLM", "post_load_weights")(model)
    for layer in layers:
        layer.refresh_mhc_norm_weight_cache.assert_called_once_with()
        layer.self_attn.refresh_attn_sink_cache.assert_called_once_with()


@pytest.mark.parametrize(
    "kind,has_draft,selector,legacy_flag,session_selector,expected",
    [
        ("tensor", False, None, None, None, "target"),
        ("tensor", True, None, None, None, "draft"),
        ("tensor", True, None, False, None, "draft"),
        ("tensor", True, None, True, None, "target"),
        ("tensor", True, "target", False, None, "target"),
        ("tensor", True, "draft", True, None, "draft"),
        ("tensor", True, "all", True, None, "all"),
        ("tensor", True, None, None, "all", "all"),
        ("tensor", True, None, False, "all", "draft"),
        ("tensor", True, None, True, "all", "target"),
        ("distributed", False, None, None, None, "target"),
        ("distributed", True, None, None, None, "target"),
        ("distributed", True, None, None, "all", "all"),
        ("distributed", True, "draft", None, None, "draft"),
        ("distributed", True, "all", None, "target", "all"),
    ],
)
def test_weight_update_preserves_legacy_routing(
    monkeypatch, kind, has_draft, selector, legacy_flag, session_selector, expected
):
    from sglang.srt.managers import io_struct

    calls = []
    selected = []
    runners = {}
    for role in ("target", "draft"):
        updater = SimpleNamespace(
            receive_weights_from_distributed=lambda **kwargs: [("weight", torch.ones(1))],
            load_weights_from_distributed=lambda weights, role=role: (calls.append(role) or True, "ok"),
            update_weights_from_tensor=lambda role=role, **kwargs: (calls.append(role) or True, "ok"),
        )
        runners[role] = SimpleNamespace(weight_updater=updater)

    def select(value):
        selected.append(value)
        return [(role, runners[role]) for role in ("target", "draft") if value == "all" or value == role]

    session = None if session_selector is None else SimpleNamespace(selector=session_selector, loaded_weights=False)
    manager = SimpleNamespace(
        tp_worker=SimpleNamespace(model_runner=runners["target"], deserialize_own_rank=lambda blobs: []),
        draft_worker=object() if has_draft else None,
        _session=session,
        tp_cpu_group=None,
        _observe_weight_load=lambda label: nullcontext(),
        _select_runners=select,
        flush_cache_after_weight_update=Mock(),
        record_weight_version_after_update=Mock(),
    )

    def replace(obj, **kwargs):
        return SimpleNamespace(**(vars(obj) | kwargs))

    operation = _load(
        "srt/managers/scheduler_components/weight_updater.py",
        "SchedulerWeightUpdaterManager",
        "update_weights_from_" + kind,
        torch=torch,
        logger=Mock(),
        msgspec=SimpleNamespace(structs=SimpleNamespace(replace=replace)),
        UpdateWeightsFromTensorReqOutput=io_struct.UpdateWeightsFromTensorReqOutput,
        UpdateWeightsFromDistributedReqOutput=io_struct.UpdateWeightsFromDistributedReqOutput,
    )
    monkeypatch.setattr(torch.distributed, "barrier", Mock())
    kwargs = {} if selector is None else {"selector": selector}
    if kind == "tensor":
        request = io_struct.UpdateWeightsFromTensorReqInput(serialized_named_tensors=[b"bucket"], **kwargs)
        if legacy_flag is not None:
            request.disable_draft_model = legacy_flag
    else:
        request = io_struct.UpdateWeightsFromDistributedReqInput(names=[], dtypes=[], shapes=[], **kwargs)
    assert operation(manager, request).success
    assert selected == [expected]
    assert calls == (["target", "draft"] if expected == "all" else [expected])
    if session is not None:
        assert manager._session.loaded_weights


@pytest.mark.parametrize("layerwise", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("merge", [False, True])
def test_tensor_lora_uses_native_offload_and_invalidates_conditioning(layerwise, initialized, merge):
    path = "multimodal_gen/runtime/pipelines_core/lora/pipeline.py"
    events = []

    @contextmanager
    def materialize(**kwargs):
        assert not layerwise
        events.append("materialize")
        yield

    def convert(**kwargs):
        assert kwargs == {"snapshot_base": not layerwise}
        events.append("convert")

    def apply(*args, **kwargs):
        events.append("apply")
        return 3

    pipeline = SimpleNamespace(
        VALID_TARGETS={"transformer"},
        lora_initialized=initialized,
        _invalidate_lora_conditioning=lambda: events.append("invalidate"),
        _has_layerwise_offload=lambda *args: layerwise,
        _temporarily_disable_offload=materialize,
        convert_to_lora_layers=convert,
        _ingest_lora_state_dict=lambda *args: events.append("ingest"),
        _get_target_lora_layers=lambda target: ([("transformer", {"linear": object()})], None),
        _resolve_lora_merge_mode=lambda *args: "merge" if merge else "unmerged",
        _should_merge_lora_for_layers=lambda *args: merge,
        _needs_lora_weight_update_context=lambda *args: merge,
        _apply_lora_to_layers=apply,
        loaded_adapter_paths={},
        cur_adapter_name={},
        cur_adapter_path={},
        is_lora_merged={},
        cur_adapter_strength={},
        cur_adapter_config={},
    )
    pipeline._weight_update_context = MethodType(
        _load(path, "LoRAPipeline", "_weight_update_context", nullcontext=nullcontext), pipeline
    )
    operation = _load(
        path,
        "LoRAPipeline",
        "set_lora_from_tensors",
        logger=Mock(),
        nullcontext=nullcontext,
        normalize_lora_state_dict=lambda state, **kwargs: state,
        dist=SimpleNamespace(get_rank=lambda: 0),
    )
    assert operation(pipeline, "adapter", {"lora_A": torch.ones(1)}) == 3
    assert events[0] == "invalidate" and events[-1] == "apply"
    assert ("convert" in events) == (not initialized)
    assert ("materialize" in events) == (merge and not layerwise)
    assert pipeline.cur_adapter_name == {"transformer": "adapter"}
    assert pipeline.is_lora_merged == {"transformer": merge}


@pytest.mark.parametrize("heads", [8, 16, 32])
def test_dsa_head_gates_keep_original_repeat_order(heads):
    operation = _load("srt/layers/attention/dsa/utils.py", None, "match_head_gate_q_scale")
    weights = torch.arange(heads, dtype=torch.float32).reshape(1, heads)
    actual = operation(weights, torch.ones(1, 32, 1))
    expected = torch.tensor([[head for head in range(heads) for _ in range(32 // heads)]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("idle", [False, True])
def test_paused_scheduler_keeps_polling_hicache(idle):
    poll = Mock()
    scheduler = SimpleNamespace(
        enable_hierarchical_cache=True,
        tree_cache=SimpleNamespace(check_hicache_events=poll),
        is_fully_idle=lambda: idle,
        metrics_reporter=Mock(),
    )
    operation = _load("srt/managers/scheduler.py", "Scheduler", "_record_scheduler_state_for_paused_engine")
    operation(scheduler)
    poll.assert_called_once_with()
    expected = (
        scheduler.metrics_reporter.record_scheduler_idle
        if idle
        else scheduler.metrics_reporter.record_scheduler_active
    )
    expected.assert_called_once_with()


@pytest.mark.parametrize(
    "condition", ["ready", "disabled", "missing", "root", "backed", "pending", "unlocked", "no_mamba", "write_back"]
)
def test_deferred_hicache_backup_requires_protected_checkpoint(condition):
    path = "srt/mem_cache/unified_cache/unified_tree_core.py"
    component = SimpleNamespace(FULL="full", MAMBA="mamba")
    root = object()
    node = SimpleNamespace(
        evicted=False,
        backuped=False,
        write_through_pending_id=None,
        hit_count=0,
        component_data={"full": SimpleNamespace(lock_ref=1), "mamba": SimpleNamespace(value=object())},
    )
    if condition == "backed":
        node.backuped = True
    elif condition == "pending":
        node.write_through_pending_id = 1
    elif condition == "unlocked":
        node.component_data["full"].lock_ref = 0
    elif condition == "no_mamba":
        node.component_data["mamba"].value = None
    arena = {7: node, 0: root}
    core = SimpleNamespace(
        root_node=root,
        _node_arena=arena,
        enable_hicache=condition != "disabled",
        is_write_back=condition == "write_back",
        enable_external_cache_linker=False,
        write_through_threshold=1,
    )
    core._inc_hit_count_and_check = MethodType(_load(path, "UnifiedTreeCore", "_inc_hit_count_and_check"), core)
    core._build_backup_kv_action = Mock(return_value="backup")
    operation = _load(path, "UnifiedTreeCore", "finish_deferred_chunked_backup", ComponentType=component)
    node_id = 0 if condition == "root" else 99 if condition == "missing" else 7
    assert operation(core, node_id) == (["backup"] if condition == "ready" else [])
    assert core._build_backup_kv_action.call_count == (1 if condition == "ready" else 0)


@pytest.mark.parametrize(
    "condition",
    [
        "finished",
        "abort",
        "unfinished",
        "stale_root",
        "stale_node",
        "stale_length",
        "no_controller",
        "no_extra_buffer",
        "new_checkpoint",
    ],
)
def test_final_short_tail_finishes_backup_before_component_cleanup(condition):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    class Key(list):
        def __init__(self, tokens, *args, **kwargs):
            super().__init__(tokens)

        def page_aligned(self, page_size):
            return self

    events = []
    marker = {"stale_root": (1, 7, 4), "stale_node": (0, 8, 4), "stale_length": (0, 7, 3)}.get(condition, (0, 7, 4))
    reason = FINISH_ABORT() if condition == "abort" else None if condition == "unfinished" else object()
    request = SimpleNamespace(
        _hicache_deferred_chunked_backup=marker,
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[5],
        kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=4),
        last_node=7,
        priority=0,
        session_id=None,
        kv_rotation_base=None,
        extra_key=None,
        cache_salt=None,
        finished_reason=reason,
    )

    def finish(node_id):
        assert node_id == 7 and request.kv.cache_protected_len == 4
        assert "cleanup" not in events
        return ["backup"]

    def insert(params):
        events.append("insert")
        return SimpleNamespace(
            last_device_node=8 if condition == "new_checkpoint" else 0, rotation_tail_declined=False
        )

    cache = SimpleNamespace(
        disable=False,
        cache_controller=None if condition == "no_controller" else object(),
        enable_mamba_extra_buffer=condition != "no_extra_buffer",
        enable_session_radix_cache=False,
        page_size=1,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.arange(10).reshape(1, 10)),
        tree_core=SimpleNamespace(
            root_node_handle=lambda extra_key: 0,
            is_root=lambda node_id: node_id == 0,
            is_eagle=False,
            finish_deferred_chunked_backup=finish,
        ),
        _components_tuple=[
            SimpleNamespace(
                component_type="mamba",
                prepare_for_caching_req=lambda **kwargs: 0,
                floor_cache_len=lambda size: size,
                cleanup_after_caching_req=lambda *args, **kwargs: events.append("cleanup"),
            )
        ],
        insert=insert,
        _apply_cache_actions=lambda actions: events.extend(actions),
    )
    operation = _load(
        "srt/mem_cache/unified_radix_cache.py",
        "UnifiedRadixCache",
        "insert_req",
        RadixKey=Key,
        InsertParams=SimpleNamespace,
        BASE_COMPONENT_TYPE="full",
        torch=torch,
    )
    operation(cache, request, up_to=5)
    assert request._hicache_deferred_chunked_backup is None
    assert events == (["insert", "backup", "cleanup"] if condition == "finished" else ["insert", "cleanup"])
    operation(cache, request, up_to=5)
    assert events.count("backup") == (1 if condition == "finished" else 0)


@pytest.mark.parametrize("inserted", [False, True])
def test_hicache_release_clears_marker_even_if_insert_was_skipped(inserted):
    cleanup = Mock()
    request = SimpleNamespace(_hicache_deferred_chunked_backup=(0, 7, 4))
    cache = SimpleNamespace(_components_tuple=[SimpleNamespace(cleanup_after_caching_req=cleanup)])
    operation = _load("srt/mem_cache/unified_radix_cache.py", "UnifiedRadixCache", "on_release")
    operation(cache, request, inserted=inserted)
    assert request._hicache_deferred_chunked_backup is None
    assert cleanup.call_count == (0 if inserted else 1)


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


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_real_deferred_backup_keeps_threshold_and_ancestor_hit_counts(backend):
    from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV

    core = _deferred_tree(backend)
    parent = _chunked_checkpoint(core)
    leaf = _chunked_checkpoint(core, 512, prefix_len=256)
    receipt = core.inc_lock_ref(leaf).to_dec_params()
    try:
        assert core.finish_deferred_chunked_backup(leaf) == []
        # Finishing the child must not add a hit to its ancestor.
        assert core.finish_deferred_chunked_backup(parent) == []
        actions = core.finish_deferred_chunked_backup(leaf)
        assert len(actions) == 1 and isinstance(actions[0], BackupKV)
        assert actions[0].node_ids == [parent, leaf]
        core.mark_write_through_pending(actions[0].node_ids, leaf)
        assert core.finish_deferred_chunked_backup(leaf) == []
    finally:
        core.dec_lock_ref(leaf, receipt)


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("condition", ["disabled", "root", "stale", "unlocked", "no_mamba", "backed", "write_back"])
def test_real_deferred_backup_rejects_unprotected_or_invalid_checkpoints(backend, condition):
    core = _deferred_tree(backend, enabled=condition != "disabled")
    core.write_through_threshold = 1
    node = _chunked_checkpoint(core)
    if condition == "no_mamba":
        from array import array

        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        # A real aligned split has FULL KV but no Mamba checkpoint of its own.
        _chunked_checkpoint(
            core,
            prefix_len=128,
            tokens=list(range(128)) + list(range(10000, 10128)),
            mamba_slot=257,
            value_offset=2000,
        )
        common = RadixKey(array("q", range(128)), extra_key="adapter", cache_salt="tenant")
        node = core.match_full_device_prefix(common)[1]
        assert not core.is_root(node)
        assert core.get_component_device_value(node, ComponentType.MAMBA) is None
    receipt = core.inc_lock_ref(node).to_dec_params()
    if condition == "root":
        target = core.root_node_handle("adapter")
    else:
        target = node
    if condition == "backed":
        core.commit_backup(node, torch.arange(2000, 2256, dtype=torch.int64), {})
    elif condition == "write_back":
        core.is_write_back = True
    elif condition == "unlocked":
        core.dec_lock_ref(node, receipt)
    elif condition == "stale":
        previous_root = core.root_node_handle("adapter")
        core.reset()
        assert core.root_node_handle("adapter") != previous_root
    try:
        assert core.finish_deferred_chunked_backup(target) == []
    finally:
        if condition not in ("stale", "unlocked"):
            core.dec_lock_ref(node, receipt)


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("condition", ["finished", "abort", "unfinished", "stale_root", "stale_node", "stale_length"])
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


def test_glm_image_epd_accepts_native_receiver_metadata():
    embedding = torch.ones(2, 4)
    processor = SimpleNamespace(
        build_input_ids=lambda prompt, **kwargs: ([1, 99, 2], [(1, 2)], None),
        hf_config=object(),
        IM_START_TOKEN_ID=1,
        IM_END_TOKEN_ID=2,
        IM_TOKEN_ID=99,
    )
    rope = SimpleNamespace(get_rope_index_glm4v=lambda **kwargs: (torch.zeros(3, 1, 3), torch.zeros(1)))
    operation = _load(
        "srt/multimodal/processors/glm4v.py",
        "Glm4vImageProcessor",
        "get_mm_data",
        Modality=SimpleNamespace(IMAGE="image"),
        MultimodalDataItem=SimpleNamespace,
        MultimodalProcessorOutput=SimpleNamespace,
        MRotaryEmbedding=rope,
        torch=torch,
    )
    result = operation(
        processor,
        "prompt",
        {"image": embedding},
        torch.tensor([[1, 2, 2]]),
        video_grid_thw=None,
        audio_feature_lens=None,
    )
    assert result.input_ids == [1, 99, 2]
    assert result.mm_items[0].precomputed_embeddings is embedding
    assert result.mrope_positions.shape == (3, 3)


@pytest.mark.parametrize("kind", ["named", "lora", "mixed_bucket", "empty"])
@pytest.mark.parametrize("output_str", [False, True])
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


@pytest.mark.parametrize("elastic", [False, True])
def test_skip_server_warmup_never_freezes_gc_before_http_bind(elastic):
    assert _run_gc_startup(skip=True, elastic=elastic) == (["ready", "callback"], "Up")


def test_normal_server_warmup_freezes_gc_before_launch_callback():
    calls, status = _run_gc_startup(skip=False)
    assert calls == [
        "warmup",
        (
            "freeze_gc",
            "http://server.invalid/freeze_gc",
            {"headers": {"Authorization": "Bearer test-admin"}, "timeout": 10, "verify": True},
        ),
        "HTTP200",
        "ready",
        "callback",
    ]
    assert status == "Up"


def test_failed_server_warmup_never_freezes_gc_or_announces_ready():
    assert _run_gc_startup(skip=False, warmup_ok=False) == (["warmup"], "Initial")


def test_elastic_joiner_without_skip_keeps_upstream_freeze_attempt():
    calls, status = _run_gc_startup(skip=False, elastic=True)
    assert [call[0] if isinstance(call, tuple) else call for call in calls] == [
        "freeze_gc",
        "GC_warning",
        "ready",
        "callback",
    ]
    assert status == "Up"


def test_skipped_warmup_waits_for_checkpoint_weights_before_ready():
    assert _run_gc_startup(skip=True, wait_weights=True) == (["wait_weights", "ready", "callback"], "Up")


def test_skipped_warmup_does_not_require_launch_callback():
    assert _run_gc_startup(skip=True, with_callback=False) == (["ready"], "Up")
