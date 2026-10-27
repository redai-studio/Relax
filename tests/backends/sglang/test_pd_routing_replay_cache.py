# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU execution of the actual PD/R3 cache and retraction boundaries."""

import ast
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest

from tests.backends.sglang.source_utils import sglang_source_path


torch = pytest.importorskip("torch")


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


@pytest.mark.parametrize("host,device", [([0, 1], [3, 7]), ([5, 2], [7, 3])])
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


def test_decode_host_receive_does_not_copy_uninitialized_route_rows():
    cache, source = make_cache()
    cache.backup(source, torch.tensor([1]), torch.tensor([3]))
    before = source.clone()
    cache.restore(source, torch.tensor([0, 1]), torch.tensor([8, 9]))
    torch.testing.assert_close(source[8], before[8])
    torch.testing.assert_close(source[9], before[3])


def test_direct_transfer_reuses_paired_cpu_indices_without_new_copy():
    transfer = load("_routed_experts_transfer_indices", "srt/mem_cache/unified_radix_cache.py", {})
    host, device = torch.tensor([1, 5]), torch.tensor([7, 3])
    actual = transfer(None, host, device)
    assert actual[0] is host and actual[1] is device


@pytest.mark.parametrize("backend", ["cpu_tensor", "host_pool"])
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


@pytest.mark.parametrize("mode,expected", [("prefill", True), ("null", True), ("decode", False)])
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


def test_first_token_stop_collects_before_decode_releases_slots():
    events = []
    req = SimpleNamespace(
        time_stats=SimpleNamespace(set_decode_prebuilt_finish_time=lambda: None, set_quick_finish_time=lambda: None),
        update_finish_state=lambda: None,
        finished=lambda: True,
    )
    namespace = {
        "DisaggregationMode": SimpleNamespace(DECODE="decode"),
        "get_disagg": lambda: SimpleNamespace(disaggregation_decode_enable_radix_cache=False),
        "get_memory": lambda: SimpleNamespace(enable_hisparse=False),
        "release_kv_cache": lambda *_: events.append("release"),
    }
    process = load(
        "process_batch_result_prebuilt", "srt/managers/scheduler_components/batch_result_processor.py", namespace
    )
    process(
        SimpleNamespace(
            disaggregation_mode="decode",
            _maybe_collect_routed_experts=lambda _: events.append("collect"),
            tree_cache=None,
            output_streamer=SimpleNamespace(stream_output=lambda *_: events.append("output")),
        ),
        SimpleNamespace(reqs=[req], return_logprob=False),
    )
    assert events == ["collect", "release", "output"]


def test_bootstrap_at_chunk_boundary_defers_host_backup():
    polls = SimpleNamespace(Failed=0, Bootstrapping=1, WaitingForInput=2)
    namespace = {"KVPoll": polls, "should_force_retry": lambda _: False}
    handle = load("handle_pending_bootstrap", "srt/disaggregation/prefill.py", namespace)
    cache = Mock()
    scheduler = SimpleNamespace(
        disagg_prefill_bootstrap_queue=SimpleNamespace(finalize_bootstrap=lambda _: True),
        cache_unfinished_disagg_prefill=cache,
    )
    req = object()
    assert handle(scheduler, req, polls.WaitingForInput, chunked=True)
    cache.assert_called_once_with(req, chunked=True)


def test_buffer_staging_free_invalidates_routes_before_slot_reuse():
    cache, source = make_cache()
    slots = torch.tensor([0, 1])
    cache.backup(source, slots, torch.tensor([3, 7]))
    freed = []
    staging = SimpleNamespace(
        _cache=SimpleNamespace(
            _host_routed_experts=cache,
            cache_controller=SimpleNamespace(
                mem_pool_host=SimpleNamespace(free=lambda indices: freed.append(not cache.covers(indices)))
            ),
        )
    )
    free = load("_free_staging_now", "srt/mem_cache/buffer_mode/pipeline.py", {})
    free(staging, slots, [])
    assert freed == [True]
    source.fill_(42)
    cache.backup(source, slots, torch.tensor([9, 2]))
    source.fill_(-1)
    cache.restore(source, slots, torch.tensor([5, 6]))
    assert (source[torch.tensor([5, 6])] == 42).all()
