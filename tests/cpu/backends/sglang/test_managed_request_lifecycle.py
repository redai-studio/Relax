# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the installed tokenizer lifecycle with CPU state and no server."""

import asyncio
import copy
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from packaging.version import Version

from tests.backends.sglang.source_utils import sglang_source_path


@pytest.fixture
async def tokenizer():
    pytest.importorskip("sglang")
    if Version(metadata.version("sglang")) < Version("0.5.21"):
        pytest.skip("Requires SGLang 0.5.21")
    sglang_source_path("srt/lora/relax_control.py")
    from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
    from sglang.srt.managers import tokenizer_manager
    from sglang.srt.managers.io_struct import AbortReq, GenerateReqInput
    from sglang.srt.runtime_context import get_context
    from sglang.srt.utils.aio_rwlock import RWLock

    assert (
        Path(tokenizer_manager.__file__).resolve() == sglang_source_path("srt/managers/tokenizer_manager.py").resolve()
    )
    manager = object.__new__(tokenizer_manager.TokenizerManager)
    manager.rid_to_state = {}
    manager._lora_draining_requests = {}
    manager._lora_release_tasks = set()
    manager.logical_rid_to_child_rids = {}
    manager.child_rid_to_logical_rid = {}
    manager.encoder_dispatch_ready = {}
    manager.enable_trace = manager.enable_metrics = manager.enable_priority_scheduling = False
    manager.disaggregation_mode = "null"
    manager.enable_lora = True
    manager.lora_registry = LoRARegistry()
    manager.lora_ref_cache = {}
    manager.tokenizer = None
    manager.is_pause = False
    manager.is_pause_cond = asyncio.Condition()
    manager.model_update_lock = RWLock()
    manager.request_logger = SimpleNamespace(log_received_request=Mock())
    manager.auto_create_handle_loop = Mock()
    manager._dispatch_to_scheduler = Mock()
    manager.config_value = lambda _: "0"
    manager.incremental_streaming_output = False
    with get_context().override_server_args(
        tokenizer_worker_num=1, max_loaded_loras=None, language_only=False, enable_strict_thinking=False
    ):
        await manager.lora_registry.register(LoRARef(lora_name="adapter", lora_path="__tensor__", pinned=True))
        yield manager, GenerateReqInput, AbortReq
        await asyncio.gather(*manager._lora_release_tasks)


def _start(manager, request_type, rid, n=1):
    request = request_type(input_ids=[1, 2], rid=rid, sampling_params={"n": n}, lora_path="adapter")
    request.normalize_batch_and_arguments()
    manager._init_req_state(request)
    return request


async def test_retired_parent_abort_reaches_children_and_keeps_lease_until_terminal(tokenizer):
    manager, request_type, abort_type = tokenizer
    parent = _start(manager, request_type, "parent", n=2)
    await manager._validate_and_resolve_lora(parent)
    logical = manager._logical_rids(parent)[0]
    parent_state = manager.rid_to_state[logical]
    children = []
    for _ in range(2):
        child = copy.copy(parent[0])
        child.regenerate_rid()
        manager._init_child_req_state(logical, child)
        manager.rid_to_state[child.rid].dispatched = True
        children.append(child.rid)
    manager._remove_req_state(logical)
    manager.abort_request(logical)
    manager.abort_request(logical)
    assert {call.args[0].rid for call in manager._dispatch_to_scheduler.call_args_list} == set(children)
    assert manager._dispatch_to_scheduler.call_count == 2
    assert manager.lora_registry.request_count(parent_state.obj.lora_id) == 1
    manager._handle_abort_req(abort_type(rid=children[0]))
    assert manager.lora_registry.request_count(parent_state.obj.lora_id) == 1
    manager._handle_abort_req(abort_type(rid=children[1]))
    manager._handle_abort_req(abort_type(rid=children[1]))
    await asyncio.gather(*manager._lora_release_tasks)
    assert manager.lora_registry.request_count(parent_state.obj.lora_id) == 0
    assert not manager.rid_to_state and not manager._lora_draining_requests
    assert not manager.logical_rid_to_child_rids and not manager.child_rid_to_logical_rid


async def test_batch_id_collision_does_not_leave_partial_request_state(tokenizer):
    manager, request_type, _ = tokenizer
    _start(manager, request_type, "existing")
    batch = request_type(input_ids=[[1], [2]], rid=["new", "existing"])
    batch.normalize_batch_and_arguments()
    with pytest.raises(ValueError, match="Duplicate request ID"):
        manager._init_req_state(batch)
    assert set(manager.rid_to_state) == {"existing"}


async def test_late_abort_waiter_does_not_remove_reused_request_id(tokenizer):
    manager, request_type, _ = tokenizer
    old = _start(manager, request_type, "reused")
    old_state = manager.rid_to_state[old.rid]
    old_state.finished = True
    manager._remove_req_state(old.rid)
    _start(manager, request_type, "reused")
    replacement = manager.rid_to_state[old.rid]
    output = {"meta_info": {"finish_reason": {"type": "abort", "status_code": 503, "message": "old failure"}}}
    assert await manager._handle_abort_finish_reason(output, old_state, is_stream=True) == output
    assert manager.rid_to_state.get(old.rid) is replacement


async def test_late_handler_failure_preserves_reused_id_and_encoder_waiter(tokenizer):
    manager, request_type, _ = tokenizer
    returned, fail = asyncio.Event(), asyncio.Event()

    async def response(request, _http):
        state = manager.rid_to_state[request.rid]
        state.finished = True
        manager._remove_req_state(request.rid)
        returned.set()
        await fail.wait()
        raise ValueError("late handler failure")
        yield  # This is the response-stream boundary, after scheduler completion.

    manager._tokenize_one_request = AsyncMock(return_value=SimpleNamespace())
    manager._send_one_request = AsyncMock()
    manager._wait_one_response = response
    request = request_type(input_ids=[1], rid="reused", lora_path="adapter")
    task = asyncio.create_task(manager.generate_request(request).__anext__())
    try:
        await asyncio.wait_for(returned.wait(), 2)
        _start(manager, request_type, "reused")
        replacement = manager.rid_to_state["reused"]
        encoder_waiter = asyncio.Event()
        manager.encoder_dispatch_ready["reused"] = encoder_waiter
        fail.set()
        with pytest.raises(ValueError, match="late handler failure"):
            await asyncio.wait_for(task, 2)
        assert manager.rid_to_state.get("reused") is replacement
        assert manager.encoder_dispatch_ready.get("reused") is encoder_waiter
        assert not encoder_waiter.is_set()
    finally:
        fail.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_cancel_while_paused_wakes_handler_without_resume(tokenizer):
    from sglang.srt.managers.tokenizer_manager import RequestAbortedError

    manager, request_type, _ = tokenizer
    manager.is_pause = True
    manager._tokenize_one_request = AsyncMock()
    request = request_type(input_ids=[1], rid="paused", lora_path="adapter")
    task = asyncio.create_task(manager.generate_request(request).__anext__())
    try:
        await asyncio.sleep(0)
        assert request.rid in manager.rid_to_state
        manager.abort_request(request.rid)
        with pytest.raises(RequestAbortedError):
            await asyncio.wait_for(task, 2)
        assert manager.is_pause
        manager._tokenize_one_request.assert_not_awaited()
        assert not manager.rid_to_state
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_parent_abort_during_child_mm_preparation_prevents_dispatch(tokenizer):
    from sglang.srt.managers.tokenizer_manager import RequestAbortedError

    manager, request_type, _ = tokenizer
    parent = _start(manager, request_type, "parent")
    await manager._validate_and_resolve_lora(parent)
    child = copy.copy(parent)
    child.regenerate_rid(prefix=parent.rid)
    assert child.rid.startswith(parent.rid + "_")
    manager._init_child_req_state(parent.rid, child)
    prepared, resume = asyncio.Event(), asyncio.Event()

    async def prepare(_):
        prepared.set()
        await resume.wait()
        return []

    manager.cuda_vmm_feature_transport = SimpleNamespace(
        prepare_for_dispatch_async=prepare, cancel_for_dispatch=Mock()
    )
    task = asyncio.create_task(manager._send_one_request(SimpleNamespace(rid=child.rid, mm_inputs=None)))
    try:
        await asyncio.wait_for(prepared.wait(), 2)
        manager.abort_request(parent.rid)
        resume.set()
        with pytest.raises(RequestAbortedError):
            await asyncio.wait_for(task, 2)
        assert not manager.rid_to_state[child.rid].dispatched
        assert all(isinstance(call.args[0], tokenizer[2]) for call in manager._dispatch_to_scheduler.call_args_list)
        manager.cuda_vmm_feature_transport.cancel_for_dispatch.assert_called_once_with([])
        manager._release_req_states_on_failure({parent.rid, child.rid})
        await asyncio.gather(*manager._lora_release_tasks)
        assert manager.lora_registry.request_count(parent.lora_id) == 0
    finally:
        resume.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_delayed_lora_resolution_does_not_attach_to_reused_id(tokenizer):
    from sglang.srt.managers.tokenizer_manager import RequestAbortedError

    manager, request_type, _ = tokenizer
    old = _start(manager, request_type, "reused")
    task = None
    try:
        async with manager.lora_registry._registry_lock.writer_lock:
            task = asyncio.create_task(manager._validate_and_resolve_lora(old))
            await asyncio.sleep(0)
            assert not task.done()
            manager._remove_req_state(old.rid)
            _start(manager, request_type, old.rid)
            replacement = manager.rid_to_state[old.rid]
        with pytest.raises(RequestAbortedError):
            await asyncio.wait_for(task, 2)
        await asyncio.gather(*manager._lora_release_tasks)
        assert manager.rid_to_state[old.rid] is replacement
        assert replacement.lora_lease is None
        assert manager.lora_registry.request_count(old.lora_id) == 0
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_child_abort_isolated_and_failed_send_can_retry(tokenizer):
    manager, request_type, abort_type = tokenizer
    parent = _start(manager, request_type, "parent")
    children = []
    for _ in range(2):
        child = copy.copy(parent)
        child.regenerate_rid(prefix=parent.rid)
        manager._init_child_req_state(parent.rid, child)
        manager.rid_to_state[child.rid].dispatched = True
        children.append(child.rid)
    manager._dispatch_to_scheduler.side_effect = RuntimeError("transport failed")
    with pytest.raises(RuntimeError, match="transport failed"):
        manager.abort_request(children[0])
    manager._dispatch_to_scheduler.reset_mock(side_effect=True)
    manager.abort_request(children[0])
    manager.abort_request(children[0])
    assert manager._dispatch_to_scheduler.call_count == 1
    assert manager._dispatch_to_scheduler.call_args.args[0].rid == children[0]
    assert not manager.rid_to_state[children[1]].abort_sent
    manager._remove_req_state(parent.rid)
    manager._remove_req_state(children[0])
    with pytest.raises(ValueError, match="Duplicate request ID"):
        _start(manager, request_type, parent.rid)
    manager._handle_abort_req(abort_type(rid=children[0]))
    manager._handle_abort_req(abort_type(rid=children[1]))
    _start(manager, request_type, parent.rid)
