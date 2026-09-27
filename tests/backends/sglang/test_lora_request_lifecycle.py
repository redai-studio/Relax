# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU tests for native LoRA references across cancellation and late
cleanup."""

import asyncio
from types import SimpleNamespace

import pytest
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager


def make_state(obj, lifecycle):
    return ReqState(
        [],
        False,
        asyncio.Event(),
        obj,
        SimpleNamespace(set_finished_time=lambda: None, get_e2e_latency=lambda: 0),
        lifecycle_id=lifecycle,
    )


async def make_manager():
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.enable_lora = True
    manager.config_value = lambda name: None
    manager.server_args = SimpleNamespace(max_loaded_loras=2)
    manager.lora_registry = LoRARegistry([LoRARef(lora_name="A", lora_id="A-id", pinned=True)])
    manager.rid_to_state = {}
    manager.logical_rid_to_child_rids = {}
    manager.child_rid_to_logical_rid = {}
    manager._dispatch_to_scheduler = lambda obj: None
    obj = SimpleNamespace(rid="request", is_single=True, lora_path="A")
    lifecycle = object()
    manager.rid_to_state[obj.rid] = make_state(obj, lifecycle)
    await manager._resolve_lora_path(obj)
    return manager, obj, lifecycle


async def unload_waiter(manager):
    lora_id = await manager.lora_registry.unregister("A")
    waiter = asyncio.create_task(manager.lora_registry.wait_for_unload(lora_id))
    await asyncio.sleep(0)
    return waiter


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_send_fails", [False, True])
async def test_dispatched_request_retains_ref_until_terminal(abort_send_fails):
    manager, obj, lifecycle = await make_manager()
    manager.rid_to_state[obj.rid].dispatched = True
    if abort_send_fails:

        def fail_abort(obj):
            raise RuntimeError("injected abort dispatch failure")

        manager._dispatch_to_scheduler = fail_abort
    waiter = await unload_waiter(manager)
    manager._discard_pending_req_states(obj, {obj.rid: lifecycle})
    await asyncio.sleep(0)
    assert obj.rid in manager.rid_to_state
    assert not waiter.done()
    # Real scheduler abort handler; a duplicate terminal echo must be harmless.
    manager._handle_abort_req(AbortReq(rid=obj.rid))
    manager._handle_abort_req(AbortReq(rid=obj.rid))
    await asyncio.wait_for(waiter, 1)


@pytest.mark.asyncio
async def test_undispatched_request_releases_once():
    manager, obj, lifecycle = await make_manager()
    waiter = await unload_waiter(manager)
    manager._discard_pending_req_states(obj, {obj.rid: lifecycle})
    manager._discard_pending_req_states(obj, {obj.rid: lifecycle})
    assert not manager.rid_to_state
    await asyncio.wait_for(waiter, 1)


@pytest.mark.asyncio
async def test_stale_cleanup_does_not_release_reused_rid():
    manager, obj, lifecycle = await make_manager()
    waiter = await unload_waiter(manager)
    manager._discard_pending_req_states(obj, {obj.rid: object()})
    await asyncio.sleep(0)
    assert manager.rid_to_state[obj.rid].lifecycle_id is lifecycle
    assert not waiter.done()
    manager._remove_req_state(obj.rid, lifecycle)
    await asyncio.wait_for(waiter, 1)


@pytest.mark.asyncio
async def test_parallel_children_hold_one_acquire_until_last_terminal():
    manager, obj, lifecycle = await make_manager()

    def init_child(child, request=None, *, lifecycle_id=None):
        manager.rid_to_state[child.rid] = make_state(child, lifecycle_id)

    manager._init_req_state = init_child
    for rid in ("child1", "child2"):
        manager._init_child_req_state(obj.rid, SimpleNamespace(rid=rid, lora_path="A"))
        manager.rid_to_state[rid].dispatched = True
    manager._remove_req_state(obj.rid, lifecycle)
    waiter = await unload_waiter(manager)
    manager._discard_pending_req_states(obj, {obj.rid: lifecycle})
    manager._remove_req_state("child1", lifecycle)
    await asyncio.sleep(0)
    assert not waiter.done()
    manager._remove_req_state("child2", lifecycle)
    await asyncio.wait_for(waiter, 1)
