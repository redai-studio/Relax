# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""One-GPU regressions for actual CUDA IPC producer lease ownership."""

import asyncio
from typing import Any

import pytest

from .multimodal_test_utils import make_request, recycle


pytestmark = pytest.mark.usefixtures("sglang_cuda")


@pytest.mark.parametrize("batch", [False, True])
def test_failed_dispatch_returns_real_cuda_ipc_lease(manager_pool: tuple[Any, Any], batch: bool) -> None:
    manager, pool = manager_pool
    req, _, proxy = make_request(manager, pool)
    assert pool.active_lease_count == 1

    def fail_send(_: Any) -> None:
        raise ConnectionError("scheduler transport failed")

    manager._dispatch_to_scheduler = fail_send
    call = manager._send_batch_request([req]) if batch else manager._send_one_request(req)
    with pytest.raises(ConnectionError, match="scheduler transport failed"):
        asyncio.run(call)
    recycle(pool)
    assert pool.active_lease_count == 0, "undispatched IPC slice leaked"


def test_cancel_before_dispatch_returns_pending_ipc_lease(manager_pool: tuple[Any, Any]) -> None:
    manager, pool = manager_pool
    req, state, proxy = make_request(manager, pool)
    state.pending_ipc_proxies = (proxy,)
    manager._release_req_states_on_failure([req.rid])
    recycle(pool)
    assert req.rid not in manager.rid_to_state
    assert pool.active_lease_count == 0, "cancelled tokenization retained its producer lease"


def test_parallel_copy_does_not_cancel_dispatched_ipc_lease(manager_pool: tuple[Any, Any]) -> None:
    manager, pool = manager_pool
    req, _, proxy = make_request(manager, pool, "first")
    manager._dispatch_to_scheduler = lambda _: None
    asyncio.run(manager._send_one_request(req))
    clone, _, _ = make_request(manager, pool, "second", proxy=proxy)
    manager._dispatch_to_scheduler = lambda _: (_ for _ in ()).throw(ConnectionError("later clone failed"))
    with pytest.raises(ConnectionError):
        asyncio.run(manager._send_one_request(clone))
    recycle(pool)
    assert pool.active_lease_count == 1, "a shared slice was returned while a consumer still owns it"
    pool.cancel_proxy(proxy)  # Simulate consumer completion for fixture teardown.
    recycle(pool)
    assert pool.active_lease_count == 0
