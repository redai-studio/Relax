# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the scale-out weight-sync port-collision hardening.

Covers the two fixes for the observed 35B elastic regression
(``init_weights_send_group_for_remote_instance`` -> EADDRINUSE 50938):

1. Bounded seed-side port allocation window (never returns a port in / above the
   OS ephemeral range; raises on window exhaustion instead of scanning upward).
2. Self-healing retry of the whole NCCL group init with a fresh, rotated port
   window on bind/init failure.
3. Fail-closed result validation (only ``success is True`` passes).

These tests deliberately drive the REAL code paths (``get_free_port`` bounding,
``RolloutManager._allocate_weight_sync_ports`` rotation, and
``RolloutManager._sync_single_engine_weights`` retry loop) rather than mocks that
trivially pass.
"""

import asyncio
from unittest.mock import patch

import pytest


try:
    from relax.distributed.ray.rollout import ScaleOutStatus  # noqa: F401
    from relax.utils import misc

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import (
    AwaitableValue,
    create_test_manager,
    make_engine_group,
    make_mock_engine,
    make_rollout_server,
)


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


from relax.utils.logging_utils import get_logger  # noqa: E402


logger = get_logger(__name__)


EPHEMERAL_FLOOR = 32768  # Linux default ephemeral range starts here.


# ---------------------------------------------------------------------------
# Fake seed node: models a single node's port table for allocation + bind, so
# concurrency/retry tests exercise real overlap semantics rather than a mock
# that ignores start_port.
# ---------------------------------------------------------------------------
class FakeSeedNode:
    """Models allocation + NCCL rank-0 bind on the seed node."""

    def __init__(self, fail_first_inits: int = 0, honor_free_scan: bool = True):
        self.busy: set[int] = set()
        self.bound_blocks: list[list[int]] = []  # successfully bound blocks
        self.requested_blocks: list[list[int]] = []  # every block handed to init
        self.fail_first_inits = fail_first_inits
        self.init_calls = 0
        self.honor_free_scan = honor_free_scan

    def free_scan(self, start_port, consecutive, max_port):
        """Mirror ``misc.get_free_port`` semantics against ``self.busy``."""
        port = start_port
        if max_port is not None and port + consecutive - 1 > max_port:
            raise RuntimeError("start exceeds window")
        if self.honor_free_scan:
            while any((port + i) in self.busy for i in range(consecutive)):
                port += 1
                if max_port is not None and port + consecutive - 1 > max_port:
                    raise RuntimeError("window exhausted")
        return port

    def alloc(self, start_port=10000, consecutive=1, max_port=None):
        base = self.free_scan(start_port, consecutive, max_port)
        return AwaitableValue(("10.0.0.1", base))

    def init(self, master_address, ports, group_rank=0, world_size=2, group_name="g", backend="nccl"):
        self.init_calls += 1
        block = [int(p) for p in ports.split(",")]
        self.requested_blocks.append(block)
        if self.init_calls <= self.fail_first_inits:
            return AwaitableValue({"success": False, "message": "address already in use"})
        # Real bind: fail if any port already bound (overlap), else reserve.
        if any(p in self.busy for p in block):
            return AwaitableValue({"success": False, "message": "address already in use"})
        self.busy.update(block)
        self.bound_blocks.append(block)
        return AwaitableValue({"success": True})


def make_seed_engine(node: FakeSeedNode):
    e = make_mock_engine(url="http://seed:1", weight_version="v1")
    e._get_current_node_ip_and_free_port.remote.side_effect = node.alloc
    e.init_weights_send_group_for_remote_instance.remote.side_effect = node.init
    e.send_weights_to_remote_instance.remote.side_effect = lambda **kw: AwaitableValue({"success": True})
    return e


def make_new_engine():
    """A remote (rank-1) engine that always accepts init/send."""
    e = make_mock_engine()
    e.init_weights_send_group_for_remote_instance.remote.side_effect = lambda **kw: AwaitableValue({"success": True})
    e.send_weights_to_remote_instance.remote.side_effect = lambda **kw: AwaitableValue({"success": True})
    return e


# ===========================================================================
# Category 1: bounded range -- the allocator never returns a port in/above the
# ephemeral range and RAISES on window exhaustion.
# ===========================================================================
class TestBoundedPortWindow:
    def test_get_free_port_raises_when_all_busy(self):
        """All ports busy -> RuntimeError, never scans past max into
        ephemeral."""
        with patch.object(misc, "is_port_available", return_value=False):
            with pytest.raises(RuntimeError):
                misc.get_free_port(start_port=20000, consecutive=2, max_port=31999)

    def test_get_free_port_returns_below_max_when_low_ports_busy(self):
        """Low ports busy -> scan upward, return first free block still <
        max."""
        busy_below = set(range(20000, 21000))

        def avail(p):
            return p not in busy_below

        with patch.object(misc, "is_port_available", side_effect=avail):
            port = misc.get_free_port(start_port=20000, consecutive=1, max_port=31999)
        assert port == 21000
        assert port < 32000
        assert port < EPHEMERAL_FLOOR

    def test_get_free_port_refuses_to_return_ephemeral(self):
        """Free ports exist only in the ephemeral range -> RAISE, never return
        them."""

        def avail(p):
            return p >= 40000  # only ephemeral ports free

        with patch.object(misc, "is_port_available", side_effect=avail):
            with pytest.raises(RuntimeError):
                misc.get_free_port(start_port=20000, consecutive=2, max_port=31999)

    def test_get_free_port_start_beyond_max_raises(self):
        with patch.object(misc, "is_port_available", return_value=True):
            with pytest.raises(RuntimeError):
                misc.get_free_port(start_port=31999, consecutive=4, max_port=31999)

    def test_get_free_port_unbounded_default_preserved(self):
        """Default (max_port=None) preserves original unbounded behaviour."""
        busy = set(range(10000, 10005))

        def avail(p):
            return p not in busy

        with patch.object(misc, "is_port_available", side_effect=avail):
            port = misc.get_free_port(start_port=10000, consecutive=1)
        assert port == 10005

    @pytest.mark.asyncio
    async def test_allocate_weight_sync_ports_within_window(self):
        """`_allocate_weight_sync_ports` passes the bound through and returns
        ports strictly inside the bounded window (real get_free_port)."""
        manager = create_test_manager()
        seed = make_mock_engine()

        def alloc(start_port=10000, consecutive=1, max_port=None):
            # Force low ports busy so the real allocator must scan upward, and
            # honour the max_port bound the manager passes.
            def avail(p):
                return p >= 25000

            with patch.object(misc, "is_port_available", side_effect=avail):
                port = misc.get_free_port(start_port=start_port, consecutive=consecutive, max_port=max_port)
            return AwaitableValue(("10.0.0.1", port))

        seed._get_current_node_ip_and_free_port.remote.side_effect = alloc

        ports = await manager._allocate_weight_sync_ports(seed, tp_size=2)
        assert len(ports) == 2
        for p in ports:
            assert manager._WEIGHT_SYNC_PORT_BASE <= int(p) < manager._WEIGHT_SYNC_PORT_MAX
            assert int(p) < EPHEMERAL_FLOOR
        # First free port at/above 25000 given low ports forced busy.
        assert int(ports[0]) == 25000


# ===========================================================================
# Category 2: real concurrency -- concurrent syncs get DISJOINT port blocks via
# the rotating cursor, with no retry needed.
# ===========================================================================
class TestConcurrentAllocationDisjoint:
    @pytest.mark.asyncio
    async def test_concurrent_syncs_get_disjoint_blocks(self):
        node = FakeSeedNode()
        seed = make_seed_engine(node)
        manager = create_test_manager()

        new_engines = [make_new_engine() for _ in range(4)]
        tp_size = 2

        async def _one(idx, engine):
            return await manager._sync_single_engine_weights(
                seed_engine=seed,
                new_engine=engine,
                engine_index=idx,
                total_engines=len(new_engines),
                master_address="10.0.0.1",
                tp_size=tp_size,
                timeout=60,
            )

        results = await asyncio.gather(*[_one(i, e) for i, e in enumerate(new_engines)])

        # All succeeded, all on the first attempt (no retry): one init per engine.
        assert all(r is True for r in results)
        assert node.init_calls == len(new_engines)
        # The blocks actually BOUND by NCCL init are pairwise disjoint and inside
        # the bounded window (assert on used blocks, not requested offsets).
        assert len(node.bound_blocks) == len(new_engines)
        seen: set[int] = set()
        for block in node.bound_blocks:
            assert not (set(block) & seen), f"overlapping block {block}"
            seen.update(block)
            for p in block:
                assert manager._WEIGHT_SYNC_PORT_BASE <= p < manager._WEIGHT_SYNC_PORT_MAX

    @pytest.mark.asyncio
    async def test_cursor_wraps_within_window(self):
        """The rotating cursor wraps back to base and never leaves the
        window."""
        manager = create_test_manager()
        tp = 2
        span = manager._WEIGHT_SYNC_PORT_MAX - manager._WEIGHT_SYNC_PORT_BASE
        starts = [manager._next_weight_sync_start_port(tp) for _ in range(span // tp + 5)]
        assert all(manager._WEIGHT_SYNC_PORT_BASE <= s <= manager._WEIGHT_SYNC_PORT_MAX - tp for s in starts)
        assert manager._WEIGHT_SYNC_PORT_BASE in starts  # wrapped at least once


# ===========================================================================
# Category 3: self-healing retry on bind/init failure.
# ===========================================================================
class TestRetryOnBindFailure:
    @pytest.mark.asyncio
    async def test_retry_succeeds_with_fresh_window(self):
        # First init attempt fails ("address already in use"), second succeeds.
        node = FakeSeedNode(fail_first_inits=1)
        seed = make_seed_engine(node)
        new = make_new_engine()
        manager = create_test_manager()

        ok = await manager._sync_single_engine_weights(
            seed_engine=seed,
            new_engine=new,
            engine_index=0,
            total_engines=1,
            master_address="10.0.0.1",
            tp_size=2,
            timeout=60,
        )
        assert ok is True
        # Two init attempts were made, on DIFFERENT (rotated) port windows.
        assert node.init_calls == 2
        assert node.requested_blocks[0] != node.requested_blocks[1]

    @pytest.mark.asyncio
    async def test_gives_up_after_bounded_retries(self):
        # Init always fails -> bounded attempts then False.
        node = FakeSeedNode(fail_first_inits=999)
        seed = make_seed_engine(node)
        new = make_new_engine()
        manager = create_test_manager()

        ok = await manager._sync_single_engine_weights(
            seed_engine=seed,
            new_engine=new,
            engine_index=0,
            total_engines=1,
            master_address="10.0.0.1",
            tp_size=2,
            timeout=60,
        )
        assert ok is False
        assert node.init_calls == manager._WEIGHT_SYNC_MAX_INIT_ATTEMPTS
        # Every attempt used a distinct (rotated) window.
        blocks = [tuple(b) for b in node.requested_blocks]
        assert len(set(blocks)) == manager._WEIGHT_SYNC_MAX_INIT_ATTEMPTS

    @pytest.mark.asyncio
    async def test_retry_advances_past_resolved_block_when_scans_converge(self):
        """Different cursor starts may scan to the same free block.

        A retry must start after the block actually returned by the seed node,
        not merely after the prior scan start.
        """
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        new = make_new_engine()
        manager = create_test_manager()
        requested_starts = []

        def alloc(start_port=10000, consecutive=1, max_port=None):
            requested_starts.append(start_port)
            # Model a large busy interval: every scan at/below 20041 resolves
            # to 20041, exactly as observed in the 35B elastic e2e.
            resolved = 20041 if start_port <= 20041 else start_port
            return AwaitableValue(("10.0.0.1", resolved))

        seed._get_current_node_ip_and_free_port.remote.side_effect = alloc
        seed.init_weights_send_group_for_remote_instance.remote.side_effect = lambda **kw: AwaitableValue(
            {"success": False, "message": "init failed"}
        )

        ok = await manager._sync_single_engine_weights(
            seed_engine=seed,
            new_engine=new,
            engine_index=0,
            total_engines=1,
            master_address="10.0.0.1",
            tp_size=8,
            timeout=60,
        )

        assert ok is False
        blocks = [
            tuple(int(port) for port in call.kwargs["ports"].split(","))
            for call in seed.init_weights_send_group_for_remote_instance.remote.call_args_list
        ]
        assert len(set(blocks)) == manager._WEIGHT_SYNC_MAX_INIT_ATTEMPTS
        assert requested_starts[1] == blocks[0][-1] + 1

    @pytest.mark.asyncio
    async def test_concurrent_retries_advance_global_resolved_cursor(self):
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        manager = create_test_manager()
        requested_blocks = []

        def alloc(start_port=10000, consecutive=1, max_port=None):
            resolved = 20041 if start_port <= 20041 else start_port
            return AwaitableValue(("10.0.0.1", resolved))

        def fail_init(**kwargs):
            requested_blocks.append(tuple(int(port) for port in kwargs["ports"].split(",")))
            return AwaitableValue({"success": False, "message": "init failed"})

        seed._get_current_node_ip_and_free_port.remote.side_effect = alloc
        seed.init_weights_send_group_for_remote_instance.remote.side_effect = fail_init

        results = await asyncio.gather(
            *[
                manager._sync_single_engine_weights(
                    seed_engine=seed,
                    new_engine=make_new_engine(),
                    engine_index=index,
                    total_engines=2,
                    master_address="10.0.0.1",
                    tp_size=8,
                    timeout=60,
                )
                for index in range(2)
            ]
        )

        assert results == [False, False]
        assert len(requested_blocks) == 2 * manager._WEIGHT_SYNC_MAX_INIT_ATTEMPTS
        assert len(set(requested_blocks)) == len(requested_blocks)

    @pytest.mark.asyncio
    async def test_sync_failure_retries_to_limit_and_releases_lock(self, patch_async_helpers):
        """A seed whose init always fails retries to the bounded limit, then
        gives up -- and the global training weight-sync lock is ALWAYS
        released.

        A failed attempt rotates to a fresh port window and retries (bounded),
        rather than aborting on the first failure. Correct behaviour:
        1. The retry loop runs the FULL bounded number of attempts (port
           hardening retry is restored -- a single failure no longer aborts it).
        2. The distributed ``_weight_sync_lock`` -- the SAME lock every training
           weight update acquires -- is ALWAYS released. It must never be held
           as an isolation mechanism, or the next training update spins forever.
        3. Regression guard: a later sync with another always-failing seed also
           retries the full bounded number of attempts (no residual global state
           poisons subsequent syncs).
        """
        node = FakeSeedNode(fail_first_inits=999)
        seed = make_seed_engine(node)
        new = make_new_engine()
        group = make_engine_group(engines=[seed])
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})

        # Drive the FULL sync path so the lock acquire/release is exercised.
        ok = await manager._sync_weights_from_seed_engine([new], timeout=60)

        assert ok is False
        # (1) Port-hardening retry restored: full bounded attempts, not a
        # single-shot abort.
        assert node.init_calls == manager._WEIGHT_SYNC_MAX_INIT_ATTEMPTS
        # (2) The global training lock is ALWAYS released -- never retained.
        manager._weight_sync_lock.release.remote.assert_called_once()
        assert manager._is_weight_updating is False

        # (3) A DIFFERENT seed on a later call must NOT be prematurely broken by
        # any residual global state: it retries the full bounded number of
        # attempts.
        node2 = FakeSeedNode(fail_first_inits=999)
        seed2 = make_seed_engine(node2)
        new2 = make_new_engine()
        ok2 = await manager._sync_single_engine_weights(
            seed_engine=seed2,
            new_engine=new2,
            engine_index=0,
            total_engines=1,
            master_address="10.0.0.1",
            tp_size=2,
            timeout=60,
        )
        assert ok2 is False
        assert node2.init_calls == manager._WEIGHT_SYNC_MAX_INIT_ATTEMPTS


# ===========================================================================
# Category 4: fail-closed result validation.
# ===========================================================================
class TestFailClosedValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_result", [None, {}, {"foo": 1}, {"success": False}, {"success": None}])
    async def test_bad_init_result_treated_as_failure(self, bad_result):
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        new = make_new_engine()
        # Seed init always returns a non-True / malformed result.
        seed.init_weights_send_group_for_remote_instance.remote.side_effect = lambda **kw: AwaitableValue(bad_result)
        manager = create_test_manager()

        ok = await manager._sync_single_engine_weights(
            seed_engine=seed,
            new_engine=new,
            engine_index=0,
            total_engines=1,
            master_address="10.0.0.1",
            tp_size=1,
            timeout=60,
        )
        assert ok is False
        # Fail-closed => retried the bounded number of times.
        assert (
            seed.init_weights_send_group_for_remote_instance.remote.call_count
            == manager._WEIGHT_SYNC_MAX_INIT_ATTEMPTS
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_result", [None, {}, {"foo": 1}, {"success": False}])
    async def test_bad_send_result_treated_as_failure(self, bad_result):
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        new = make_new_engine()
        # Init passes (True) but send returns malformed/non-True on the seed.
        seed.init_weights_send_group_for_remote_instance.remote.side_effect = lambda **kw: AwaitableValue(
            {"success": True}
        )
        seed.send_weights_to_remote_instance.remote.side_effect = lambda **kw: AwaitableValue(bad_result)
        manager = create_test_manager()

        ok = await manager._sync_single_engine_weights(
            seed_engine=seed,
            new_engine=new,
            engine_index=0,
            total_engines=1,
            master_address="10.0.0.1",
            tp_size=1,
            timeout=60,
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_success_true_passes(self):
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        new = make_new_engine()
        seed.init_weights_send_group_for_remote_instance.remote.side_effect = lambda **kw: AwaitableValue(
            {"success": True}
        )
        seed.send_weights_to_remote_instance.remote.side_effect = lambda **kw: AwaitableValue({"success": True})
        manager = create_test_manager()

        ok = await manager._sync_single_engine_weights(
            seed_engine=seed,
            new_engine=new,
            engine_index=0,
            total_engines=1,
            master_address="10.0.0.1",
            tp_size=1,
            timeout=60,
        )
        assert ok is True
        # Success on first attempt: exactly one init call.
        assert seed.init_weights_send_group_for_remote_instance.remote.call_count == 1
