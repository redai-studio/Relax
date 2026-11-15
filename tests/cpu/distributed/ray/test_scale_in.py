# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for scale-in request creation, engine selection, draining, removal,
and cleanup."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


try:
    from relax.distributed.ray.rollout import (
        EngineGroupLifecycle,
        ScaleInRequest,
        ScaleInStatus,
        ScaleOutRequest,
        ScaleOutStatus,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import (
    AwaitableValue,
    create_test_manager,
    make_engine_group,
    make_mock_engine,
    make_rollout_server,
    mock_ray_get,
)


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


# ==================== create_scale_in_request ==============================


class TestCreateScaleInRequest:
    def test_basic_creation(self, patch_ray_get):
        initial = make_engine_group(engines=[make_mock_engine(), make_mock_engine()])
        scaled = make_engine_group(
            engines=[make_mock_engine()],
            is_scaled_out=True,
            rank_offset=2,
        )
        srv = make_rollout_server(engine_groups=[initial, scaled])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_in_request(num_replicas=2)
        assert result["status"] == "PENDING"

    def test_below_initial_count_rejected(self, patch_ray_get):
        """Cannot scale below the number of initial engines."""
        initial = make_engine_group(engines=[make_mock_engine(), make_mock_engine()])
        scaled = make_engine_group(
            engines=[make_mock_engine()],
            is_scaled_out=True,
            rank_offset=2,
        )
        srv = make_rollout_server(engine_groups=[initial, scaled])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_in_request(num_replicas=1)
        assert result["status"] == "REJECTED"

    def test_already_at_target_noop(self, patch_ray_get):
        """If current_total <= target, return NOOP."""
        initial = make_engine_group(engines=[make_mock_engine(), make_mock_engine()])
        srv = make_rollout_server(engine_groups=[initial])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_in_request(num_replicas=2)
        assert result["status"] == "NOOP"

    def test_above_target_noop(self, patch_ray_get):
        initial = make_engine_group(engines=[make_mock_engine()])
        srv = make_rollout_server(engine_groups=[initial])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_in_request(num_replicas=5)
        assert result["status"] == "NOOP"

    def test_mutual_exclusion_with_scale_out(self, patch_ray_get):
        initial = make_engine_group(engines=[make_mock_engine()])
        srv = make_rollout_server(engine_groups=[initial])
        manager = create_test_manager(servers={"default": srv})
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.CREATING,
        )

        result = manager.create_scale_in_request(num_replicas=1, engine_urls=["a:1"])
        assert result["status"] == "CONFLICT"

    def test_mutual_exclusion_with_scale_in(self, patch_ray_get):
        initial = make_engine_group(engines=[make_mock_engine()])
        srv = make_rollout_server(engine_groups=[initial])
        manager = create_test_manager(servers={"default": srv})
        manager._scale_in_requests["r1"] = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.DRAINING,
        )

        result = manager.create_scale_in_request(engine_urls=["a:1"])
        assert result["status"] == "CONFLICT"

    def test_model_not_found(self, patch_ray_get):
        manager = create_test_manager(servers={})
        with pytest.raises(ValueError, match="not found"):
            manager.create_scale_in_request(model_name="nope", engine_urls=["a:1"])

    def test_neither_replicas_nor_urls(self, patch_ray_get):
        """Neither num_replicas>0 nor engine_urls given == target 0 < initial
        count.

        Returns a clean REJECTED instead of raising ValueError (which the HTTP
        handler would map to 500). See create_scale_in_request.
        """
        g = make_engine_group()
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})
        result = manager.create_scale_in_request()
        assert result["status"] == "REJECTED"

    def test_by_engine_urls(self, patch_ray_get):
        initial = make_engine_group(engines=[make_mock_engine()])
        srv = make_rollout_server(engine_groups=[initial])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_in_request(engine_urls=["http://a:1"])
        assert result["status"] == "PENDING"
        req = manager._scale_in_requests[result["request_id"]]
        assert req.engine_urls == ["http://a:1"]

    def test_dry_run_field(self, patch_ray_get):
        initial = make_engine_group(engines=[make_mock_engine()])
        scaled = make_engine_group(
            engines=[make_mock_engine()],
            is_scaled_out=True,
            rank_offset=1,
        )
        srv = make_rollout_server(engine_groups=[initial, scaled])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_in_request(num_replicas=1, dry_run=True)
        req = manager._scale_in_requests[result["request_id"]]
        assert req.dry_run is True

    def test_force_field(self, patch_ray_get):
        initial = make_engine_group(engines=[make_mock_engine()])
        scaled = make_engine_group(
            engines=[make_mock_engine()],
            is_scaled_out=True,
            rank_offset=1,
        )
        srv = make_rollout_server(engine_groups=[initial, scaled])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_in_request(num_replicas=1, force=True)
        req = manager._scale_in_requests[result["request_id"]]
        assert req.force is True


# =================== _select_engines_for_removal ===========================


class TestSelectEnginesForRemoval:
    def test_only_scaled_out_engines(self, patch_ray_get):
        """Initial engines are never selected for removal."""
        e_init = make_mock_engine(url="http://init:1")
        e_scaled = make_mock_engine(url="http://scaled:2")
        g_init = make_engine_group(engines=[e_init])
        g_scaled = make_engine_group(
            engines=[e_scaled],
            is_scaled_out=True,
            rank_offset=1,
        )
        srv = make_rollout_server(engine_groups=[g_init, g_scaled])
        manager = create_test_manager(servers={"default": srv})

        req = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.PENDING,
            num_replicas=1,  # keep 1 total
        )
        infos = manager._select_engines_for_removal(req, srv)
        assert len(infos) == 1
        # The selected engine should be from the scaled-out group
        group, idx = infos[0]
        assert group.is_scaled_out

    def test_lifo_ordering(self, patch_ray_get):
        """Most recently added (tail) engines are removed first."""
        e_init = make_mock_engine()
        e_s1 = make_mock_engine(url="http://s1:1")
        e_s2 = make_mock_engine(url="http://s2:2")
        g_init = make_engine_group(engines=[e_init])
        g_s1 = make_engine_group(
            engines=[e_s1],
            is_scaled_out=True,
            rank_offset=1,
        )
        g_s2 = make_engine_group(
            engines=[e_s2],
            is_scaled_out=True,
            rank_offset=2,
        )
        srv = make_rollout_server(engine_groups=[g_init, g_s1, g_s2])
        manager = create_test_manager(servers={"default": srv})

        # Remove 1: should pick from g_s2 (last added)
        req = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.PENDING,
            num_replicas=2,
        )
        infos = manager._select_engines_for_removal(req, srv)
        assert len(infos) == 1
        assert infos[0][0] is g_s2

    def test_num_replicas_no_removal_needed(self, patch_ray_get):
        """When current <= target, return empty."""
        e_init = make_mock_engine()
        g_init = make_engine_group(engines=[e_init])
        srv = make_rollout_server(engine_groups=[g_init])
        manager = create_test_manager(servers={"default": srv})

        req = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.PENDING,
            num_replicas=5,
        )
        assert manager._select_engines_for_removal(req, srv) == []

    @pytest.mark.asyncio
    async def test_by_engine_urls(self, patch_ray_get):
        """Select engines matching specific URLs."""
        e1 = make_mock_engine(url="http://a:1")
        e2 = make_mock_engine(url="http://b:2")
        g = make_engine_group(engines=[e1, e2], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        req = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.PENDING,
            engine_urls=["http://a:1"],
        )
        candidates = await manager._resolve_scale_in_url_candidates(req, srv)
        infos = manager._select_engines_for_removal(req, srv, url_candidates=candidates)
        assert len(infos) == 1

    @pytest.mark.asyncio
    async def test_by_engine_urls_normalization(self, patch_ray_get):
        """URL normalization: http://host:port matches host:port."""
        e1 = make_mock_engine(url="http://a:1")
        g = make_engine_group(engines=[e1], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        req = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.PENDING,
            engine_urls=["a:1"],
        )
        candidates = await manager._resolve_scale_in_url_candidates(req, srv)
        infos = manager._select_engines_for_removal(req, srv, url_candidates=candidates)
        assert len(infos) == 1

    @pytest.mark.asyncio
    async def test_dead_engines_skipped(self, patch_ray_get):
        """Dead (None) engines are not candidates."""
        g = make_engine_group(engines=[None, None], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        req = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.PENDING,
            num_replicas=0,
            engine_urls=["a:1"],
        )
        candidates = await manager._resolve_scale_in_url_candidates(req, srv)
        infos = manager._select_engines_for_removal(req, srv, url_candidates=candidates)
        assert len(infos) == 0

    @pytest.mark.asyncio
    async def test_url_probe_does_not_block_eviction_fence(self, patch_ray_get):
        probe_started = asyncio.Event()
        release_probe = asyncio.Event()

        async def blocked_url():
            probe_started.set()
            await release_probe.wait()
            return "http://elastic:1"

        engine = make_mock_engine(url="http://elastic:1")
        engine.get_url.remote.return_value = blocked_url()
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        group.pg = (MagicMock(), [], [])
        srv = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": srv})
        req = ScaleInRequest(
            request_id="r-url",
            status=ScaleInStatus.PENDING,
            engine_urls=["elastic:1"],
        )
        manager._scale_in_requests[req.request_id] = req

        resolve_task = asyncio.create_task(manager._resolve_scale_in_url_candidates(req, srv))
        await asyncio.wait_for(probe_started.wait(), timeout=1)

        manager._handle_evictions([("default", group, 0)])

        assert group.eviction_requested is True
        assert group.lifecycle_status is EngineGroupLifecycle.DRAINING
        release_probe.set()
        assert await resolve_task == [(group, 0, engine)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mutation", ["replace_actor", "remove_group"])
    async def test_url_candidates_are_revalidated_before_claim(self, patch_ray_get, mutation):
        engine = make_mock_engine(url="http://elastic:1")
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": srv})
        req = ScaleInRequest(
            request_id="r-url",
            status=ScaleInStatus.PENDING,
            engine_urls=["elastic:1"],
        )

        candidates = await manager._resolve_scale_in_url_candidates(req, srv)
        if mutation == "replace_actor":
            group.all_engines[0] = make_mock_engine(url="http://elastic:1")
        else:
            srv.engine_groups.remove(group)

        assert manager._select_engines_for_removal(req, srv, url_candidates=candidates) == []

    @pytest.mark.asyncio
    async def test_url_probes_run_concurrently(self, patch_ray_get):
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_probes = asyncio.Event()

        async def blocked_url(started, url):
            started.set()
            await release_probes.wait()
            return url

        first = make_mock_engine(url="http://first:1")
        second = make_mock_engine(url="http://second:2")
        first.get_url.remote.return_value = blocked_url(first_started, "http://first:1")
        second.get_url.remote.return_value = blocked_url(second_started, "http://second:2")
        group = make_engine_group(engines=[first, second], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": srv})
        req = ScaleInRequest(
            request_id="r-url",
            status=ScaleInStatus.PENDING,
            engine_urls=["first:1", "second:2"],
        )

        resolve_task = asyncio.create_task(manager._resolve_scale_in_url_candidates(req, srv))
        await asyncio.wait_for(asyncio.gather(first_started.wait(), second_started.wait()), timeout=1)
        release_probes.set()

        assert await resolve_task == [(group, 0, first), (group, 1, second)]


# ========================= _drain_engines ==================================


class TestDrainEngines:
    @pytest.mark.asyncio
    async def test_removes_all_engines_from_router(self):
        e1 = make_mock_engine(url="http://a:1")
        e2 = make_mock_engine(url="http://b:2")
        g = make_engine_group(engines=[e1, e2], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        engine_infos = [(g, 0), (g, 1)]
        # Use force=True to skip sleep
        unregistered, failed = await manager._drain_engines(engine_infos, timeout=5, force=True)

        assert unregistered == engine_infos
        assert failed == []
        e1.unregister_from_router.remote.assert_called_once_with(wait_for_removal=True, timeout=5.0)
        e2.unregister_from_router.remote.assert_called_once_with(wait_for_removal=True, timeout=5.0)

    @pytest.mark.asyncio
    async def test_router_removal_failure_is_reported(self):
        engine = make_mock_engine()
        engine.unregister_from_router.remote.return_value = AwaitableValue(False)
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})
        monitor = MagicMock()
        monitor._engine_group = group
        manager._health_monitors.append(monitor)

        unregistered, failed = await manager._drain_engines([(group, 0)], timeout=5, force=True)

        assert unregistered == []
        assert failed == ["group_0_engine_0"]
        monitor.mark_intentionally_removed.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_skips_drain_wait(self):
        e1 = make_mock_engine()
        g = make_engine_group(engines=[e1], is_scaled_out=True)
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[g])})

        import time

        start = time.monotonic()
        await manager._drain_engines([(g, 0)], timeout=100, force=True)
        elapsed = time.monotonic() - start
        assert elapsed < 5  # force should skip the 100s wait


# ========================= _scale_in =======================================


class TestScaleInExecution:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("force", [False, True])
    async def test_scale_in_fence_rejects_update_started_during_removal(self, force, patch_ray_get):
        engine = make_mock_engine(url="http://elastic:1")
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        server = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": server})
        request = ScaleInRequest(
            request_id="r-fence",
            status=ScaleInStatus.PENDING,
            engine_urls=["http://elastic:1"],
            timeout_secs=5,
            force=force,
        )
        removal_started = asyncio.Event()
        finish_removal = asyncio.Event()

        async def _remove(*_args, **_kwargs):
            removal_started.set()
            await finish_removal.wait()
            return ["group_0_engine_0"], []

        manager._remove_live_engines = _remove
        scale_in_task = asyncio.create_task(manager._scale_in(request))
        await removal_started.wait()

        assert group.lifecycle_status is EngineGroupLifecycle.DRAINING
        assert manager.set_weight_updating(True) is False
        assert manager._training_weight_updating is False

        finish_removal.set()
        await scale_in_task

        assert request.status is ScaleInStatus.COMPLETED
        assert group.lifecycle_status is EngineGroupLifecycle.ACTIVE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("owner", ["_training_weight_updating", "_scale_out_weight_updating"])
    async def test_scale_in_fence_waits_for_existing_weight_update(self, owner, patch_ray_get):
        engine = make_mock_engine(url="http://elastic:1")
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})
        setattr(manager, owner, True)
        manager._remove_live_engines = AsyncMock(return_value=(["engine"], []))
        request = ScaleInRequest(
            request_id="r-existing-update",
            status=ScaleInStatus.PENDING,
            engine_urls=["http://elastic:1"],
            timeout_secs=5,
        )

        scale_in_task = asyncio.create_task(manager._scale_in(request))
        while group.lifecycle_status is not EngineGroupLifecycle.DRAINING:
            await asyncio.sleep(0)

        manager._remove_live_engines.assert_not_called()
        assert manager.set_weight_updating(True) is False

        if owner == "_training_weight_updating":
            assert manager.set_weight_updating(False) is True
        else:
            manager._scale_out_weight_updating = False
        await scale_in_task

        manager._remove_live_engines.assert_awaited_once()
        assert request.status is ScaleInStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_scale_in_fence_timeout_restores_active_without_cleanup(self, patch_ray_get):
        engine = make_mock_engine(url="http://elastic:1")
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})
        manager._training_weight_updating = True
        manager._remove_live_engines = MagicMock()
        request = ScaleInRequest(
            request_id="r-timeout",
            status=ScaleInStatus.PENDING,
            engine_urls=["http://elastic:1"],
            timeout_secs=0.01,
        )

        await manager._scale_in(request)

        assert request.status is ScaleInStatus.FAILED
        assert "weight-update fence" in request.error_message
        assert group.lifecycle_status is EngineGroupLifecycle.ACTIVE
        manager._remove_live_engines.assert_not_called()

    @pytest.mark.asyncio
    async def test_sigterm_during_explicit_scale_in_timeout_keeps_persistent_fence(self, patch_ray_get):
        engine = make_mock_engine(url="http://elastic:1")
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        group.pg = (MagicMock(), [], [])
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})
        manager._training_weight_updating = True
        manager._remove_live_engines = MagicMock()
        request = ScaleInRequest(
            request_id="r-timeout-sigterm",
            status=ScaleInStatus.PENDING,
            engine_urls=["http://elastic:1"],
            timeout_secs=0.01,
        )
        manager._scale_in_requests[request.request_id] = request

        scale_in_task = asyncio.create_task(manager._scale_in(request))
        while group.lifecycle_status is not EngineGroupLifecycle.DRAINING:
            await asyncio.sleep(0)

        manager._handle_evictions([("default", group, 0)])
        assert group.eviction_requested is True

        await scale_in_task

        assert request.status is ScaleInStatus.FAILED
        assert group.lifecycle_status is EngineGroupLifecycle.DRAINING
        assert group.eviction_requested is True
        assert manager.set_weight_updating(True) is False
        manager._remove_live_engines.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_removes_only_engines_unregistered_from_router(self, patch_ray_get):
        removable = make_mock_engine(url="http://a:1")
        kept = make_mock_engine(url="http://b:2")
        kept.unregister_from_router.remote.return_value = AwaitableValue(False)
        group = make_engine_group(engines=[removable, kept], is_scaled_out=True)
        server = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": server})
        monitor = MagicMock()
        monitor._engine_group = group
        manager._health_monitors.append(monitor)
        request = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.PENDING,
            engine_urls=["http://a:1", "http://b:2"],
            force=True,
        )

        await manager._scale_in(request)

        assert request.status == ScaleInStatus.FAILED
        assert request.removed_engines == ["group_0_engine_0"]
        assert request.failed_engines == ["group_0_engine_1"]
        assert group.all_engines[0] is None
        assert group.all_engines[1] is kept
        removable.shutdown.remote.assert_called_once()
        kept.shutdown.remote.assert_not_called()
        monitor.mark_intentionally_removed.assert_called_once_with(0)


@pytest.mark.asyncio
async def test_live_removal_orders_router_drain_dcs_shutdown_and_pg(patch_ray_get):
    events = []
    engine = make_mock_engine()
    engine.unregister_from_router.remote.side_effect = lambda **_kwargs: (
        events.append("router") or AwaitableValue(True)
    )
    engine.unregister_dcs.remote.side_effect = lambda: events.append("dcs") or AwaitableValue(None)
    engine.shutdown.remote.side_effect = lambda: events.append("shutdown") or AwaitableValue(None)
    group = make_engine_group(engines=[engine], is_scaled_out=True)
    group.pg = (MagicMock(), [], [])
    server = make_rollout_server(engine_groups=[group])
    manager = create_test_manager(servers={"default": server})

    async def _sleep(_seconds):
        events.append("drain")

    with (
        patch("relax.distributed.ray.rollout.asyncio.sleep", side_effect=_sleep),
        patch(
            "relax.distributed.ray.rollout.ray.util.remove_placement_group",
            side_effect=lambda _pg: events.append("pg"),
        ),
    ):
        removed, failed = await manager._remove_live_engines(
            server,
            [(group, 0)],
            drain_timeout=1,
            shutdown_timeout=2,
            force=False,
        )

    assert removed == ["group_0_engine_0"]
    assert failed == []
    assert events == ["router", "drain", "dcs", "shutdown", "pg"]


@pytest.mark.asyncio
async def test_batch_live_removal_runs_each_phase_concurrently_and_drains_once(patch_ray_get):
    events = []

    def concurrent_phase(name):
        started = 0
        both_started = asyncio.Event()

        async def run():
            nonlocal started
            started += 1
            events.append(f"{name}_start_{started}")
            if started == 2:
                both_started.set()
            await both_started.wait()
            events.append(f"{name}_end")
            return True

        return run

    router_phase = concurrent_phase("router")
    dcs_phase = concurrent_phase("dcs")
    shutdown_phase = concurrent_phase("shutdown")
    groups = []
    for rank_offset in (1, 2):
        engine = make_mock_engine()
        engine.unregister_from_router.remote.side_effect = lambda **_kwargs: router_phase()
        engine.unregister_dcs.remote.side_effect = lambda: dcs_phase()
        engine.shutdown.remote.side_effect = lambda: shutdown_phase()
        group = make_engine_group(engines=[engine], is_scaled_out=True, rank_offset=rank_offset)
        group.pg = (MagicMock(), [], [])
        groups.append(group)

    server = make_rollout_server(engine_groups=groups)
    manager = create_test_manager(servers={"default": server})

    async def drain_once(_seconds):
        events.append("drain")

    with (
        patch("relax.distributed.ray.rollout.asyncio.sleep", side_effect=drain_once) as sleep,
        patch("relax.distributed.ray.rollout.ray.util.remove_placement_group") as remove_pg,
    ):
        removed, failed = await manager._remove_live_engines(
            server,
            [(groups[0], 0), (groups[1], 0)],
            drain_timeout=1,
            shutdown_timeout=2,
            force=False,
        )

    assert removed == ["group_1_engine_0", "group_2_engine_0"]
    assert failed == []
    sleep.assert_awaited_once_with(1)
    assert events.count("drain") == 1
    assert events.index("router_start_2") < events.index("router_end") < events.index("drain")
    assert events.index("drain") < events.index("dcs_start_1")
    assert events.index("dcs_start_2") < events.index("dcs_end")
    assert events.index("dcs_end") < events.index("shutdown_start_1")
    assert events.index("shutdown_start_2") < events.index("shutdown_end")
    assert remove_pg.call_count == 2
    assert server.engine_groups == []


@pytest.mark.asyncio
async def test_dcs_failure_warns_and_continues_shutdown(patch_ray_get):
    engine = make_mock_engine()
    engine.unregister_dcs.remote.side_effect = RuntimeError("DCS unavailable")
    group = make_engine_group(engines=[engine], is_scaled_out=True)
    server = make_rollout_server(engine_groups=[group])
    manager = create_test_manager(servers={"default": server})

    removed, failed = await manager._remove_live_engines(
        server,
        [(group, 0)],
        drain_timeout=1,
        shutdown_timeout=2,
        force=True,
    )

    assert removed == ["group_0_engine_0"]
    assert failed == []
    engine.shutdown.remote.assert_called_once()
    assert group.all_engines[0] is None
    assert server.engine_groups == []


# ========================= _remove_engine ==================================


class TestRemoveEngine:
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, patch_ray_get):
        e1 = make_mock_engine()
        g = make_engine_group(engines=[e1], is_scaled_out=True)

        await manager_remove_engine(g, 0)

        assert g.all_engines[0] is None

    @pytest.mark.asyncio
    async def test_fallback_to_ray_kill_on_shutdown_failure(self):
        e1 = make_mock_engine()
        e1.shutdown.remote.return_value = AwaitableValue(None)
        # Make shutdown raise a timeout
        e1.shutdown.remote.side_effect = Exception("timeout")
        g = make_engine_group(engines=[e1], is_scaled_out=True)

        with patch("ray.kill") as mock_kill, patch("ray.get", side_effect=mock_ray_get):
            manager = create_test_manager()
            await manager._remove_engine(g, 0, shutdown_timeout=1)
            mock_kill.assert_called_once_with(e1)
        assert g.all_engines[0] is None

    @pytest.mark.asyncio
    async def test_dcs_failure_still_shuts_down_engine(self):
        engine = make_mock_engine()
        engine.unregister_dcs.remote.side_effect = RuntimeError("DCS unavailable")
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        manager = create_test_manager()

        with patch("ray.kill") as kill:
            await manager._remove_engine(group, 0, shutdown_timeout=1)

        assert group.all_engines[0] is None
        engine.shutdown.remote.assert_called_once()
        kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_node_engine_removal(self, patch_ray_get):
        """All sub-actors of a multi-node engine are removed."""
        args = type("A", (), {"num_gpus_per_node": 4})()
        engines = [make_mock_engine() for _ in range(2)]
        g = make_engine_group(
            args=args,
            engines=engines,
            num_gpus_per_engine=8,
            is_scaled_out=True,
        )
        # nodes_per_engine = 8 / 4 = 2
        assert g.nodes_per_engine == 2

        manager = create_test_manager()
        await manager._remove_engine(g, 0, shutdown_timeout=5)
        assert g.all_engines[0] is None
        assert g.all_engines[1] is None


async def manager_remove_engine(g, idx):
    """Helper to call _remove_engine with a test manager."""
    with patch("ray.get", side_effect=mock_ray_get):
        manager = create_test_manager()
        await manager._remove_engine(g, idx, shutdown_timeout=5)


# ===================== _cleanup_engine_groups ==============================


class TestCleanupEngineGroups:
    def test_removes_empty_groups(self):
        g_live = make_engine_group(engines=[make_mock_engine()])
        g_empty = make_engine_group(engines=[None, None], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g_live, g_empty])
        manager = create_test_manager(servers={"default": srv})

        manager._cleanup_engine_groups(srv)
        assert len(srv.engine_groups) == 1
        assert srv.engine_groups[0] is g_live

    def test_keeps_non_empty_groups(self):
        g1 = make_engine_group(engines=[make_mock_engine()])
        g2 = make_engine_group(engines=[make_mock_engine()], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g1, g2])
        manager = create_test_manager(servers={"default": srv})

        manager._cleanup_engine_groups(srv)
        assert len(srv.engine_groups) == 2

    def test_stops_health_monitors_for_removed_groups(self):
        g_empty = make_engine_group(engines=[None], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g_empty])
        manager = create_test_manager(servers={"default": srv})

        mock_monitor = MagicMock()
        mock_monitor._engine_group = g_empty
        manager._health_monitors.append(mock_monitor)

        manager._cleanup_engine_groups(srv)
        mock_monitor.stop.assert_called_once()
        assert mock_monitor not in manager._health_monitors

    def test_removes_placement_group(self):
        mock_pg = MagicMock()
        g_empty = make_engine_group(engines=[None], is_scaled_out=True)
        g_empty.pg = (mock_pg, [], [])
        srv = make_rollout_server(engine_groups=[g_empty])
        manager = create_test_manager(servers={"default": srv})

        with patch("ray.util.remove_placement_group") as mock_remove:
            manager._cleanup_engine_groups(srv)
            mock_remove.assert_called_once_with(mock_pg)


# ===================== Scale-in status queries =============================


class TestScaleInStatusQueries:
    def test_get_status_found(self):
        manager = create_test_manager()
        manager._scale_in_requests["r1"] = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.COMPLETED,
        )
        result = manager.get_scale_in_status("r1")
        assert result["status"] == "COMPLETED"

    def test_get_status_not_found(self):
        manager = create_test_manager()
        assert manager.get_scale_in_status("nope") is None

    def test_list_all(self):
        manager = create_test_manager()
        manager._scale_in_requests["r1"] = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.COMPLETED,
        )
        manager._scale_in_requests["r2"] = ScaleInRequest(
            request_id="r2",
            status=ScaleInStatus.FAILED,
        )
        result = manager.list_all_scale_in_requests()
        assert len(result) == 2

    def test_list_filter_by_status(self):
        manager = create_test_manager()
        manager._scale_in_requests["r1"] = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.COMPLETED,
        )
        manager._scale_in_requests["r2"] = ScaleInRequest(
            request_id="r2",
            status=ScaleInStatus.FAILED,
        )
        result = manager.list_all_scale_in_requests(status_filter="COMPLETED")
        assert len(result) == 1
        assert result[0]["status"] == "COMPLETED"

    def test_list_filter_by_model(self):
        manager = create_test_manager()
        manager._scale_in_requests["r1"] = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.COMPLETED,
            model_name="actor",
        )
        manager._scale_in_requests["r2"] = ScaleInRequest(
            request_id="r2",
            status=ScaleInStatus.COMPLETED,
            model_name="reward",
        )
        result = manager.list_all_scale_in_requests(model_name="actor")
        assert len(result) == 1

    def test_list_invalid_status(self):
        manager = create_test_manager()
        with pytest.raises(ValueError, match="Invalid status"):
            manager.list_all_scale_in_requests(status_filter="BOGUS")
