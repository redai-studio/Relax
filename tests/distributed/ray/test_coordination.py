# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for mutual exclusion, GC of terminal requests, eviction monitoring,
and engine info queries."""

import asyncio
import threading
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
    create_test_manager,
    make_engine_group,
    make_mock_engine,
    make_rollout_server,
)


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


# ==================== _find_active_scale_request ===========================


class TestFindActiveScaleRequest:
    def test_no_active_requests(self):
        manager = create_test_manager()
        assert manager._find_active_scale_request() is None

    def test_finds_active_scale_out(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.CREATING,
        )
        result = manager._find_active_scale_request()
        assert result is not None
        assert result["type"] == "scale_out"
        assert result["request_id"] == "r1"

    def test_finds_active_scale_in(self):
        manager = create_test_manager()
        manager._scale_in_requests["r1"] = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.DRAINING,
        )
        result = manager._find_active_scale_request()
        assert result is not None
        assert result["type"] == "scale_in"

    def test_ignores_terminal_scale_out(self):
        manager = create_test_manager()
        for status in (ScaleOutStatus.ACTIVE, ScaleOutStatus.FAILED, ScaleOutStatus.CANCELLED, ScaleOutStatus.PARTIAL):
            manager._scale_out_requests[status.value] = ScaleOutRequest(
                request_id=status.value,
                status=status,
            )
        assert manager._find_active_scale_request() is None

    def test_ignores_terminal_scale_in(self):
        manager = create_test_manager()
        for status in (ScaleInStatus.COMPLETED, ScaleInStatus.FAILED):
            manager._scale_in_requests[status.value] = ScaleInRequest(
                request_id=status.value,
                status=status,
            )
        assert manager._find_active_scale_request() is None

    def test_scale_out_checked_before_scale_in(self):
        """When both are active, scale_out is returned first."""
        manager = create_test_manager()
        manager._scale_out_requests["so"] = ScaleOutRequest(
            request_id="so",
            status=ScaleOutStatus.PENDING,
        )
        manager._scale_in_requests["si"] = ScaleInRequest(
            request_id="si",
            status=ScaleInStatus.PENDING,
        )
        result = manager._find_active_scale_request()
        assert result["type"] == "scale_out"

    @pytest.mark.parametrize(
        "status_str",
        ["PENDING", "CREATING", "CONNECTING", "HEALTH_CHECKING", "WEIGHT_SYNCING", "READY", "REMOVING"],
    )
    def test_all_non_terminal_scale_out_detected(self, status_str):
        status = ScaleOutStatus(status_str)
        manager = create_test_manager()
        manager._scale_out_requests["r"] = ScaleOutRequest(
            request_id="r",
            status=status,
        )
        assert manager._find_active_scale_request() is not None

    @pytest.mark.parametrize(
        "status_str",
        ["PENDING", "DRAINING", "REMOVING"],
    )
    def test_all_non_terminal_scale_in_detected(self, status_str):
        status = ScaleInStatus(status_str)
        manager = create_test_manager()
        manager._scale_in_requests["r"] = ScaleInRequest(
            request_id="r",
            status=status,
        )
        assert manager._find_active_scale_request() is not None

    def test_finds_claimed_graceful_eviction_without_request(self):
        group = make_engine_group(is_scaled_out=True)
        group.lifecycle_status = EngineGroupLifecycle.DRAINING
        group.eviction_requested = True
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})

        result = manager._find_active_scale_request()

        assert result == {
            "type": "graceful_eviction",
            "request_id": "default:0",
            "status": "DRAINING",
        }

    def test_explicit_scale_in_precedes_its_draining_group(self):
        group = make_engine_group(is_scaled_out=True)
        group.lifecycle_status = EngineGroupLifecycle.DRAINING
        group.eviction_requested = True
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})
        manager._scale_in_requests["scale-in"] = ScaleInRequest(
            request_id="scale-in",
            status=ScaleInStatus.DRAINING,
        )

        result = manager._find_active_scale_request()

        assert result["type"] == "scale_in"
        assert result["request_id"] == "scale-in"


# ===================== _gc_terminal_requests ===============================


class TestGCTerminalRequests:
    def test_no_gc_under_limit(self):
        manager = create_test_manager()
        manager._max_terminal_requests = 10
        for i in range(5):
            manager._scale_out_requests[f"r{i}"] = ScaleOutRequest(
                request_id=f"r{i}",
                status=ScaleOutStatus.ACTIVE,
            )
        manager._gc_terminal_requests()
        assert len(manager._scale_out_requests) == 5

    def test_gc_evicts_oldest(self):
        manager = create_test_manager()
        manager._max_terminal_requests = 3

        for i in range(5):
            req = ScaleOutRequest(
                request_id=f"r{i}",
                status=ScaleOutStatus.ACTIVE,
            )
            req.updated_at = 1000 + i  # r0 is oldest
            manager._scale_out_requests[f"r{i}"] = req

        manager._gc_terminal_requests()
        assert len(manager._scale_out_requests) == 3
        # r0, r1 should be evicted (oldest)
        assert "r0" not in manager._scale_out_requests
        assert "r1" not in manager._scale_out_requests
        assert "r4" in manager._scale_out_requests

    def test_gc_preserves_non_terminal(self):
        manager = create_test_manager()
        manager._max_terminal_requests = 1

        # 3 terminal
        for i in range(3):
            req = ScaleOutRequest(
                request_id=f"t{i}",
                status=ScaleOutStatus.ACTIVE,
            )
            req.updated_at = 1000 + i
            manager._scale_out_requests[f"t{i}"] = req

        # 1 non-terminal (should never be evicted)
        nt = ScaleOutRequest(request_id="nt", status=ScaleOutStatus.CREATING)
        nt.updated_at = 0  # oldest timestamp
        manager._scale_out_requests["nt"] = nt

        manager._gc_terminal_requests()
        # Only 1 terminal kept, but non-terminal is preserved
        assert "nt" in manager._scale_out_requests
        terminal_count = sum(1 for r in manager._scale_out_requests.values() if r.is_terminal())
        assert terminal_count == 1

    def test_gc_scale_in_requests(self):
        manager = create_test_manager()
        manager._max_terminal_requests = 2

        for i in range(4):
            req = ScaleInRequest(
                request_id=f"si{i}",
                status=ScaleInStatus.COMPLETED,
            )
            req.updated_at = 2000 + i
            manager._scale_in_requests[f"si{i}"] = req

        manager._gc_terminal_requests()
        assert len(manager._scale_in_requests) == 2


# =================== Eviction monitoring ==================================


class TestCheckAndHandleEvictions:
    def test_detects_evicted_engines(self, patch_ray_get):
        e_normal = make_mock_engine(evicted=False)
        e_evicted = make_mock_engine(evicted=True)
        g = make_engine_group(engines=[e_normal, e_evicted], is_scaled_out=True)
        g.pg = (MagicMock(), [], [])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        with (
            patch.object(manager, "_handle_evictions") as mock_handle,
            patch("relax.distributed.ray.rollout.ray.wait", side_effect=lambda refs, **_kwargs: (refs, [])),
        ):
            manager._check_and_handle_evictions()
            mock_handle.assert_called_once()
            assert mock_handle.call_args.args[0] == [("default", g, 1)]

    def test_no_evictions(self, patch_ray_get):
        e = make_mock_engine(evicted=False)
        g = make_engine_group(engines=[e])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        with patch.object(manager, "_handle_evictions") as mock_handle:
            manager._check_and_handle_evictions()
            mock_handle.assert_not_called()

    def test_pending_probe_does_not_hide_another_eviction(self, patch_ray_get):
        pending = make_mock_engine(evicted=False)
        evicted = make_mock_engine(evicted=True)
        group = make_engine_group(engines=[pending, evicted], is_scaled_out=True)
        group.pg = (MagicMock(), [], [])
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})

        def _ready_only(refs, **_kwargs):
            return [ref for ref in refs if getattr(ref, "value", False)], []

        with (
            patch.object(manager, "_handle_evictions") as handle,
            patch("relax.distributed.ray.rollout.ray.wait", side_effect=_ready_only),
        ):
            manager._check_and_handle_evictions()

        handle.assert_called_once_with([("default", group, 1)])

    def test_all_dead_engines_skipped(self, patch_ray_get):
        g = make_engine_group(engines=[None, None])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        with patch.object(manager, "_handle_evictions") as mock_handle:
            manager._check_and_handle_evictions()
            mock_handle.assert_not_called()

    def test_collects_ready_evictions_into_one_batch(self, patch_ray_get):
        first = make_mock_engine(evicted=True)
        second = make_mock_engine(evicted=True)
        first_group = make_engine_group(engines=[first], is_scaled_out=True, rank_offset=1)
        second_group = make_engine_group(engines=[second], is_scaled_out=True, rank_offset=2)
        first_group.pg = (MagicMock(), [], [])
        second_group.pg = (MagicMock(), [], [])
        manager = create_test_manager(
            servers={"default": make_rollout_server(engine_groups=[first_group, second_group])}
        )

        with (
            patch.object(manager, "_handle_evictions") as handle,
            patch("relax.distributed.ray.rollout.ray.wait", side_effect=lambda refs, **_kwargs: (refs, [])),
        ):
            manager._check_and_handle_evictions()

        handle.assert_called_once_with([("default", first_group, 0), ("default", second_group, 0)])


class TestHandleEvictions:
    def test_batch_claims_all_groups_before_waiting_once(self, patch_ray_get):
        first_group = make_engine_group(engines=[make_mock_engine()], is_scaled_out=True, rank_offset=1)
        second_group = make_engine_group(engines=[make_mock_engine()], is_scaled_out=True, rank_offset=2)
        first_group.pg = (MagicMock(), [], [])
        second_group.pg = (MagicMock(), [], [])
        server = make_rollout_server(engine_groups=[first_group, second_group])
        manager = create_test_manager(servers={"default": server})
        manager._training_weight_updating = True
        removed = ["group_1_engine_0", "group_2_engine_0"]

        def release_owner(_seconds):
            assert first_group.lifecycle_status is EngineGroupLifecycle.DRAINING
            assert second_group.lifecycle_status is EngineGroupLifecycle.DRAINING
            manager._training_weight_updating = False

        with (
            patch("relax.distributed.ray.rollout.time.sleep", side_effect=release_owner) as sleep,
            patch.object(
                manager, "_remove_live_engines", new_callable=AsyncMock, return_value=(removed, [])
            ) as remove,
        ):
            manager._handle_evictions([("default", first_group, 0), ("default", second_group, 0)])

        sleep.assert_called_once()
        remove.assert_awaited_once_with(
            server,
            [(first_group, 0), (second_group, 0)],
            drain_timeout=30.0,
            shutdown_timeout=30.0,
            force=False,
        )

    def test_batch_dcs_failure_does_not_block_cleanup(self, patch_ray_get):
        dcs_failed_engine = make_mock_engine()
        dcs_failed_engine.unregister_dcs.remote.side_effect = RuntimeError("DCS unavailable")
        healthy_engine = make_mock_engine()
        dcs_failed_group = make_engine_group(engines=[dcs_failed_engine], is_scaled_out=True, rank_offset=1)
        healthy_group = make_engine_group(engines=[healthy_engine], is_scaled_out=True, rank_offset=2)
        dcs_failed_group.pg = (MagicMock(), [], [])
        healthy_group.pg = (MagicMock(), [], [])
        server = make_rollout_server(engine_groups=[dcs_failed_group, healthy_group])
        manager = create_test_manager(servers={"default": server})
        manager.args.scale_in_drain_timeout = 0

        with patch("ray.util.remove_placement_group"):
            manager._handle_evictions([("default", dcs_failed_group, 0), ("default", healthy_group, 0)])

        assert dcs_failed_group.lifecycle_status is EngineGroupLifecycle.REMOVED
        assert healthy_group.lifecycle_status is EngineGroupLifecycle.REMOVED
        assert dcs_failed_group not in server.engine_groups
        assert healthy_group not in server.engine_groups
        assert dcs_failed_group.all_engines[0] is None
        assert healthy_group.all_engines[0] is None
        dcs_failed_engine.shutdown.remote.assert_called_once()
        healthy_engine.shutdown.remote.assert_called_once()
        assert manager.set_weight_updating(True) is True

    @pytest.mark.parametrize("request_kind", ["scale_out", "scale_in"])
    def test_eviction_claim_blocks_new_scale_request(self, patch_ray_get, request_kind):
        initial = make_engine_group(engines=[make_mock_engine()])
        engine = make_mock_engine()
        group = make_engine_group(engines=[engine], is_scaled_out=True, rank_offset=1)
        group.pg = (MagicMock(), [], [])
        server = make_rollout_server(engine_groups=[initial, group])
        manager = create_test_manager(servers={"default": server})
        manager.args.scale_in_drain_timeout = 0
        removal_started = threading.Event()
        allow_removal = threading.Event()

        async def block_removal(*_args, **_kwargs):
            removal_started.set()
            assert allow_removal.wait(timeout=5)
            return ["group_1_engine_0"], []

        with patch.object(manager, "_remove_live_engines", side_effect=block_removal):
            eviction_thread = threading.Thread(
                target=manager._handle_evictions,
                args=([("default", group, 0)],),
            )
            eviction_thread.start()
            assert removal_started.wait(timeout=5)

            if request_kind == "scale_out":
                result = manager.create_scale_out_request(num_replicas=3)
            else:
                result = manager.create_scale_in_request(num_replicas=1)

            assert result["status"] == "CONFLICT"
            assert group.lifecycle_status is EngineGroupLifecycle.DRAINING
            allow_removal.set()
            eviction_thread.join(timeout=5)
            assert not eviction_thread.is_alive()

    def test_inserted_scale_out_does_not_delay_eviction_fence(self, patch_ray_get):
        initial = make_engine_group(engines=[make_mock_engine()])
        engine = make_mock_engine()
        group = make_engine_group(engines=[engine], is_scaled_out=True, rank_offset=1)
        group.pg = (MagicMock(), [], [])
        server = make_rollout_server(engine_groups=[initial, group])
        manager = create_test_manager(servers={"default": server})
        manager.args.scale_in_drain_timeout = 0
        removal_started = threading.Event()
        allow_removal = threading.Event()

        result = manager.create_scale_out_request(num_replicas=3)
        assert result["status"] == "PENDING"

        async def block_removal(*_args, **_kwargs):
            removal_started.set()
            assert allow_removal.wait(timeout=5)
            return ["group_1_engine_0"], []

        with patch.object(manager, "_remove_live_engines", side_effect=block_removal):
            eviction_thread = threading.Thread(
                target=manager._handle_evictions,
                args=([("default", group, 0)],),
            )
            eviction_thread.start()
            assert removal_started.wait(timeout=5)

            assert group.lifecycle_status is EngineGroupLifecycle.DRAINING
            assert manager.set_weight_updating(True) is False
            allow_removal.set()
            eviction_thread.join(timeout=5)
            assert not eviction_thread.is_alive()

    def test_pending_target_scale_in_adopts_eviction_without_extra_removal(self, patch_ray_get):
        initial_engine = make_mock_engine()
        initial = make_engine_group(engines=[initial_engine])
        elastic_engine = make_mock_engine()
        group = make_engine_group(engines=[elastic_engine], is_scaled_out=True, rank_offset=1)
        group.pg = (MagicMock(), [], [])
        server = make_rollout_server(engine_groups=[initial, group])
        manager = create_test_manager(servers={"default": server})
        manager.args.scale_in_drain_timeout = 0

        result = manager.create_scale_in_request(num_replicas=1)
        assert result["status"] == "PENDING"

        manager._handle_evictions([("default", group, 0)])

        assert group.lifecycle_status is EngineGroupLifecycle.DRAINING
        elastic_engine.unregister_from_router.remote.assert_not_called()

        request = manager._scale_in_requests[result["request_id"]]
        with patch("ray.util.remove_placement_group"):
            asyncio.run(manager._scale_in(request))

        assert request.status is ScaleInStatus.COMPLETED
        assert request.selected_engines == ["group_1_engine_0"]
        elastic_engine.unregister_from_router.remote.assert_called_once()
        initial_engine.unregister_from_router.remote.assert_not_called()

    def test_scale_in_by_url_leaves_other_eviction_for_next_batch(self, patch_ray_get):
        initial = make_engine_group(engines=[make_mock_engine()])
        first_engine = make_mock_engine(url="http://first:1")
        second_engine = make_mock_engine(url="http://second:2")
        first_group = make_engine_group(engines=[first_engine], is_scaled_out=True, rank_offset=1)
        second_group = make_engine_group(engines=[second_engine], is_scaled_out=True, rank_offset=2)
        first_group.pg = (MagicMock(), [], [])
        second_group.pg = (MagicMock(), [], [])
        server = make_rollout_server(engine_groups=[initial, first_group, second_group])
        manager = create_test_manager(servers={"default": server})
        manager.args.scale_in_drain_timeout = 0

        result = manager.create_scale_in_request(engine_urls=["first:1"])
        assert result["status"] == "PENDING"
        manager._handle_evictions([("default", first_group, 0), ("default", second_group, 0)])

        request = manager._scale_in_requests[result["request_id"]]
        with patch("ray.util.remove_placement_group"):
            asyncio.run(manager._scale_in(request))

            assert request.status is ScaleInStatus.COMPLETED
            assert request.selected_engines == ["group_1_engine_0"]
            assert first_group.lifecycle_status is EngineGroupLifecycle.REMOVED
            assert second_group.lifecycle_status is EngineGroupLifecycle.DRAINING
            second_engine.unregister_from_router.remote.assert_not_called()

            manager._handle_evictions([("default", second_group, 0)])

        assert second_group.lifecycle_status is EngineGroupLifecycle.REMOVED
        assert first_engine.unregister_from_router.remote.call_count == 1
        assert second_engine.unregister_from_router.remote.call_count == 1

    def test_eviction_fence_stays_closed_after_weight_update_timeout(self, patch_ray_get):
        engine = make_mock_engine()
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        group.pg = (MagicMock(), [], [])
        server = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": server})
        manager._training_weight_updating = True

        with patch("relax.distributed.ray.rollout.time.monotonic", side_effect=[0.0, 91.0]):
            manager._handle_evictions([("default", group, 0)])

        assert group.lifecycle_status is EngineGroupLifecycle.DRAINING
        assert manager.set_weight_updating(True) is False
        engine.unregister_from_router.remote.assert_not_called()

    def test_duplicate_eviction_has_single_removal_owner(self, patch_ray_get):
        engine = make_mock_engine()
        group = make_engine_group(engines=[engine], is_scaled_out=True)
        group.pg = (MagicMock(), [], [])
        server = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": server})
        manager.args.scale_in_drain_timeout = 0

        manager._handle_evictions([("default", group, 0)])
        manager._handle_evictions([("default", group, 0)])

        engine.unregister_from_router.remote.assert_called_once()
        engine.unregister_dcs.remote.assert_called_once()
        engine.shutdown.remote.assert_called_once()

    def test_marks_intentionally_removed(self, patch_ray_get):
        e = make_mock_engine()
        g = make_engine_group(engines=[e], is_scaled_out=True)
        g.pg = (MagicMock(), [], [])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})
        manager.args.scale_in_drain_timeout = 0

        mock_monitor = MagicMock()
        mock_monitor._engine_group = g
        manager._health_monitors.append(mock_monitor)

        with patch("ray.kill"):
            manager._handle_evictions([("default", g, 0)])

        mock_monitor.mark_intentionally_removed.assert_called_with(0)

    def test_sets_engine_to_none(self, patch_ray_get):
        e = make_mock_engine()
        g = make_engine_group(engines=[e], is_scaled_out=True)
        g.pg = (MagicMock(), [], [])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})
        manager.args.scale_in_drain_timeout = 0

        with patch("ray.kill"):
            manager._handle_evictions([("default", g, 0)])

        assert g.all_engines[0] is None

    def test_cleans_up_empty_groups(self, patch_ray_get):
        e = make_mock_engine()
        g = make_engine_group(engines=[e], is_scaled_out=True)
        g.pg = (MagicMock(), [], [])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})
        manager.args.scale_in_drain_timeout = 0

        with patch("ray.kill"):
            manager._handle_evictions([("default", g, 0)])

        # Group should be cleaned up since it's now empty
        assert len(srv.engine_groups) == 0


# ======================== get_engines_info =================================


class TestGetEnginesInfo:
    def test_basic_info(self, patch_ray_get):
        e1 = make_mock_engine(url="http://a:1")
        e2 = make_mock_engine(url="http://b:2")
        g = make_engine_group(engines=[e1, e2], num_gpus_per_engine=2)
        srv = make_rollout_server(
            engine_groups=[g],
            router_ip="10.0.0.1",
            router_port=3000,
        )
        manager = create_test_manager(servers={"default": srv})

        info = manager.get_engines_info()
        assert info["total_engines"] == 2
        assert "default" in info["models"]
        model = info["models"]["default"]
        assert model["router_ip"] == "10.0.0.1"
        assert len(model["engine_groups"]) == 1

    def test_dead_engines_show_dead_status(self, patch_ray_get):
        e1 = make_mock_engine()
        g = make_engine_group(engines=[e1, None])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        info = manager.get_engines_info()
        engines = info["models"]["default"]["engine_groups"][0]["engines"]
        assert engines[0]["status"] == "active"
        assert engines[1]["status"] == "dead"

    def test_filter_by_model(self, patch_ray_get):
        e1 = make_mock_engine()
        g1 = make_engine_group(engines=[e1])
        srv1 = make_rollout_server(engine_groups=[g1])
        g2 = make_engine_group(engines=[make_mock_engine()])
        srv2 = make_rollout_server(engine_groups=[g2])
        manager = create_test_manager(servers={"actor": srv1, "reward": srv2})

        info = manager.get_engines_info(model_name="actor")
        assert "actor" in info["models"]
        assert "reward" not in info["models"]

    def test_multiple_groups(self, patch_ray_get):
        g1 = make_engine_group(
            engines=[make_mock_engine()],
            worker_type="prefill",
        )
        g2 = make_engine_group(
            engines=[make_mock_engine(), make_mock_engine()],
            worker_type="decode",
            rank_offset=1,
        )
        srv = make_rollout_server(engine_groups=[g1, g2])
        manager = create_test_manager(servers={"default": srv})

        info = manager.get_engines_info()
        groups = info["models"]["default"]["engine_groups"]
        assert len(groups) == 2
        assert groups[0]["worker_type"] == "prefill"
        assert groups[1]["worker_type"] == "decode"
        assert info["total_engines"] == 3


# ==================== set_weight_updating ==================================


class TestSetWeightUpdating:
    def test_manager_owns_flag_without_signal_handler_rpc(self, patch_ray_get):
        e1 = make_mock_engine()
        e2 = make_mock_engine()
        g = make_engine_group(engines=[e1, e2])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        assert manager.set_weight_updating(True) is True
        assert manager._is_weight_updating is True
        e1.set_weight_updating.remote.assert_not_called()
        e2.set_weight_updating.remote.assert_not_called()

    def test_unsets_flag(self, patch_ray_get):
        e1 = make_mock_engine()
        g = make_engine_group(engines=[e1])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        manager.set_weight_updating(True)
        manager.set_weight_updating(False)
        assert manager._is_weight_updating is False

    def test_skips_dead_engines(self, patch_ray_get):
        g = make_engine_group(engines=[None, make_mock_engine()])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        # Should not raise
        manager.set_weight_updating(True)
