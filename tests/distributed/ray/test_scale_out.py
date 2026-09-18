# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for scale-out request creation, idempotency, and cancellation."""

import pytest


try:
    from relax.distributed.ray.rollout import (
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
    make_mock_args,
    make_mock_engine,
    make_rollout_server,
)


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def test_engine_rank_allocator_never_reuses_removed_rank():
    group = make_engine_group(engines=[make_mock_engine()], rank_offset=4)
    manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})

    first = manager._reserve_engine_ranks(1)
    group.all_engines[0] = None
    second = manager._reserve_engine_ranks(1)

    assert (first, second) == (5, 6)


# ==================== create_scale_out_request =============================


class TestCreateScaleOutRequestRayNative:
    """Tests for ray_native mode (num_replicas > 0)."""

    def test_basic_creation(self, patch_ray_get):
        e = make_mock_engine()
        g = make_engine_group(engines=[e, e])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(num_replicas=4)

        assert result["status"] == "PENDING"
        assert "request_id" in result
        # delta = 4 - 2 = 2
        assert result["num_replicas"] == 2

    def test_idempotency_at_target(self, patch_ray_get):
        """When effective_current >= target, return NOOP."""
        engines = [make_mock_engine() for _ in range(4)]
        g = make_engine_group(engines=engines)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(num_replicas=4)
        assert result["status"] == "NOOP"

    def test_idempotency_above_target(self, patch_ray_get):
        engines = [make_mock_engine() for _ in range(5)]
        g = make_engine_group(engines=engines)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(num_replicas=3)
        assert result["status"] == "NOOP"

    def test_in_flight_blocks_new_create(self, patch_ray_get):
        """A non-terminal in-flight request blocks new creates (mutual
        exclusion)."""
        engines = [make_mock_engine() for _ in range(2)]
        g = make_engine_group(engines=engines)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        manager._scale_out_requests["inflight"] = ScaleOutRequest(
            request_id="inflight",
            status=ScaleOutStatus.CREATING,
            model_name="default",
            num_replicas=2,
        )
        result = manager.create_scale_out_request(num_replicas=4)
        assert result["status"] == "CONFLICT"

    def test_terminal_in_flight_does_not_block(self, patch_ray_get):
        """A terminal (ACTIVE) in-flight request does NOT block new creates."""
        engines = [make_mock_engine() for _ in range(2)]
        g = make_engine_group(engines=engines)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        manager._scale_out_requests["done"] = ScaleOutRequest(
            request_id="done",
            status=ScaleOutStatus.ACTIVE,
            model_name="default",
            num_replicas=1,
        )
        # ACTIVE is terminal -> no conflict, new request proceeds
        result = manager.create_scale_out_request(num_replicas=5)
        assert result["status"] == "PENDING"

    def test_request_stored_in_dict(self, patch_ray_get):
        engines = [make_mock_engine()]
        g = make_engine_group(engines=engines)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(num_replicas=3)
        assert result["request_id"] in manager._scale_out_requests

    def test_invalid_model(self, patch_ray_get):
        manager = create_test_manager(servers={})
        with pytest.raises(ValueError, match="not found"):
            manager.create_scale_out_request(model_name="nonexistent", num_replicas=2)

    def test_custom_timeout(self, patch_ray_get):
        engines = [make_mock_engine()]
        g = make_engine_group(engines=engines)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(num_replicas=3, timeout_secs=999)
        req = manager._scale_out_requests[result["request_id"]]
        assert req.timeout_secs == 999


class TestCreateScaleOutRequestLogicalCounting:
    """``current_total`` in ray_native mode must use the *logical* engine 口径
    (node-0 engines, excluding dead ``None`` slots) so it matches
    ``_fetch_engines`` and scale-in.

    The physical ``len(all_engines)``口径 over-counts non-node-0 workers and dead
    slots -> false NOOP.
    """

    def test_dead_slot_not_counted(self, patch_ray_get):
        """A single-node group with a dead ``None`` slot must count only the
        live logical engine (1), so target=2 yields a real delta, not NOOP."""
        g = make_engine_group(engines=[make_mock_engine(), None])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(num_replicas=2)

        assert result["status"] == "PENDING"
        assert result["num_replicas"] == 1  # target 2 - current 1

    def test_multinode_counts_logical_engines_only(self, patch_ray_get):
        """A multi-node group (nodes_per_engine=2) holds 4 physical actors for
        2 logical engines.

        current must be 2, so target=3 -> delta 1 (not NOOP as the physical口径
        len(all_engines)==4 would produce).
        """
        args = make_mock_args(num_gpus_per_engine=16, num_gpus_per_node=8)
        engines = [make_mock_engine() for _ in range(4)]  # 2 logical x 2 nodes
        g = make_engine_group(args=args, engines=engines, num_gpus_per_engine=16)
        assert g.nodes_per_engine == 2
        assert len(g.engines) == 2
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(args=args, servers={"default": srv})

        result = manager.create_scale_out_request(num_replicas=3)

        assert result["status"] == "PENDING"
        assert result["num_replicas"] == 1  # target 3 - current 2


class TestCreateScaleOutRequestExternal:
    """Tests for external mode (engine_urls provided)."""

    def test_basic_creation(self, patch_ray_get):
        g = make_engine_group(engines=[make_mock_engine(url="http://a:1")])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(
            engine_urls=["http://b:2", "http://c:3"],
        )
        assert result["status"] == "PENDING"
        req = manager._scale_out_requests[result["request_id"]]
        # URLs should be normalized (scheme stripped)
        assert set(req.engine_urls) == {"b:2", "c:3"}

    def test_idempotency_all_known(self, patch_ray_get):
        """All provided URLs already exist -> NOOP."""
        g = make_engine_group(engines=[make_mock_engine(url="http://a:1")])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(engine_urls=["http://a:1"])
        assert result["status"] == "NOOP"

    def test_idempotency_partial_filter(self, patch_ray_get):
        """Some URLs already known, only new ones proceed."""
        g = make_engine_group(engines=[make_mock_engine(url="http://a:1")])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(
            engine_urls=["http://a:1", "http://b:2"],
        )
        assert result["status"] == "PENDING"
        req = manager._scale_out_requests[result["request_id"]]
        assert req.engine_urls == ["b:2"]

    def test_in_flight_urls_blocked(self, patch_ray_get):
        """A non-terminal in-flight request blocks new external creates."""
        g = make_engine_group(engines=[make_mock_engine(url="http://a:1")])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})
        manager._scale_out_requests["inflight"] = ScaleOutRequest(
            request_id="inflight",
            status=ScaleOutStatus.CONNECTING,
            model_name="default",
            engine_urls=["b:2"],
        )

        result = manager.create_scale_out_request(
            engine_urls=["http://b:2", "http://c:3"],
        )
        assert result["status"] == "CONFLICT"

    def test_scheme_normalization(self, patch_ray_get):
        """http:// and bare host:port are treated identically."""
        g = make_engine_group(engines=[make_mock_engine(url="http://x:1")])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager.create_scale_out_request(engine_urls=["x:1"])
        assert result["status"] == "NOOP"


class TestCreateScaleOutRequestErrors:
    def test_neither_replicas_nor_urls(self, patch_ray_get):
        g = make_engine_group()
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})
        with pytest.raises(ValueError, match="Either"):
            manager.create_scale_out_request()

    def test_zero_replicas_empty_urls(self, patch_ray_get):
        g = make_engine_group()
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})
        with pytest.raises(ValueError):
            manager.create_scale_out_request(num_replicas=0, engine_urls=[])


class TestCreateScaleOutRequestMutualExclusion:
    """Mutual exclusion: reject when another scale operation is active."""

    def test_conflict_with_active_scale_out(self, patch_ray_get):
        g = make_engine_group()
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        manager._scale_out_requests["existing"] = ScaleOutRequest(
            request_id="existing",
            status=ScaleOutStatus.CREATING,
            model_name="default",
            num_replicas=2,
        )

        result = manager.create_scale_out_request(num_replicas=4)
        assert result["status"] == "CONFLICT"

    def test_conflict_with_active_scale_in(self, patch_ray_get):
        g = make_engine_group()
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        manager._scale_in_requests["existing"] = ScaleInRequest(
            request_id="existing",
            status=ScaleInStatus.DRAINING,
            model_name="default",
        )

        result = manager.create_scale_out_request(num_replicas=4)
        assert result["status"] == "CONFLICT"

    def test_no_conflict_with_terminal_requests(self, patch_ray_get):
        """Terminal requests don't block new ones."""
        engines = [make_mock_engine()]
        g = make_engine_group(engines=engines)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        manager._scale_out_requests["done"] = ScaleOutRequest(
            request_id="done",
            status=ScaleOutStatus.ACTIVE,
            model_name="default",
        )
        manager._scale_in_requests["done2"] = ScaleInRequest(
            request_id="done2",
            status=ScaleInStatus.COMPLETED,
        )

        result = manager.create_scale_out_request(num_replicas=3)
        assert result["status"] == "PENDING"


# ======================== cancel_scale_out =================================


class TestCancelScaleOut:
    def test_cancel_pending(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.PENDING,
        )
        result = manager.cancel_scale_out("r1")
        assert result["status"] == "CANCELLED"

    def test_cancel_creating(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.CREATING,
        )
        result = manager.cancel_scale_out("r1")
        assert result["status"] == "CANCELLED"

    def test_cannot_cancel_active(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.ACTIVE,
        )
        result = manager.cancel_scale_out("r1")
        assert "error" in result

    def test_cannot_cancel_health_checking(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.HEALTH_CHECKING,
        )
        result = manager.cancel_scale_out("r1")
        assert "error" in result

    def test_not_found(self):
        manager = create_test_manager()
        assert manager.cancel_scale_out("nonexistent") is None


# =================== cancel_all_scale_out_requests =========================


class TestCancelAllScaleOutRequests:
    @pytest.mark.asyncio
    async def test_dry_run(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.PENDING,
        )
        manager._scale_out_requests["r2"] = ScaleOutRequest(
            request_id="r2",
            status=ScaleOutStatus.ACTIVE,
        )

        result = await manager.cancel_all_scale_out_requests(dry_run=True)
        assert "r1" in result["succeeded"]
        assert len(result["skipped"]) == 1
        assert result["dry_run"] is True
        # Dry run should NOT change actual status
        assert manager._scale_out_requests["r1"].status == ScaleOutStatus.PENDING

    @pytest.mark.asyncio
    async def test_actual_cancel(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.PENDING,
        )
        manager._scale_out_requests["r2"] = ScaleOutRequest(
            request_id="r2",
            status=ScaleOutStatus.CREATING,
        )

        result = await manager.cancel_all_scale_out_requests()
        assert set(result["succeeded"]) == {"r1", "r2"}
        assert manager._scale_out_requests["r1"].status == ScaleOutStatus.CANCELLED
        assert manager._scale_out_requests["r2"].status == ScaleOutStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_filter_by_status(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.PENDING,
        )
        manager._scale_out_requests["r2"] = ScaleOutRequest(
            request_id="r2",
            status=ScaleOutStatus.CREATING,
        )

        result = await manager.cancel_all_scale_out_requests(status_filter="PENDING")
        assert result["succeeded"] == ["r1"]
        # r2 not touched
        assert manager._scale_out_requests["r2"].status == ScaleOutStatus.CREATING

    @pytest.mark.asyncio
    async def test_filter_by_model(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.PENDING,
            model_name="actor",
        )
        manager._scale_out_requests["r2"] = ScaleOutRequest(
            request_id="r2",
            status=ScaleOutStatus.PENDING,
            model_name="reward",
        )

        result = await manager.cancel_all_scale_out_requests(model_name="actor")
        assert result["succeeded"] == ["r1"]
        assert manager._scale_out_requests["r2"].status == ScaleOutStatus.PENDING

    @pytest.mark.asyncio
    async def test_invalid_status_filter(self):
        manager = create_test_manager()
        with pytest.raises(ValueError, match="Invalid status"):
            await manager.cancel_all_scale_out_requests(status_filter="BOGUS")


# ==================== get_scale_out_status / list ==========================


class TestScaleOutStatusQueries:
    def test_get_status_found(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.ACTIVE,
        )
        result = manager.get_scale_out_status("r1")
        assert result is not None
        assert result["status"] == "ACTIVE"

    def test_get_status_not_found(self):
        manager = create_test_manager()
        assert manager.get_scale_out_status("nope") is None

    def test_list_all(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.ACTIVE,
        )
        manager._scale_out_requests["r2"] = ScaleOutRequest(
            request_id="r2",
            status=ScaleOutStatus.FAILED,
        )
        result = manager.list_all_scale_out_requests()
        assert len(result) == 2

    def test_list_filter_by_status(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.ACTIVE,
        )
        manager._scale_out_requests["r2"] = ScaleOutRequest(
            request_id="r2",
            status=ScaleOutStatus.FAILED,
        )
        result = manager.list_all_scale_out_requests(status_filter="ACTIVE")
        assert len(result) == 1
        assert result[0]["status"] == "ACTIVE"

    def test_list_sorted_descending(self):
        manager = create_test_manager()
        import time

        r1 = ScaleOutRequest(request_id="r1", status=ScaleOutStatus.ACTIVE)
        time.sleep(0.01)
        r2 = ScaleOutRequest(request_id="r2", status=ScaleOutStatus.ACTIVE)
        manager._scale_out_requests["r1"] = r1
        manager._scale_out_requests["r2"] = r2

        result = manager.list_all_scale_out_requests()
        assert result[0]["request_id"] == "r2"  # newer first

    def test_list_invalid_status(self):
        manager = create_test_manager()
        with pytest.raises(ValueError, match="Invalid status"):
            manager.list_all_scale_out_requests(status_filter="BOGUS")
