# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for weight synchronization: seed engine selection, single engine sync,
parallel sync with fallback, and validate_seed_engine."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


try:
    from relax.distributed.ray.rollout import ScaleOutStatus, _WeightSyncTimeoutError  # noqa

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


# =================== _get_healthy_seed_engines =============================


class TestGetHealthySeedEngines:
    def test_prefers_initial_engines(self, patch_ray_get):
        """Initial (non-scaled-out) engines come first in the list."""
        e_init = make_mock_engine(weight_version="v1")
        e_scaled = make_mock_engine(weight_version="v1")
        g_init = make_engine_group(engines=[e_init])
        g_scaled = make_engine_group(engines=[e_scaled], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g_init, g_scaled])
        manager = create_test_manager(servers={"default": srv})

        candidates = manager._get_healthy_seed_engines("default")
        assert len(candidates) == 2
        assert candidates[0] is e_init

    def test_filters_default_version(self, patch_ray_get):
        """Engines with weight_version='default' are excluded."""
        e1 = make_mock_engine(weight_version="default")
        e2 = make_mock_engine(weight_version="v1")
        g = make_engine_group(engines=[e1, e2])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        candidates = manager._get_healthy_seed_engines("default")
        assert len(candidates) == 1
        assert candidates[0] is e2

    def test_filters_none_version(self, patch_ray_get):
        e1 = make_mock_engine(weight_version=None)
        g = make_engine_group(engines=[e1])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        candidates = manager._get_healthy_seed_engines("default")
        assert len(candidates) == 0

    def test_handles_dead_engines(self, patch_ray_get):
        g = make_engine_group(engines=[None, None])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        candidates = manager._get_healthy_seed_engines("default")
        assert len(candidates) == 0

    def test_handles_exception(self, patch_ray_get):
        e1 = make_mock_engine()
        e1.get_weight_version.remote.side_effect = Exception("dead")
        g = make_engine_group(engines=[e1])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        candidates = manager._get_healthy_seed_engines("default")
        assert len(candidates) == 0

    def test_model_not_found(self, patch_ray_get):
        manager = create_test_manager(servers={})
        assert manager._get_healthy_seed_engines("default") == []

    def test_get_healthy_seed_engine_delegates(self, patch_ray_get):
        """The singular version returns the first candidate."""
        e1 = make_mock_engine(weight_version="v1")
        g = make_engine_group(engines=[e1])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = manager._get_healthy_seed_engine("default")
        assert result is e1

    def test_get_healthy_seed_engine_returns_none(self, patch_ray_get):
        manager = create_test_manager(servers={})
        assert manager._get_healthy_seed_engine("default") is None


# =================== _validate_seed_engine =================================


class TestValidateSeedEngine:
    @pytest.mark.asyncio
    async def test_valid_seed(self):
        engine = make_mock_engine(
            url="http://10.0.0.1:30000",
            weight_version="v1",
        )
        manager = create_test_manager()
        result = await manager._validate_seed_engine(engine)
        assert result is not None
        version, addr = result
        assert version == "v1"
        assert addr == "10.0.0.1"

    @pytest.mark.asyncio
    async def test_invalid_weight_version_default(self):
        engine = make_mock_engine(weight_version="default")
        manager = create_test_manager()
        result = await manager._validate_seed_engine(engine)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_weight_version_none(self):
        engine = make_mock_engine(weight_version=None)
        manager = create_test_manager()
        result = await manager._validate_seed_engine(engine)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_url(self):
        engine = make_mock_engine(url=None, weight_version="v1")
        manager = create_test_manager()
        result = await manager._validate_seed_engine(engine)
        assert result is None

    @pytest.mark.asyncio
    async def test_exception_on_get_version(self):
        engine = make_mock_engine(weight_version="v1")
        engine.get_weight_version.remote.side_effect = Exception("dead")
        manager = create_test_manager()
        result = await manager._validate_seed_engine(engine)
        assert result is None


# ================= _sync_single_engine_weights ==============================


class TestSyncSingleEngineWeights:
    @pytest.mark.asyncio
    async def test_successful_sync(self, patch_async_helpers):
        seed = make_mock_engine()
        new = make_mock_engine()
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
        seed.init_weights_send_group_for_remote_instance.remote.assert_called_once()
        new.init_weights_send_group_for_remote_instance.remote.assert_called_once()
        seed.send_weights_to_remote_instance.remote.assert_called_once()
        new.send_weights_to_remote_instance.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_nccl_init_failure(self, patch_async_helpers):
        seed = make_mock_engine()
        new = make_mock_engine()
        # Simulate NCCL init failure on the new engine
        new.init_weights_send_group_for_remote_instance.remote.return_value = AwaitableValue(
            {"success": False, "message": "NCCL error"}
        )
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
    async def test_send_weights_failure(self, patch_async_helpers):
        seed = make_mock_engine()
        new = make_mock_engine()
        new.send_weights_to_remote_instance.remote.return_value = AwaitableValue(
            {"success": False, "message": "send failed"}
        )
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
    async def test_exception_returns_false(self, patch_async_helpers):
        seed = make_mock_engine()
        new = make_mock_engine()
        seed.init_weights_send_group_for_remote_instance.remote.side_effect = RuntimeError("actor dead")
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
    async def test_init_timeout_does_not_retry(self, patch_async_helpers):
        seed = make_mock_engine()
        new = make_mock_engine()

        def hang(**kwargs):
            return asyncio.get_running_loop().create_future()

        seed.init_weights_send_group_for_remote_instance.remote.side_effect = hang
        new.init_weights_send_group_for_remote_instance.remote.side_effect = hang
        manager = create_test_manager()

        with pytest.raises(TimeoutError):
            await manager._sync_single_engine_weights(
                seed_engine=seed,
                new_engine=new,
                engine_index=0,
                total_engines=1,
                master_address="10.0.0.1",
                tp_size=1,
                timeout=0.01,
            )

        seed.init_weights_send_group_for_remote_instance.remote.assert_called_once()
        new.init_weights_send_group_for_remote_instance.remote.assert_called_once()
        seed.send_weights_to_remote_instance.remote.assert_not_called()
        new.send_weights_to_remote_instance.remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_timeout_does_not_retry(self, patch_async_helpers):
        seed = make_mock_engine()
        new = make_mock_engine()

        def hang(**kwargs):
            return asyncio.get_running_loop().create_future()

        seed.send_weights_to_remote_instance.remote.side_effect = hang
        new.send_weights_to_remote_instance.remote.side_effect = hang
        manager = create_test_manager()

        with pytest.raises(TimeoutError):
            await manager._sync_single_engine_weights(
                seed_engine=seed,
                new_engine=new,
                engine_index=0,
                total_engines=1,
                master_address="10.0.0.1",
                tp_size=1,
                timeout=0.01,
            )

        seed.init_weights_send_group_for_remote_instance.remote.assert_called_once()
        new.init_weights_send_group_for_remote_instance.remote.assert_called_once()
        seed.send_weights_to_remote_instance.remote.assert_called_once()
        new.send_weights_to_remote_instance.remote.assert_called_once()


# ================ _sync_weights_from_seed_engine ===========================


class TestSyncWeightsFromSeedEngine:
    @pytest.mark.asyncio
    async def test_empty_engines_returns_true(self, patch_async_helpers):
        manager = create_test_manager()
        ok = await manager._sync_weights_from_seed_engine([], timeout=10)
        assert ok is True

    @pytest.mark.asyncio
    async def test_no_seed_candidates_returns_false(self, patch_async_helpers):
        # No servers -> no seed candidates
        manager = create_test_manager(servers={})
        new_engine = make_mock_engine()
        ok = await manager._sync_weights_from_seed_engine([new_engine], timeout=10)
        assert ok is False

    @pytest.mark.asyncio
    async def test_timeout_does_not_fall_back_to_another_seed(self, patch_async_helpers):
        seed1 = make_mock_engine(url="http://seed1:1", weight_version="v1")
        seed2 = make_mock_engine(url="http://seed2:1", weight_version="v1")
        new = make_mock_engine()
        group = make_engine_group(engines=[seed1, seed2])
        manager = create_test_manager(servers={"default": make_rollout_server(engine_groups=[group])})

        with patch.object(
            manager,
            "_sync_single_engine_weights",
            new_callable=AsyncMock,
            side_effect=_WeightSyncTimeoutError("remote operation timed out"),
        ) as sync:
            ok = await manager._sync_weights_from_seed_engine([new], timeout=60)

        assert ok is False
        sync.assert_awaited_once()
        manager._weight_sync_lock.release.remote.assert_called_once()
        new.continue_generation.remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_parallel_sync(self, patch_async_helpers):
        """All engines sync successfully from a single seed."""
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        g = make_engine_group(engines=[seed])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        new_engines = [make_mock_engine() for _ in range(3)]
        ok = await manager._sync_weights_from_seed_engine(
            new_engines,
            timeout=60,
            model_name="default",
        )
        assert ok is True
        # All new engines should have had flush_cache called
        for e in new_engines:
            e.flush_cache.remote.assert_called()

    @pytest.mark.asyncio
    async def test_seed_fallback_on_failure(self, patch_async_helpers):
        """When the primary seed fails, the next candidate is tried."""
        # First seed: valid but sync will fail
        bad_seed = make_mock_engine(url="http://bad:1", weight_version="v1")
        bad_seed.init_weights_send_group_for_remote_instance.remote.return_value = AwaitableValue(
            {"success": False, "message": "NCCL error"}
        )
        # Second seed: good
        good_seed = make_mock_engine(url="http://good:2", weight_version="v1")

        g = make_engine_group(engines=[bad_seed, good_seed])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        new_engines = [make_mock_engine()]
        ok = await manager._sync_weights_from_seed_engine(
            new_engines,
            timeout=60,
            model_name="default",
        )
        assert ok is True

    @pytest.mark.asyncio
    async def test_all_seeds_fail(self, patch_async_helpers):
        """When all seed candidates fail, sync returns False."""
        bad_seed = make_mock_engine(url="http://bad:1", weight_version="v1")
        bad_seed.init_weights_send_group_for_remote_instance.remote.return_value = AwaitableValue(
            {"success": False, "message": "err"}
        )
        g = make_engine_group(engines=[bad_seed])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        new_engines = [make_mock_engine()]
        ok = await manager._sync_weights_from_seed_engine(
            new_engines,
            timeout=60,
            model_name="default",
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_none_engines_filtered(self, patch_async_helpers):
        """None entries in new_engines are skipped."""
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        g = make_engine_group(engines=[seed])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        ok = await manager._sync_weights_from_seed_engine(
            [None, make_mock_engine(), None],
            timeout=60,
            model_name="default",
        )
        assert ok is True

    @pytest.mark.asyncio
    async def test_all_none_engines(self, patch_async_helpers):
        """All-None engine list is treated as empty -> True."""
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        g = make_engine_group(engines=[seed])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        # The pending list after filtering None is empty, so no sync needed
        # But total_live = 0, so success_count == total_live -> True
        ok = await manager._sync_weights_from_seed_engine(
            [None, None],
            timeout=60,
            model_name="default",
        )
        assert ok is True

    @pytest.mark.asyncio
    async def test_pause_resume_on_new_engines(self, patch_async_helpers):
        """New engines are paused before sync and resumed after."""
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        g = make_engine_group(engines=[seed])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        new_engine = make_mock_engine()
        await manager._sync_weights_from_seed_engine(
            [new_engine],
            timeout=60,
            model_name="default",
        )
        new_engine.pause_generation.remote.assert_called()
        new_engine.continue_generation.remote.assert_called()

    @pytest.mark.asyncio
    async def test_failed_engine_remains_paused_for_rollback(self, patch_async_helpers):
        """A failed engine must not resume before the caller rolls it back."""
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        seed.init_weights_send_group_for_remote_instance.remote.return_value = AwaitableValue(
            {"success": False, "message": "NCCL error"}
        )
        group = make_engine_group(engines=[seed])
        server = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": server})
        new_engine = make_mock_engine()
        ok = await manager._sync_weights_from_seed_engine(
            [new_engine],
            timeout=60,
            model_name="default",
        )

        assert ok is False
        new_engine.pause_generation.remote.assert_called_once()
        new_engine.continue_generation.remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_pause_failure_prevents_weight_sync(self, patch_async_helpers):
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        group = make_engine_group(engines=[seed])
        server = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": server})
        new_engine = make_mock_engine()
        new_engine.pause_generation.remote.return_value = AwaitableValue(RuntimeError("pause failed"))

        ok = await manager._sync_weights_from_seed_engine(
            [new_engine],
            timeout=60,
            model_name="default",
        )

        assert ok is False
        seed.init_weights_send_group_for_remote_instance.remote.assert_not_called()

    @pytest.mark.asyncio
    async def test_lock_acquired_and_released(self, patch_async_helpers):
        """The weight_sync_lock is acquired and released."""
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        g = make_engine_group(engines=[seed])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        await manager._sync_weights_from_seed_engine(
            [make_mock_engine()],
            timeout=60,
            model_name="default",
        )
        manager._weight_sync_lock.acquire.remote.assert_called()
        manager._weight_sync_lock.release.remote.assert_called()

    @pytest.mark.asyncio
    async def test_scale_out_claims_lifecycle_owner_before_weight_sync_lock(self, patch_async_helpers):
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        group = make_engine_group(engines=[seed])
        server = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": server})

        def acquire_lock():
            assert manager._scale_out_weight_updating is True
            return AwaitableValue(True)

        manager._weight_sync_lock.acquire.remote.side_effect = acquire_lock

        ok = await manager._sync_weights_from_seed_engine(
            [make_mock_engine()],
            timeout=60,
            model_name="default",
        )

        assert ok is True
        assert manager._scale_out_weight_updating is False
        manager._weight_sync_lock.release.remote.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_failure_releases_weight_sync_lock(self, patch_async_helpers):
        """A sync that fails on every attempt still ALWAYS releases the shared
        weight-sync lock (P0 invariant).

        The lock is the SAME distributed lock every training weight update
        acquires; retaining it on failure would hang training forever. The
        finally block must release it unconditionally regardless of why sync
        failed.
        """
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        seed.init_weights_send_group_for_remote_instance.remote.return_value = AwaitableValue(
            {"success": False, "message": "NCCL error"}
        )
        group = make_engine_group(engines=[seed])
        server = make_rollout_server(engine_groups=[group])
        manager = create_test_manager(servers={"default": server})

        ok = await manager._sync_weights_from_seed_engine(
            [make_mock_engine()],
            timeout=60,
            model_name="default",
        )

        assert ok is False
        manager._weight_sync_lock.acquire.remote.assert_called()
        # P0 invariant: the shared training lock is released even when sync fails.
        manager._weight_sync_lock.release.remote.assert_called()


# ============= sync_weights_for_scaled_out_engines =========================


class TestSyncWeightsForScaledOutEngines:
    @pytest.mark.asyncio
    async def test_no_scaled_out_engines(self, patch_async_helpers):
        """Returns success with synced_count=0 when no scaled-out engines."""
        g = make_engine_group(engines=[make_mock_engine()], is_scaled_out=False)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        result = await manager.sync_weights_for_scaled_out_engines()
        assert result["success"] is True
        assert result["synced_count"] == 0

    @pytest.mark.asyncio
    async def test_model_not_found(self, patch_async_helpers):
        manager = create_test_manager(servers={})
        result = await manager.sync_weights_for_scaled_out_engines("missing")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_syncs_scaled_out_engines(self, patch_async_helpers):
        seed = make_mock_engine(url="http://seed:1", weight_version="v1")
        g_init = make_engine_group(engines=[seed])
        e_scaled = make_mock_engine()
        g_scaled = make_engine_group(engines=[e_scaled], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g_init, g_scaled])
        manager = create_test_manager(servers={"default": srv})

        result = await manager.sync_weights_for_scaled_out_engines()
        assert result["success"] is True
        assert result["synced_count"] == 1
