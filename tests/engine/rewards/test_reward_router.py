#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Full-pipeline tests for format-aware reward routing.

Exercises RewardExecutor -> resolve_rm_type -> RewardWorker.compute with real
Ray actors: mixed math + multiple-choice batches, registry-driven dispatch,
fallback degradations with deduplicated warnings, and exact backward
compatibility of the default (flag-free) path.

Run with: pytest tests/engine/rewards/test_reward_router.py -v
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest


_HAS_FULL_PIPELINE = False

try:
    import ray

    if not ray.is_initialized():
        try:
            ray.init(num_cpus=8, ignore_reinit_error=True)
        except ValueError as e:
            if "When connecting to an existing cluster" in str(e):
                # Already connected to an existing cluster
                ray.init(ignore_reinit_error=True)
            else:
                raise

    from relax.engine.rewards import RewardExecutor, async_rm, batched_async_rm
    from relax.engine.rewards import registry as reward_registry
    from relax.utils.types import Sample

    _HAS_FULL_PIPELINE = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not _HAS_FULL_PIPELINE, reason="Full pipeline requires Ray + relax imports")


def _make_args(**overrides) -> SimpleNamespace:
    """Mock args namespace with the same six keys existing tests use.

    New router flags are intentionally absent from the defaults so these tests
    double as a guard that the executor reads them with getattr.
    """
    defaults = {
        "rm_type": "openr1mm",
        "custom_rm_path": None,
        "reward_max_concurrency": 64,
        "reward_num_workers": 4,
        "rm_url": None,
        "reward_key": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_sample(response: str, label: str, metadata: dict | None = None) -> "Sample":
    return Sample(response=response, label=label, metadata=metadata or {})


def _kill_executor_workers():
    """Kill named Ray actors held by the current RewardExecutor singleton.

    Without this, dropping ``RewardExecutor._instance`` only releases the
    Python actor handles; on Python 3.10 the next test can reach
    ``options(get_if_exists=True)`` before Ray finishes evicting the named
    actors, get a handle to a dying actor, and hit ActorDiedError.
    """
    inst = RewardExecutor._instance
    if inst is None:
        return
    for w in inst._workers:
        try:
            ray.kill(w)
        except Exception:
            pass
    inst._workers = []


@pytest.fixture(autouse=True)
def _preconfigure_logging():
    """Run the one-time root-logger configuration before any in-test warning so
    pytest's caplog handler survives the test call phase."""
    from relax.utils.logging_utils import configure_logger

    configure_logger()
    yield


@pytest.fixture(autouse=True)
def _reset_state():
    """Fresh executor singleton, clean warn/counter state, and a registry
    snapshot so in-test registrations do not leak across tests."""
    _kill_executor_workers()
    RewardExecutor._instance = None
    saved = dict(reward_registry._REWARDS)
    reward_registry.reset_router_state()
    yield
    _kill_executor_workers()
    RewardExecutor._instance = None
    reward_registry._REWARDS.clear()
    reward_registry._REWARDS.update(saved)
    reward_registry.reset_router_state()


class TestMixedBatchRouting:
    """Per-sample routing of mixed-format batches (the flagship case)."""

    @pytest.mark.asyncio
    async def test_mixed_batch_math_mc_exact_values(self):
        args = _make_args(rm_type="math")
        samples = [
            _make_sample("The answer is \\boxed{7}", "7"),
            _make_sample("The answer is \\boxed{8}", "7"),
            _make_sample("<answer>A</answer>", "<answer>A</answer>", metadata={"rm_type": "multiple_choice"}),
            _make_sample("<answer>B</answer>", "<answer>A</answer>", metadata={"rm_type": "multiple_choice"}),
        ]
        rewards = await asyncio.wait_for(batched_async_rm(args, samples), timeout=60)
        assert rewards == [1.0, 0.0, 1.0, 0.0]
        assert reward_registry.get_router_stats() == {}

    @pytest.mark.asyncio
    async def test_registered_reward_routed_end_to_end(self):
        """One registration line routes samples; zero router-code changes."""

        async def t18_half(args, sample):
            return 0.5

        reward_registry.register_reward("t18_half", t18_half, mode="async")
        args = _make_args(rm_type="math")
        samples = [
            _make_sample("The answer is \\boxed{7}", "7"),
            _make_sample("anything", "anything", metadata={"rm_type": "t18_half"}),
            _make_sample("<answer>B</answer>", "<answer>A</answer>", metadata={"rm_type": "multiple_choice"}),
        ]
        rewards = await asyncio.wait_for(batched_async_rm(args, samples), timeout=60)
        assert rewards == [1.0, 0.5, 0.0]

    @pytest.mark.asyncio
    async def test_label_inference_routes_mixed_batch(self):
        """With --rm-type-infer, unlabeled samples route by label format."""
        args = _make_args(rm_type=None, rm_type_infer=True)
        samples = [
            _make_sample("The answer is \\boxed{7}", "7"),
            _make_sample("<answer>A</answer>", "<answer>A</answer>"),
        ]
        rewards = await asyncio.wait_for(batched_async_rm(args, samples), timeout=60)
        assert rewards == [1.0, 1.0]
        assert reward_registry.get_router_stats() == {}


class TestBackCompatDefaults:
    """Flag-free behavior must be bit-identical to the pre-registry
    baseline."""

    @pytest.mark.asyncio
    async def test_missing_rm_type_exact_message(self):
        args = _make_args(rm_type=None)
        sample = _make_sample("foo", "bar")
        with pytest.raises(NotImplementedError) as excinfo:
            await async_rm(args, sample)
        assert str(excinfo.value) == "Rule-based RM type is not specified."

    @pytest.mark.asyncio
    async def test_unknown_rm_type_still_raises_with_type_name(self):
        args = _make_args(rm_type="totally_unknown_type")
        sample = _make_sample("foo", "bar")
        with pytest.raises(NotImplementedError, match="totally_unknown_type"):
            await async_rm(args, sample)

    @pytest.mark.asyncio
    async def test_boxed_prefix_preserved(self):
        args = _make_args(rm_type="math")
        sample = _make_sample("The answer is \\boxed{7}", "7", metadata={"rm_type": "boxed_openr1mm"})
        result = await async_rm(args, sample)
        assert result == 1.0

    def test_all_legacy_types_registered(self):
        legacy_sync = [
            "deepscaler",
            "geo3k",
            "openr1mm",
            "multiple_choice",
            "dapo",
            "math",
            "mopd",
            "f1",
            "gpqa",
            "ifbench",
            "random",
        ]
        for name in legacy_sync:
            spec = reward_registry.get_reward_spec(name)
            assert spec is not None and spec.mode == "sync", name
        for name in ["remote_rm", "dapo-genrm", "dummy"]:
            spec = reward_registry.get_reward_spec(name)
            assert spec is not None and spec.mode == "async", name


class TestFallbackDegradations:
    """Unknown/missing types degrade per --rm-type-fallback with warnings."""

    @pytest.mark.asyncio
    async def test_unknown_falls_back_with_single_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=reward_registry.logger.name)
        args = _make_args(rm_type="math", rm_type_fallback="math")
        samples = [_make_sample("The answer is \\boxed{7}", "7", metadata={"rm_type": "mystery"}) for _ in range(3)]
        rewards = await asyncio.wait_for(batched_async_rm(args, samples), timeout=60)
        assert rewards == [1.0, 1.0, 1.0]
        assert reward_registry.get_router_stats() == {("unknown", "mystery"): 3}
        mystery_records = [r for r in caplog.records if "mystery" in r.message]
        assert len(mystery_records) == 1

    @pytest.mark.asyncio
    async def test_missing_falls_back(self):
        args = _make_args(rm_type=None, rm_type_fallback="math")
        sample = _make_sample("The answer is \\boxed{7}", "7")
        result = await async_rm(args, sample)
        assert result == 1.0
        assert reward_registry.get_router_stats() == {("missing", ""): 1}

    @pytest.mark.asyncio
    async def test_mixed_batch_with_zero_fallback_shape_safe(self):
        args = _make_args(rm_type="math", rm_type_fallback="zero")
        samples = [
            _make_sample("The answer is \\boxed{7}", "7"),
            _make_sample("The answer is \\boxed{8}", "7"),
            _make_sample("<answer>A</answer>", "<answer>A</answer>", metadata={"rm_type": "multiple_choice"}),
            _make_sample("<answer>B</answer>", "<answer>A</answer>", metadata={"rm_type": "multiple_choice"}),
            _make_sample("no type at all", "unparseable label", metadata={"rm_type": "mystery"}),
        ]
        rewards = await asyncio.wait_for(batched_async_rm(args, samples), timeout=60)
        assert len(rewards) == 5
        assert all(isinstance(r, (int, float)) for r in rewards)
        assert rewards == [1.0, 0.0, 1.0, 0.0, 0.0]
        assert reward_registry.get_router_stats() == {("unknown", "mystery"): 1}
