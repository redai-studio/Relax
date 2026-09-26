#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the openr1mm reward function through rm_hub's full execution path.

Tests exercise the complete reward pipeline as defined in rm_hub/__init__.py:
  RewardExecutor -> RewardWorker.compute() -> get_openr1mm_rule_based_reward()

High-concurrency tests validate correctness and stability under parallel load.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Import strategy: try to load the full rm_hub pipeline (requires Ray + math_verify).
# Falls back to loading openr1mm.py directly for environments without Ray.
# ---------------------------------------------------------------------------

_HAS_FULL_PIPELINE = False
_HAS_MATH_VERIFY = False

# Try the full pipeline first (rm_hub/__init__.py path)
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

    from relax.engine.rewards import (
        RewardExecutor,
        RewardWorker,
        async_rm,
        batched_async_rm,
    )
    from relax.engine.rewards.openr1mm import get_openr1mm_rule_based_reward
    from relax.utils.types import Sample

    _HAS_FULL_PIPELINE = True
    _HAS_MATH_VERIFY = True
except ImportError:
    # Fallback: load openr1mm.py directly for math_verify-only tests
    _MODULE_PATH = Path(__file__).resolve().parents[3] / "relax" / "engine" / "rewards" / "openr1mm.py"
    try:
        _spec = importlib.util.spec_from_file_location("openr1mm", _MODULE_PATH)
        _openr1mm = importlib.util.module_from_spec(_spec)
        sys.modules["openr1mm"] = _openr1mm
        _spec.loader.exec_module(_openr1mm)
        get_openr1mm_rule_based_reward = _openr1mm.get_openr1mm_rule_based_reward
        _HAS_MATH_VERIFY = True
    except ImportError:
        _HAS_MATH_VERIFY = False
        get_openr1mm_rule_based_reward = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(not _HAS_MATH_VERIFY, reason="math_verify is not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides) -> SimpleNamespace:
    """Create a mock args namespace that mimics the training args structure."""
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
    """Create a Sample for testing.

    Uses the real Sample class when available.
    """
    if _HAS_FULL_PIPELINE:
        return Sample(response=response, label=label, metadata=metadata or {})
    # Lightweight fallback
    return SimpleNamespace(response=response, label=label, metadata=metadata or {})


# ---------------------------------------------------------------------------
# Direct function tests (openr1mm.get_openr1mm_rule_based_reward)
# ---------------------------------------------------------------------------


class TestOpenR1MMSymbolicVerification:
    """Tests that exercise the math_verify symbolic path."""

    def test_exact_numeric_match(self):
        assert get_openr1mm_rule_based_reward("42", "42") == 1.0

    def test_boxed_answer_correct(self):
        assert get_openr1mm_rule_based_reward("The answer is \\boxed{7}", "7") == 1.0

    def test_boxed_answer_wrong(self):
        assert get_openr1mm_rule_based_reward("The answer is \\boxed{8}", "7") == 0.0

    def test_fraction_equivalence(self):
        assert get_openr1mm_rule_based_reward("\\frac{1}{2}", "0.5") == 1.0

    def test_negative_number(self):
        assert get_openr1mm_rule_based_reward("-3", "-3") == 1.0
        assert get_openr1mm_rule_based_reward("-3", "3") == 0.0


class TestOpenR1MMStringMatching:
    """Tests that exercise the string-based fallback matching."""

    def test_answer_tag_match(self):
        response = "<think>some reasoning</think>\n<answer>42</answer>"
        label = "<answer>42</answer>"
        assert get_openr1mm_rule_based_reward(response, label) == 1.0

    def test_answer_tag_mismatch(self):
        response = "<think>some reasoning</think>\n<answer>99</answer>"
        label = "<answer>42</answer>"
        assert get_openr1mm_rule_based_reward(response, label) == 0.0

    def test_answer_tag_in_response_only(self):
        assert get_openr1mm_rule_based_reward("<answer>hello</answer>", "hello") == 1.0

    def test_plain_text_exact_match(self):
        assert get_openr1mm_rule_based_reward("Paris", "Paris") == 1.0

    def test_plain_text_mismatch(self):
        assert get_openr1mm_rule_based_reward("London", "Paris") == 0.0


class TestOpenR1MMEdgeCases:
    """Edge cases and robustness checks."""

    def test_empty_response(self):
        assert get_openr1mm_rule_based_reward("", "42") == 0.0

    def test_both_empty(self):
        assert get_openr1mm_rule_based_reward("", "") == 1.0

    def test_whitespace_in_answer_tags(self):
        response = "<answer>  42  </answer>"
        label = "<answer>42</answer>"
        assert get_openr1mm_rule_based_reward(response, label) == 1.0

    def test_malformed_latex_fallback(self):
        response = "\\invalid{command}"
        label = "\\invalid{command}"
        assert get_openr1mm_rule_based_reward(response, label) == 1.0

    def test_long_response_with_boxed(self):
        response = (
            "<think>Let me think step by step. "
            "First, we compute 2+2=4. Then 4*3=12. "
            "Therefore the answer is 12.</think>\n"
            "The final answer is \\boxed{12}."
        )
        assert get_openr1mm_rule_based_reward(response, "12") == 1.0


class TestOpenR1MMRewardHacking:
    """Reject responses that repeat the <think>/<answer> structure to game the
    reward (only one think/answer pair should score)."""

    def test_repeated_answer_blocks_rejected(self):
        response = "reasoning</think>\n<answer>9</answer>\n</think>\n<answer>9</answer>\n</think>\n<answer>9</answer>"
        label = "<think>compute</think>\n<answer>9</answer>"
        assert get_openr1mm_rule_based_reward(response, label) == 0.0

    def test_repeated_close_think_rejected(self):
        response = "x</think>\ny</think>\n<answer>9</answer>"
        assert get_openr1mm_rule_based_reward(response, "9") == 0.0

    def test_repeated_open_answer_rejected(self):
        response = "<answer>9</answer><answer>9</answer>"
        assert get_openr1mm_rule_based_reward(response, "9") == 0.0

    def test_repeated_close_answer_rejected(self):
        response = "<answer>9</answer></answer>"
        assert get_openr1mm_rule_based_reward(response, "9") == 0.0

    def test_single_pair_still_scores(self):
        response = "reasoning</think>\n<answer>9</answer>"
        assert get_openr1mm_rule_based_reward(response, "9") == 1.0


# ===========================================================================
# Full-pipeline tests (rm_hub/__init__.py: RewardWorker / RewardExecutor)
# ===========================================================================

requires_full_pipeline = pytest.mark.skipif(
    not _HAS_FULL_PIPELINE,
    reason="Full pipeline requires Ray + relax imports",
)


@requires_full_pipeline
class TestRewardWorkerDirect:
    """Test RewardWorker.compute() dispatched via Ray actor."""

    @pytest.fixture(autouse=True)
    def _worker(self):
        self.worker = RewardWorker.remote()
        yield
        ray.kill(self.worker)

    def test_openr1mm_correct(self):
        result = ray.get(self.worker.compute.remote("openr1mm", "\\boxed{7}", "7"))
        assert result == 1.0

    def test_openr1mm_wrong(self):
        result = ray.get(self.worker.compute.remote("openr1mm", "\\boxed{8}", "7"))
        assert result == 0.0

    def test_openr1mm_string_match(self):
        result = ray.get(self.worker.compute.remote("openr1mm", "<answer>Paris</answer>", "Paris"))
        assert result == 1.0

    def test_openr1mm_fraction(self):
        result = ray.get(self.worker.compute.remote("openr1mm", "\\frac{1}{2}", "0.5"))
        assert result == 1.0

    def test_unknown_rm_type_raises(self):
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(self.worker.compute.remote("nonexistent_type", "foo", "bar"))


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


@requires_full_pipeline
class TestRewardExecutorSingleSample:
    """Test RewardExecutor.execute() with single samples (the async_rm
    path)."""

    @pytest.fixture(autouse=True)
    def _reset_executor(self):
        """Reset the singleton so each test gets a clean executor."""
        _kill_executor_workers()
        RewardExecutor._instance = None
        yield
        _kill_executor_workers()
        RewardExecutor._instance = None

    @pytest.mark.asyncio
    async def test_execute_correct_answer(self):
        args = _make_args()
        sample = _make_sample(response="\\boxed{42}", label="42")
        result = await async_rm(args, sample)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_execute_wrong_answer(self):
        args = _make_args()
        sample = _make_sample(response="\\boxed{99}", label="42")
        result = await async_rm(args, sample)
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_execute_string_match(self):
        args = _make_args()
        sample = _make_sample(response="<answer>hello</answer>", label="hello")
        result = await async_rm(args, sample)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_execute_metadata_rm_type_override(self):
        """rm_type from sample metadata should override args.rm_type."""
        args = _make_args(rm_type="deepscaler")
        sample = _make_sample(
            response="\\boxed{7}",
            label="7",
            metadata={"rm_type": "openr1mm"},
        )
        result = await async_rm(args, sample)
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_execute_unknown_rm_type_raises(self):
        args = _make_args(rm_type="totally_unknown_type")
        sample = _make_sample(response="foo", label="bar")
        with pytest.raises(NotImplementedError, match="totally_unknown_type"):
            await async_rm(args, sample)


@requires_full_pipeline
class TestBatchedAsyncRM:
    """Test batched_async_rm() — the main batched entry point."""

    @pytest.fixture(autouse=True)
    def _reset_executor(self):
        _kill_executor_workers()
        RewardExecutor._instance = None
        yield
        _kill_executor_workers()
        RewardExecutor._instance = None

    @pytest.mark.asyncio
    async def test_batch_all_correct(self):
        args = _make_args()
        samples = [_make_sample(response=f"\\boxed{{{i}}}", label=str(i)) for i in range(10)]
        rewards = await batched_async_rm(args, samples)
        assert rewards == [1.0] * 10

    @pytest.mark.asyncio
    async def test_batch_mixed_results(self):
        args = _make_args()
        samples = [
            _make_sample(response="\\boxed{1}", label="1"),  # correct
            _make_sample(response="\\boxed{2}", label="99"),  # wrong
            _make_sample(response="<answer>hello</answer>", label="hello"),  # correct
            _make_sample(response="wrong", label="right"),  # wrong
        ]
        rewards = await batched_async_rm(args, samples)
        assert rewards == [1.0, 0.0, 1.0, 0.0]

    @pytest.mark.asyncio
    async def test_batch_empty(self):
        args = _make_args()
        rewards = await batched_async_rm(args, [])
        assert rewards == []


# ===========================================================================
# High-concurrency correctness and stability tests
# ===========================================================================


@requires_full_pipeline
class TestHighConcurrency:
    """Validate reward computation correctness and stability under high
    concurrency.

    These tests stress-test the RewardExecutor + RewardWorker pipeline with
    many concurrent requests to verify:
      1. Result correctness: each sample gets the right reward regardless of
         scheduling order or worker assignment.
      2. No deadlocks / hangs: all requests complete within a reasonable timeout.
      3. No data races: worker round-robin and semaphore are safe under load.
    """

    @pytest.fixture(autouse=True)
    def _reset_executor(self):
        _kill_executor_workers()
        RewardExecutor._instance = None
        yield
        _kill_executor_workers()
        RewardExecutor._instance = None

    @pytest.mark.asyncio
    async def test_concurrent_correctness_large_batch(self):
        """200 concurrent requests — verify every result is correct."""
        args = _make_args(reward_max_concurrency=64, reward_num_workers=4)
        n = 200
        samples = []
        expected = []
        for i in range(n):
            if i % 3 == 0:
                # correct numeric
                samples.append(_make_sample(response=f"\\boxed{{{i}}}", label=str(i)))
                expected.append(1.0)
            elif i % 3 == 1:
                # wrong numeric
                samples.append(_make_sample(response=f"\\boxed{{{i}}}", label=str(i + 1)))
                expected.append(0.0)
            else:
                # string match
                samples.append(_make_sample(response=f"<answer>ans_{i}</answer>", label=f"ans_{i}"))
                expected.append(1.0)

        rewards = await asyncio.wait_for(
            batched_async_rm(args, samples),
            timeout=120,
        )

        assert len(rewards) == n
        for idx, (got, want) in enumerate(zip(rewards, expected)):
            assert got == want, (
                f"Sample {idx}: response={samples[idx].response!r}, "
                f"label={samples[idx].label!r}, got={got}, want={want}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_no_deadlock(self):
        """Saturate concurrency limit and verify no hang occurs.

        Uses max_concurrency=8 with 100 tasks to force semaphore contention.
        """
        args = _make_args(reward_max_concurrency=8, reward_num_workers=2)
        n = 100
        samples = [_make_sample(response=f"\\boxed{{{i}}}", label=str(i)) for i in range(n)]
        rewards = await asyncio.wait_for(
            batched_async_rm(args, samples),
            timeout=120,
        )
        assert len(rewards) == n
        assert all(r == 1.0 for r in rewards)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_rm_types_via_metadata(self):
        """Different rm_types in the same batch should all route correctly."""
        args = _make_args(rm_type="openr1mm")
        samples = [
            _make_sample(response="\\boxed{42}", label="42", metadata={"rm_type": "openr1mm"}),
            _make_sample(response="\\boxed{42}", label="42", metadata={"rm_type": "openr1mm"}),
            _make_sample(response="\\boxed{42}", label="42", metadata={"rm_type": "openr1mm"}),
            _make_sample(response="\\boxed{42}", label="42", metadata={"rm_type": "openr1mm"}),
            _make_sample(response="\\boxed{42}", label="42", metadata={"rm_type": "openr1mm"}),
            _make_sample(response="\\boxed{42}", label="42", metadata={"rm_type": "openr1mm"}),
            _make_sample(response="A", label="A", metadata={"rm_type": "multiple_choice"}),
        ]
        rewards = await asyncio.wait_for(
            batched_async_rm(args, samples),
            timeout=60,
        )
        rewards = list(rewards)
        assert len(rewards) == len(samples)
        # All three should be correct (1.0)
        assert all(r == 1.0 for r in rewards), f"Got {rewards}"

    @pytest.mark.asyncio
    async def test_concurrent_repeated_identical_requests(self):
        """Same input repeated many times — deterministic results expected."""
        args = _make_args()
        n = 50
        samples = [_make_sample(response="\\boxed{7}", label="7")] * n
        rewards = await asyncio.wait_for(
            batched_async_rm(args, samples),
            timeout=60,
        )
        assert rewards == [1.0] * n

    @pytest.mark.asyncio
    async def test_concurrent_interleaved_correct_incorrect(self):
        """Interleave correct and incorrect answers to detect ordering bugs."""
        args = _make_args(reward_max_concurrency=32, reward_num_workers=4)
        n = 100
        samples = []
        expected = []
        for i in range(n):
            if i % 2 == 0:
                samples.append(_make_sample(response=f"\\boxed{{{i}}}", label=str(i)))
                expected.append(1.0)
            else:
                samples.append(_make_sample(response=f"\\boxed{{{i}}}", label=str(i + 1000)))
                expected.append(0.0)

        rewards = await asyncio.wait_for(
            batched_async_rm(args, samples),
            timeout=120,
        )
        assert rewards == expected

    @pytest.mark.asyncio
    async def test_executor_singleton_under_concurrent_access(self):
        """Multiple concurrent get_or_create calls should return the same
        instance."""
        RewardExecutor._instance = None

        async def get_executor():
            return RewardExecutor.get_or_create(max_concurrency=64, num_workers=4)

        executors = await asyncio.gather(*[get_executor() for _ in range(50)])
        # All should be the exact same object
        assert all(e is executors[0] for e in executors)

    @pytest.mark.asyncio
    async def test_worker_round_robin_distribution(self):
        """Verify requests are distributed across workers (round-robin)."""
        args = _make_args(reward_max_concurrency=64, reward_num_workers=4)
        executor = RewardExecutor.get_or_create(
            max_concurrency=args.reward_max_concurrency,
            num_workers=args.reward_num_workers,
        )
        # Force worker initialization
        await executor._ensure_workers()

        initial_index = executor._worker_index
        n = 20
        samples = [_make_sample(response=f"\\boxed{{{i}}}", label=str(i)) for i in range(n)]
        rewards = await asyncio.wait_for(
            batched_async_rm(args, samples),
            timeout=60,
        )
        assert all(r == 1.0 for r in rewards)
        # worker_index should have advanced by n (each request picks one worker)
        assert executor._worker_index == initial_index + n
