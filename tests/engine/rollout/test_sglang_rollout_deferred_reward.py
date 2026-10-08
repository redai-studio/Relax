# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""With framework-deferred scoring the rollout never calls the reward function
while it generates -- the judge is asleep then -- and evaluation is scored in
one score phase after every dataset has generated."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


try:
    from relax.engine.rollout import sglang_rollout

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from relax.utils.types import Sample


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing sglang dependencies")


def _args(*, deferred: bool, group_rm: bool = False, **overrides) -> SimpleNamespace:
    values = dict(
        defer_reward_to_post_process=deferred,
        custom_reward_post_process_path=None,
        group_rm=group_rm,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        sglang_enable_deterministic_inference=False,
        reward_key=None,
        eval_reward_key=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def rollout(monkeypatch):
    """The rollout with generation and the reward function replaced by
    recorders."""
    scored: list = []

    async def fake_dispatch(state, args, sample, sampling_params, evaluation=False):
        sample.response = "answer"
        sample.status = Sample.Status.COMPLETED
        return sample

    async def fake_async_rm(args, sample, **kwargs):
        scored.append(sample.index)
        return 1.0

    async def fake_batched_async_rm(args, samples, **kwargs):
        scored.extend(sample.index for sample in samples)
        return [1.0 for _ in samples]

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: SimpleNamespace(opd_manager=None, aborted=False))
    monkeypatch.setattr(sglang_rollout, "_dispatch_generate", fake_dispatch)
    monkeypatch.setattr(sglang_rollout, "async_rm", fake_async_rm)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", fake_batched_async_rm)
    return scored


async def test_sglang_rollout_scores_inline_by_default(rollout):
    sample = await sglang_rollout.generate_and_rm(_args(deferred=False), Sample(index=3), {})

    assert rollout == [3]
    assert sample.reward == 1.0


async def test_sglang_rollout_deferred_reward_makes_no_reward_call_while_generating(rollout):
    args = _args(deferred=True)

    sample = await sglang_rollout.generate_and_rm(args, Sample(index=3), {})
    group = await sglang_rollout.generate_and_rm_group(
        _args(deferred=True, group_rm=True), [Sample(index=4), Sample(index=5)], {}
    )

    assert rollout == []
    assert sample.reward is None
    assert [member.reward for member in group] == [None, None]


async def test_sglang_rollout_deferred_reward_accepts_a_finished_sample_without_reward(rollout):
    """A completed sample carried over from an earlier step has no reward yet;
    it is scored with the batch it is finally published in."""
    carried = Sample(index=7, response="answer", status=Sample.Status.COMPLETED)

    assert await sglang_rollout.generate_and_rm(_args(deferred=True), carried, {}) is carried
    with pytest.raises(AssertionError):
        await sglang_rollout.generate_and_rm(_args(deferred=False), carried, {})


async def test_sglang_rollout_custom_post_process_keeps_inline_rewards(rollout):
    """A userland hook owns the deferral; the inline reward call is whatever
    the run configured (typically a dummy) and still happens."""
    args = _args(deferred=True, custom_reward_post_process_path="my_module.post_process")

    await sglang_rollout.generate_and_rm(args, Sample(index=3), {})

    assert rollout == [3]


async def test_sglang_rollout_deferred_eval_scores_all_datasets_in_one_phase(monkeypatch):
    calls: list = []

    async def fake_run_deferred_scoring(args, groups, *, restore_rollout=False):
        calls.append(([[sample.index for sample in group] for group in groups], restore_rollout))
        for group in groups:
            for sample in group:
                sample.reward = {"acc": float(sample.index)}

    monkeypatch.setattr(sglang_rollout, "run_deferred_scoring", fake_run_deferred_scoring)
    args = _args(
        deferred=True,
        eval_reward_key="acc",
        eval_datasets=[
            SimpleNamespace(name="math", n_samples_per_eval_prompt=2),
            SimpleNamespace(name="code", n_samples_per_eval_prompt=1),
        ],
    )
    results = {
        "math": {"rewards": [], "samples": [Sample(index=i) for i in range(4)]},
        "code": {"rewards": [], "samples": [Sample(index=i) for i in (10, 11)]},
    }

    await sglang_rollout._score_deferred_eval_results(args, results)

    # One score phase for both datasets, in prompt groups, with rollout brought back after.
    assert calls == [([[0, 1], [2, 3], [10], [11]], True)]
    assert results["math"]["rewards"] == [0.0, 1.0, 2.0, 3.0]
    assert results["code"]["rewards"] == [10.0, 11.0]
