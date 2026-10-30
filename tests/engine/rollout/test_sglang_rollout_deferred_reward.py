# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""With framework-deferred scoring the rollout never calls the reward function
while it generates -- the judge is asleep then -- and evaluation is scored in
one score phase after every dataset has generated."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


try:
    from relax.engine.rollout import sglang_rollout

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from relax.utils.types import Sample


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing sglang dependencies")


class _OpdManager:
    """Stands in for the rollout's OPD manager; records inline teacher
    requests."""

    def __init__(self):
        self.prefilled: list = []

    async def prefill(self, samples, encode_multimodal_inputs=None):
        self.prefilled.append(samples)


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

    async def fake_run_deferred_scoring(args, groups, *, evaluation=False):
        calls.append(([[sample.index for sample in group] for group in groups], evaluation))
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

    # One score phase for both datasets, in prompt groups, flagged as an evaluation
    # (rollout is brought back after, and no distillation is run).
    assert calls == [([[0, 1], [2, 3], [10], [11]], True)]
    assert results["math"]["rewards"] == [0.0, 1.0, 2.0, 3.0]
    assert results["code"]["rewards"] == [10.0, 11.0]


def _opd_args(*, teacher_deferred: bool, **overrides) -> SimpleNamespace:
    return _args(deferred=False, use_opd=True, opd_type="sglang", opd_teacher_defer=teacher_deferred, **overrides)


@pytest.fixture
def opd_rollout(rollout, monkeypatch):
    manager = _OpdManager()
    monkeypatch.setattr(
        sglang_rollout, "GenerateState", lambda args: SimpleNamespace(opd_manager=manager, aborted=False)
    )
    return manager


async def test_sglang_rollout_asks_the_teacher_inline_by_default(opd_rollout):
    sample = await sglang_rollout.generate_and_rm(_opd_args(teacher_deferred=False), Sample(index=3), {})

    assert opd_rollout.prefilled == [sample]


async def test_deferred_teacher_no_request_during_generation(opd_rollout):
    """A deferred teacher is asleep while rollout generates: none of the three
    inline sites may ask it."""
    await sglang_rollout.generate_and_rm(_opd_args(teacher_deferred=True), Sample(index=3), {})
    await sglang_rollout.generate_and_rm_group(
        _opd_args(teacher_deferred=True, group_rm=True), [Sample(index=4), Sample(index=5)], {}
    )

    assert opd_rollout.prefilled == []


async def test_sglang_rollout_deferred_teacher_still_scores_rewards_inline(opd_rollout, rollout):
    """Deferring the teacher does not defer the reward."""
    sample = await sglang_rollout.generate_and_rm(_opd_args(teacher_deferred=True), Sample(index=3), {})

    assert rollout == [3]
    assert sample.reward == 1.0


# ----------------------------------------------------------------------
# Publishing waits for generation to stop.
# ----------------------------------------------------------------------


class _GenerationState:
    """Stands in for ``GenerateState``: every submitted group finishes at once
    except the last of each round, which stays in flight until released -- the
    over-sampling surplus that is still generating when the target is met."""

    def __init__(self):
        self.remaining_batch_size = 0
        self.pendings: set = set()
        self.protected_pendings: set = set()
        self.prefetched_samples_ref = None
        self.last_step_current_deficit = 0
        self.release = asyncio.Event()

    def submit_generate_tasks(self, groups):
        for position, group in enumerate(groups):
            in_flight = position == len(groups) - 1
            self.pendings.add(asyncio.create_task(self._generate(group, in_flight)))
        self.remaining_batch_size += len(groups)

    async def _generate(self, group, in_flight):
        if in_flight:
            await self.release.wait()
        return group

    def reset(self):
        self.pendings = set()


@pytest.fixture
def generation(monkeypatch):
    """The real ``generate_rollout_async`` over a fake engine, with ``abort``
    and the publishing exit replaced by recorders."""
    events: list[str] = []
    state = _GenerationState()
    failure: dict = {}

    async def fake_abort(args, rollout_id):
        events.append("abort")
        if "abort" in failure:
            raise failure["abort"]
        state.release.set()
        while state.pendings:
            _done, state.pendings = await asyncio.wait(state.pendings)
        events.append("generation stopped")
        return [], []

    async def fake_transfer(args, batch, count, rollout_id, data_system_client, is_last=False):
        events.append(f"publish {count} group(s), {len(state.pendings)} still generating")

    async def no_profile(args, rollout_id):
        return None

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "abort", fake_abort)
    monkeypatch.setattr(sglang_rollout, "transfer_batch_to_data_system", fake_transfer)
    monkeypatch.setattr(sglang_rollout, "start_sglang_profile", no_profile)
    monkeypatch.setattr(sglang_rollout, "stop_sglang_profile", no_profile)
    # The data source hands its groups back directly.
    monkeypatch.setattr(sglang_rollout.ray, "get", lambda ref: ref)

    def groups(count):
        return [
            [Sample(index=i, prompt="question", response="answer", label="label", status=Sample.Status.COMPLETED)]
            for i in range(count)
        ]

    data_source = SimpleNamespace(get_samples=SimpleNamespace(remote=groups))

    async def run(**flags):
        args = _args(
            deferred=False,
            rollout_global_dataset=True,
            dynamic_sampling_filter_path=None,
            fully_async=False,
            num_rollout=10,
            rollout_batch_size=2,
            # One group more than the step commits: it is still in flight when the target is met.
            over_sampling_batch_size=3,
            n_samples_per_prompt=1,
            use_dynamic_global_batch_size=False,
            debug_rollout_only=False,
        )
        for name, value in flags.items():
            setattr(args, name, value)
        return await sglang_rollout.generate_rollout_async(args, 0, data_source, None)

    return SimpleNamespace(run=run, events=events, failure=failure, state=state)


async def test_sglang_rollout_publishes_before_abort_by_default(generation):
    """Without deferred scoring the order is what it always was: publish while
    the surplus is still generating, abort it afterwards."""
    await generation.run()

    assert generation.events == ["publish 2 group(s), 1 still generating", "abort", "generation stopped"]


@pytest.mark.parametrize(
    "flags",
    [
        {"defer_reward_to_post_process": True},
        {"use_opd": True, "opd_type": "sglang", "opd_teacher_defer": True},
    ],
)
async def test_sglang_rollout_deferred_scoring_waits_for_generation_to_stop(generation, flags):
    """Deferred scoring runs inside the publishing exit and takes rollout's
    GPUs: nothing may still be generating when it starts."""
    output, _aborted = await generation.run(**flags)

    assert generation.events == ["abort", "generation stopped", "publish 2 group(s), 0 still generating"]
    assert len(output.samples) == 2


async def test_sglang_rollout_custom_post_process_keeps_the_publishing_order(generation):
    """A userland hook owns its swap; the framework does not reorder its
    step."""
    await generation.run(defer_reward_to_post_process=True, custom_reward_post_process_path="my_module.post_process")

    assert generation.events[0] == "publish 2 group(s), 1 still generating"


async def test_sglang_rollout_deferred_scoring_publishes_nothing_if_generation_cannot_be_stopped(generation):
    generation.failure["abort"] = RuntimeError("router unreachable")

    with pytest.raises(RuntimeError, match="router unreachable"):
        await generation.run(defer_reward_to_post_process=True)
    # Let the cancelled publishing task unwind, then free the stand-in engine.
    await asyncio.sleep(0)
    generation.state.release.set()

    assert generation.events == ["abort"]
