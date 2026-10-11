# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
from types import ModuleType, SimpleNamespace

import pytest

from relax.inference.lifecycle import LifecycleCoordinator, run_deferred_batch


@pytest.mark.parametrize("student_prefill", [False, True])
async def test_deferred_writeback_precedes_publish_and_never_overlaps(monkeypatch, student_prefill):
    events = []
    active = {"rollout"}

    class Pool:
        def __init__(self, name):
            self.name = name
            self.onload = SimpleNamespace(remote=self.activate)
            self.offload = SimpleNamespace(remote=self.deactivate)

        async def activate(self):
            assert not active, f"overlap: {active} -> {self.name}"
            active.add(self.name)
            events.append(self.name + ":on")

        async def deactivate(self):
            active.discard(self.name)
            events.append(self.name + ":off")

    rollout_pool, teacher, judge = Pool("rollout"), Pool("teacher"), Pool("genrm")
    rollout = SimpleNamespace(
        onload=rollout_pool.activate,
        offload=rollout_pool.deactivate,
        lifecycle_coordinator=LifecycleCoordinator(),
        _inference_genrm_managers=[judge],
    )
    module = ModuleType("relax.distributed.ray.rollout")
    module.get_local_rollout_manager = lambda: rollout
    monkeypatch.setitem(sys.modules, module.__name__, module)
    rewards = ModuleType("relax.engine.rewards")

    async def score(args, samples):
        assert active == {"genrm"}
        return [0.25, 0.75]

    rewards.batched_async_rm = score
    monkeypatch.setitem(sys.modules, rewards.__name__, rewards)
    utils = ModuleType("relax.utils.utils")
    utils.post_process_rewards = lambda args, samples: ([s.reward for s in samples], [s.reward for s in samples])
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    samples = [SimpleNamespace(index=7, response_length=1), SimpleNamespace(index=7, response_length=2)]

    class OPD:
        topk_worker = SimpleNamespace(spec=SimpleNamespace(student_at_teacher=student_prefill))

        async def prefill_teacher(self, rows, strict):
            assert strict and active == {"teacher"}
            for ordinal, row in enumerate(rows):
                row.teacher_log_probs = [-0.1 - ordinal] * row.response_length
            events.append("teacher:writeback")

        async def prefill_student(self, rows, encode):
            assert active == {"rollout"}
            events.append("student:writeback")

        def finish_prefill(self, rows, strict):
            assert strict and not active
            assert [s.teacher_log_probs for s in rows] == [[-0.1], [-1.1, -1.1]]

    async def publish():
        assert not active
        assert all(hasattr(s, "teacher_log_probs") and hasattr(s, "_inference_reward_result") for s in samples)
        events.append("publish")

    args = SimpleNamespace(
        _inference_teacher_managers={"teacher": teacher},
        opd_teacher_defer=True,
        defer_reward_to_post_process=True,
        custom_reward_post_process_path=None,
    )
    await run_deferred_batch(args, samples, OPD(), None, publish)
    assert events[-2:] == ["genrm:off", "publish"]
    assert events.index("teacher:writeback") < events.index("genrm:on")
    if student_prefill:
        assert events.index("teacher:off", 3) < events.index("student:writeback")


async def test_deferred_teacher_failure_never_publishes(monkeypatch):
    events = []

    async def on():
        events.append("on")

    async def off():
        events.append("off")

    async def fail(*args, **kwargs):
        raise RuntimeError("one sample missing probabilities")

    teacher = SimpleNamespace(onload=SimpleNamespace(remote=on), offload=SimpleNamespace(remote=off))
    rollout = SimpleNamespace(offload=off, lifecycle_coordinator=LifecycleCoordinator(), _inference_genrm_managers=[])
    module = ModuleType("relax.distributed.ray.rollout")
    module.get_local_rollout_manager = lambda: rollout
    monkeypatch.setitem(sys.modules, module.__name__, module)
    rewards = ModuleType("relax.engine.rewards")
    rewards.batched_async_rm = fail
    monkeypatch.setitem(sys.modules, rewards.__name__, rewards)
    args = SimpleNamespace(_inference_teacher_managers={"teacher": teacher}, opd_teacher_defer=True)

    async def publish():
        pytest.fail("Partial Teacher results reached training")

    with pytest.raises(RuntimeError, match="missing probabilities"):
        await run_deferred_batch(args, [], SimpleNamespace(prefill_teacher=fail), None, publish)
    assert events == ["off", "off", "on", "off"]


@pytest.mark.parametrize("custom_hook", [False, True])
@pytest.mark.parametrize("iterator_groups", [False, True])
async def test_deferred_eval_preserves_prompt_groups_and_returned_rewards(monkeypatch, custom_hook, iterator_groups):
    async def nothing():
        return None

    judge = SimpleNamespace(onload=SimpleNamespace(remote=nothing), offload=SimpleNamespace(remote=nothing))
    rollout = SimpleNamespace(
        offload=nothing, lifecycle_coordinator=LifecycleCoordinator(), _inference_genrm_managers=[judge]
    )
    module = ModuleType("relax.distributed.ray.rollout")
    module.get_local_rollout_manager = lambda: rollout
    monkeypatch.setitem(sys.modules, module.__name__, module)
    samples = [SimpleNamespace(prompt=prompt, reward=0.0) for prompt in ["math", "math", "code", "code"]]
    calls = []

    async def score(args, group):
        assert len(group) == 2 and group[0].prompt == group[1].prompt
        calls.append(group[0].prompt)
        return [{"score": 0.2}, {"score": 0.8}]

    rewards = ModuleType("relax.engine.rewards")
    rewards.batched_async_rm = score
    monkeypatch.setitem(sys.modules, rewards.__name__, rewards)
    utils = ModuleType("relax.utils.utils")

    def process(args, rows):
        assert not args.rewards_normalization
        return [0.2, 0.8, 0.2, 0.8], [99] * 4

    utils.post_process_rewards = process
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    args = SimpleNamespace(
        defer_reward_to_post_process=True,
        group_rm=True,
        eval_reward_key=None,
        reward_key="score",
        custom_reward_post_process_path="custom" if custom_hook else None,
    )
    groups = [samples[:2], samples[2:]]
    if iterator_groups:
        groups = [iter(group) for group in groups]
    await run_deferred_batch(args, samples, None, None, nothing, evaluation=True, reward_groups=groups)
    assert [sample.reward["score"] for sample in samples] == [0.2, 0.8, 0.2, 0.8]
    assert calls == ([] if custom_hook else ["math", "code"])
    assert all(not hasattr(sample, "_inference_reward_result") for sample in samples)
