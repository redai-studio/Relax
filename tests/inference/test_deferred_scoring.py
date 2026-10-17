# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from relax.engine.rollout.deferred_scoring import DeferredBatch, DeferredBatchState


def _sample(index=1, *, versions=("v1",)):
    return SimpleNamespace(
        index=index,
        group_index=0,
        response_length=2,
        tokens=[1, 2, 3],
        loss_mask=[1, 1],
        weight_versions=list(versions),
        reward=None,
        teacher_log_probs=None,
        remove_sample=False,
    )


def _lifecycle(events):
    async def offload():
        events.append("offload:rollout")

    async def activate(role):
        events.append(f"activate:{role}")

    async def deactivate(role):
        events.append(f"deactivate:{role}")

    return dict(offload_rollout=offload, activate_role=activate, deactivate_role=deactivate)


async def test_deferred_batch_scores_copies_and_commits_original_rows_after_all_releases():
    originals = [_sample(), _sample()]
    events = []
    batch = DeferredBatch("run")

    async def prepare(samples):
        events.append("prepare")
        assert samples[0] is not originals[0]

    async def reward(groups):
        events.append("reward")
        for ordinal, sample in enumerate(groups[0]):
            sample.reward = ordinal + 1

    async def teacher(samples):
        events.append("teacher")
        for ordinal, sample in enumerate(samples):
            sample.teacher_log_probs = [-0.1 - ordinal, -0.2 - ordinal]
        assert originals[0].reward is None
        assert originals[0].teacher_log_probs is None

    def validate(samples):
        events.append("validate")
        assert all(len(sample.teacher_log_probs) == 2 for sample in samples)

    async def commit(scored):
        events.append("commit")
        assert scored.state == DeferredBatchState.PUBLISHING
        assert scored.samples == originals
        assert [sample.reward for sample in originals] == [1, 2]
        assert originals[1].teacher_log_probs == [-1.1, -1.2]

    await batch.complete(
        4,
        [originals],
        **_lifecycle(events),
        reward_stage=reward,
        prepare_teacher=prepare,
        teacher_stage=teacher,
        assemble_validate=validate,
        commit=commit,
    )

    assert batch.state == DeferredBatchState.COMMITTED
    assert batch.keys[0] != batch.keys[1]
    assert batch.keys[0].sample_index == batch.keys[1].sample_index == 1
    assert events == [
        "prepare",
        "offload:rollout",
        "activate:genrm",
        "reward",
        "deactivate:genrm",
        "activate:teacher",
        "teacher",
        "deactivate:teacher",
        "validate",
        "commit",
    ]
    previous = list(events)
    await batch.complete(4, [originals], **_lifecycle(events), commit=commit)
    assert events == previous


@pytest.mark.parametrize("failure", ["teacher", "deactivate", "validation"])
async def test_deferred_failure_never_writes_originals_or_publishes(failure):
    original = _sample()
    events = []
    batch = DeferredBatch("run")
    lifecycle = _lifecycle(events)

    async def teacher(samples):
        samples[0].teacher_log_probs = [-1, -2]
        if failure == "teacher":
            raise RuntimeError("teacher failed")

    async def deactivate(role):
        events.append(f"deactivate:{role}")
        if failure == "deactivate":
            raise RuntimeError("release unconfirmed")

    def validate(samples):
        if failure == "validation":
            raise ValueError("missing row")

    async def commit(batch):
        events.append("commit")

    lifecycle["deactivate_role"] = deactivate
    with pytest.raises((RuntimeError, ValueError)):
        await batch.complete(
            0, [[original]], **lifecycle, teacher_stage=teacher, assemble_validate=validate, commit=commit
        )
    assert original.teacher_log_probs is None
    assert batch.state == DeferredBatchState.FAILED
    assert "commit" not in events


async def test_deferred_student_stage_restores_one_pinned_version_then_releases_rollout():
    events = []
    batch = DeferredBatch("run")

    async def restore(version):
        events.append(f"restore:{version}")

    async def student(samples):
        events.append("student")

    await batch.complete(0, [[_sample()]], **_lifecycle(events), student_stage=student, restore_student=restore)

    assert events == ["offload:rollout", "restore:v1", "student", "deactivate:rollout"]


async def test_deferred_student_restore_failure_still_releases_rollout_lease():
    events = []
    batch = DeferredBatch("run")

    async def restore(version):
        events.append(f"restore:{version}")
        raise RuntimeError("restored a different policy version")

    async def student(samples):
        raise AssertionError("student scoring must not run after a failed restore")

    async def commit(batch):
        events.append("commit")

    with pytest.raises(RuntimeError, match="different policy version"):
        await batch.complete(
            0, [[_sample()]], **_lifecycle(events), student_stage=student, restore_student=restore, commit=commit
        )

    assert events == ["offload:rollout", "restore:v1", "deactivate:rollout"]
    assert batch.state == DeferredBatchState.FAILED


@pytest.mark.parametrize("versions", [(), ("v1", "v2")])
async def test_deferred_student_unknown_or_mixed_versions_fail_before_gpu_transition(versions):
    events = []
    batch = DeferredBatch("run")

    async def student(samples):
        raise AssertionError("invalid version must not be scored")

    with pytest.raises(ValueError, match="version"):
        await batch.complete(0, [[_sample(versions=versions)]], **_lifecycle(events), student_stage=student)

    assert events == []
    assert batch.state == DeferredBatchState.FAILED


async def test_deferred_batch_capacity_and_duplicate_object_reject_before_scoring():
    events = []
    sample = _sample()
    with pytest.raises(ValueError, match="capacity"):
        await DeferredBatch("run", max_samples=1).complete(0, [[sample, _sample()]], **_lifecycle(events))
    with pytest.raises(ValueError, match="multiple export"):
        await DeferredBatch("run").complete(0, [[sample, sample]], **_lifecycle(events))
    assert events == []


async def test_deferred_deadline_drains_activated_role_and_does_not_commit():
    events = []
    batch = DeferredBatch("run", timeout=0.02)

    async def teacher(samples):
        await asyncio.Event().wait()

    with pytest.raises(asyncio.TimeoutError):
        await batch.complete(
            0, [[_sample()]], **_lifecycle(events), teacher_stage=teacher, assemble_validate=lambda samples: None
        )
    assert events[-1] == "deactivate:teacher"
    assert batch.state == DeferredBatchState.FAILED


async def test_deferred_partial_publication_failure_is_not_a_commit():
    events = []
    batch = DeferredBatch("run")

    async def commit(batch):
        assert batch.state == DeferredBatchState.PUBLISHING
        raise RuntimeError("TQ write receipt failed")

    with pytest.raises(RuntimeError, match="receipt"):
        await batch.complete(0, [[_sample()]], **_lifecycle(events), commit=commit)
    assert batch.state == DeferredBatchState.FAILED
    with pytest.raises(RuntimeError, match="already FAILED"):
        await batch.complete(0, batch.groups, **_lifecycle(events), commit=commit)


@pytest.fixture
def opd():
    from relax.engine.rollout.on_policy_distillation import OpdManager
    from relax.utils.types import Sample

    return OpdManager, Sample


def _opd_args(mode, advantage=False):
    return SimpleNamespace(
        opd_token_selection=mode,
        opd_log_prob_top_k=2,
        opd_kl_coef=0.1 if advantage else 0.0,
        opd_loss_coef=0.0 if advantage else 0.1,
        opd_teacher_prompt_key=None,
        opd_teacher_image_key=None,
    )


@pytest.mark.parametrize("mode", ["student_sampled", "student_topk", "teacher_topk", "union"])
@pytest.mark.parametrize("advantage", [False, True])
def test_deferred_opd_assembly_reuses_inline_math_with_complete_rows(opd, mode, advantage):
    Manager, Sample = opd
    manager = Manager(_opd_args(mode, advantage))
    sample = Sample(
        tokens=[1, 2, 3], response_length=2, loss_mask=[1, 1], teacher_log_probs=[-1, -2], rollout_log_probs=[-2, -3]
    )
    sample.student_topk_token_ids = np.array([[3, 4], [5, 6]], dtype=np.int32)
    sample.teacher_topk_token_ids = np.array([[3, 7], [5, 8]], dtype=np.int32)
    for field in (
        "student_topk_log_probs",
        "teacher_topk_log_probs",
        "teacher_at_student_topk_log_probs",
        "student_at_teacher_topk_log_probs",
    ):
        setattr(sample, field, np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32))
    expected = Sample(**vars(sample))
    manager._assemble_transfer([expected])

    manager.assemble_validate([sample])

    for field in manager.schema_opd_transfer_data():
        assert np.array_equal(getattr(sample, field), getattr(expected, field))


@pytest.mark.parametrize(
    "field", ["teacher_topk_token_ids", "teacher_topk_log_probs", "student_at_teacher_topk_log_probs"]
)
def test_deferred_opd_required_cross_scoring_rows_never_silently_disappear(opd, field):
    Manager, Sample = opd
    manager = Manager(_opd_args("teacher_topk", True))
    sample = Sample(tokens=[1, 2, 3], response_length=2)
    sample.teacher_topk_token_ids = np.array([[2, 3], [4, 5]], dtype=np.int32)
    sample.teacher_topk_log_probs = np.ones((2, 2), dtype=np.float32)
    sample.student_at_teacher_topk_log_probs = np.ones((2, 2), dtype=np.float32)
    setattr(sample, field, None)

    with pytest.raises(ValueError, match=field):
        manager.assemble_validate([sample])


def test_deferred_opd_zero_response_has_explicit_empty_channels(opd):
    Manager, Sample = opd
    sample = Sample(tokens=[1], response_length=0)
    manager = Manager(_opd_args("union", True))

    manager.assemble_validate([sample])

    assert sample.opd_topk_token_ids.shape == (0, 2)
    assert sample.opd_topk_teacher_log_probs.shape == (0, 2)
    assert sample.opd_topk_ksz.shape == (0,)


async def test_deferred_teacher_stage_drains_sibling_tasks_and_rejects_partial_failure(opd, monkeypatch):
    Manager, Sample = opd
    import relax.engine.rollout.on_policy_distillation as module

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    manager = Manager(_opd_args("student_sampled"))
    samples = [Sample(tokens=[1, 2], response_length=1), Sample(tokens=[1, 2], response_length=1)]
    completed = []

    async def fetch(sample, session):
        if sample is samples[0]:
            raise ValueError("malformed response")
        await asyncio.sleep(0)
        completed.append(True)
        return True

    monkeypatch.setattr(module, "_create_teacher_client_session", lambda args: Session())
    monkeypatch.setattr(manager, "_teacher_prefill", fetch)
    with pytest.raises(ValueError, match="malformed"):
        await manager.teacher_prefill(samples)
    assert completed == [True]

    async def partial(sample, session):
        return sample is samples[0]

    monkeypatch.setattr(manager, "_teacher_prefill", partial)
    with pytest.raises(RuntimeError, match="ordinals"):
        await manager.teacher_prefill(samples)


async def test_deferred_student_failure_cannot_reuse_old_cross_values(opd, monkeypatch):
    Manager, Sample = opd
    import relax.engine.rollout.on_policy_distillation as module

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    manager = Manager(_opd_args("teacher_topk", True))
    sample = Sample(tokens=[1, 2], response_length=1)
    sample.student_at_teacher_topk_log_probs = np.array([[-1.0, -2.0]])

    async def fail_without_result(*args):
        return None

    monkeypatch.setattr(module, "_create_teacher_client_session", lambda args: Session())
    monkeypatch.setattr(manager, "_student_prefill", fail_without_result)
    with pytest.raises(RuntimeError, match="student fetch failed"):
        await manager.student_prefill([sample])
    assert sample.student_at_teacher_topk_log_probs is None


async def test_deferred_prepare_reuses_existing_opsd_input_builder(opd, monkeypatch):
    Manager, Sample = opd
    args = _opd_args("student_sampled")
    args.opd_teacher_prompt_key = "teacher_prompt"
    manager = Manager(args)
    sample = Sample(tokens=[1, 2], response_length=1)
    calls = []

    async def prepare(args, sample):
        calls.append(sample)
        sample.teacher_tokens = [9, 8, 2]
        sample.teacher_prompt_length = 2

    monkeypatch.setattr(manager.opsd_worker, "build_teacher_inputs", prepare)
    await manager.prepare([sample])
    assert calls == [sample]
    assert sample.teacher_tokens == [9, 8, 2]
