# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relax.engine.rollout.deferred_opd import DeferredOpdSession
from relax.utils.types import Sample


@pytest.mark.asyncio
async def test_deferred_opd_staging_preserves_final_marker_without_publication():
    publish = AsyncMock()
    session = DeferredOpdSession(
        SimpleNamespace(), 3, None, SimpleNamespace(schema_opd_transfer_data=lambda: []), publish=publish
    )
    sample = Sample(index=1, tokens=[1, 2], response_length=1)
    await session.transfer(None, [[sample]], 0, 3, None)
    await session.transfer(None, [], 1, 3, None, is_last=True)
    assert session.staged_batches == 1
    publish.assert_not_called()
    await session.cancel()
    assert session.staged_batches == 0
    publish.assert_not_called()


@pytest.mark.asyncio
async def test_deferred_opd_empty_final_batch_without_data_fails_instead_of_hanging_consumer():
    session = DeferredOpdSession(
        SimpleNamespace(), 3, None, SimpleNamespace(schema_opd_transfer_data=lambda: []), publish=AsyncMock()
    )
    with pytest.raises(RuntimeError, match="no staged batch"):
        await session.transfer(None, [], 0, 3, None, is_last=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_genrm_session_scores_before_publish_and_reuses_processed_reward(monkeypatch, fail):
    from contextlib import asynccontextmanager

    from relax.engine.rollout import scoring_phase
    from relax.engine.rollout.deferred import PublicationReceipt, PublicationState
    from relax.utils import misc
    from relax.utils.utils import convert_samples_to_train_data

    args = SimpleNamespace(
        _genrm_instances_resolved={"judge": {}},
        defer_reward_to_post_process=True,
        _genrm_colocate_with_rollout=True,
        custom_reward_post_process_path="test.hook",
        reward_key=None,
        _inference_phase_coordinator=object(),
        multimodal_keys=None,
        use_opd=False,
        debug_train_only=False,
    )
    events = []

    @asynccontextmanager
    async def phase(args, phase):
        events.append("enter")
        try:
            yield True
        finally:
            events.append("leave")

    def hook(scoring_args, samples):
        assert scoring_args._managed_genrm_scoring_phase
        events.append("score")
        if fail:
            raise ConnectionError("judge unavailable")
        for sample in samples:
            sample.reward = 0.75
        return [0.25] * len(samples)

    async def publish(args, payload, count, rollout_id, client, *, is_last, operation_id):
        events.append(("publish", count, is_last))
        batch = convert_samples_to_train_data(args, payload[0])
        assert batch["rewards"].tolist() == [0.25]
        assert "metadata" not in batch.keys()
        return PublicationReceipt(operation_id, PublicationState.CONFIRMED)

    monkeypatch.setattr(scoring_phase, "async_scoring_phase", phase)
    monkeypatch.setattr(misc, "load_function", lambda path: hook)
    session = DeferredOpdSession.maybe_create(args, 0, None, None, publish=publish)
    assert session is not None
    await session.transfer(args, [[Sample(index=1, tokens=[1, 2], response_length=1)]], 0, 0, None)
    await session.transfer(args, [[Sample(index=2, tokens=[1, 3], response_length=1)]], 1, 0, None, is_last=True)
    if fail:
        with pytest.raises(RuntimeError, match="did not publish"):
            await session.flush()
        assert events == ["enter", "score", "leave"]
    else:
        await session.flush()
        assert events == ["enter", "score", "score", "leave", ("publish", 0, False), ("publish", 1, True)]


def test_managed_genrm_hook_uses_coordinator_lifecycle_without_legacy_swap(monkeypatch):
    from examples.generate_reward_model import post_process_genrm_swap as hook

    async def score(samples):
        return [0.75] * len(samples)

    monkeypatch.setattr(hook, "_score_all", score)
    args = SimpleNamespace(_managed_genrm_scoring_phase=True, advantage_estimator="grpo", rewards_normalization=False)
    sample = Sample(response_length=1)
    assert hook.custom_reward_post_process(args, [sample]) == [0.75]
    assert sample.reward == 0.75


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_sync_scoring_keeps_phase_until_callback_finishes(fail):
    from relax.engine.inference.phases import PhaseCoordinator
    from relax.engine.rollout.scoring_phase import async_scoring_phase
    from relax.utils.async_utils import run_sync_to_completion

    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    events = []

    async def call(owner, method):
        events.append((owner, method))

    coordinator = PhaseCoordinator({"inference": ("student",), "genrm": ("judge",)}, "inference", call)
    args = SimpleNamespace(
        _genrm_instances_resolved={"judge": {}},
        _genrm_colocate_with_rollout=True,
        _inference_phase_coordinator=SimpleNamespace(switch=SimpleNamespace(remote=coordinator.switch)),
    )

    def hook():
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        events.append("scored")
        if fail:
            raise ConnectionError("judge unavailable")

    async def score():
        async with async_scoring_phase(args, "genrm"):
            await run_sync_to_completion(hook)

    task = asyncio.create_task(score())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert events == [("student", "offload"), ("judge", "onload")]
    finally:
        release.set()
    error = ConnectionError if fail else asyncio.CancelledError
    with pytest.raises(error):
        await asyncio.wait_for(task, 2)
    assert events[-3:] == ["scored", ("judge", "offload"), ("student", "onload")]


@pytest.mark.asyncio
async def test_genrm_judge_transport_failure_propagates_instead_of_zero_reward():
    import httpx

    from examples.generate_reward_model import post_process_genrm_swap as hook

    def fail(request):
        raise httpx.ConnectError("judge unavailable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        with pytest.raises(RuntimeError, match="judge call failed"):
            await hook._score_one(client, "http://judge/generate", "q", "1", "Answer: 1")


def test_deferred_topk_uses_model_vocabulary_without_mutating_training_args(monkeypatch):
    from relax.utils import misc

    monkeypatch.setattr(
        misc, "get_hf_config", lambda path: SimpleNamespace(text_config=SimpleNamespace(vocab_size=17))
    )
    args = SimpleNamespace(hf_checkpoint="student", vocab_size=32)
    session = DeferredOpdSession(
        args,
        0,
        None,
        SimpleNamespace(is_topk=True, schema_opd_transfer_data=lambda: ["opd_topk_token_ids"]),
        publish=AsyncMock(),
    )
    assert session.executor.args.vocab_size == 17
    assert args.vocab_size == 32


@pytest.mark.asyncio
async def test_shared_teacher_scores_all_batches_before_returning_to_student(monkeypatch):
    from relax.engine.rollout import deferred_opd
    from relax.engine.rollout.deferred import PublicationReceipt, PublicationState

    monkeypatch.setattr(deferred_opd, "deferred_opd_active", lambda args: True)
    events = []

    async def switch(phase):
        events.append(phase)

    async def teacher(samples):
        events.append(("teacher score", samples[0].index))
        for sample in samples:
            sample.teacher_log_probs = [-0.1] * sample.response_length
        return [True] * len(samples)

    async def student(samples, *_):
        events.append(("student score", samples[0].index))

    async def publish(*args, is_last, operation_id):
        events.append(("publish", is_last))
        return PublicationReceipt(operation_id, PublicationState.CONFIRMED)

    args = SimpleNamespace(
        _inference_phase_coordinator=SimpleNamespace(switch=SimpleNamespace(remote=switch)),
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        colocate=True,
        resource={"actor": [1, 1], "rollout": [1, 1], "teacher": [1, 1]},
    )
    manager = SimpleNamespace(
        schema_opd_transfer_data=lambda: ["teacher_log_probs"],
        prepare_teacher_inputs=AsyncMock(),
        score_teacher=teacher,
        needs_student_prefill=True,
        score_student_at_teacher=student,
        assemble_transfer=lambda samples: None,
    )
    session = DeferredOpdSession.maybe_create(args, 0, None, manager, publish=publish)
    for index in range(2):
        sample = Sample(index=index, tokens=[1, 2], response_length=1)
        await session.transfer(args, [[sample]], index, 0, None, is_last=index == 1)
    await session.flush()
    assert events == [
        "teacher",
        ("teacher score", 0),
        ("teacher score", 1),
        "inference",
        ("student score", 0),
        ("student score", 1),
        ("publish", False),
        ("publish", True),
    ]


@pytest.mark.asyncio
async def test_teacher_failure_with_valid_stale_target_prevents_publication(monkeypatch):
    from relax.engine.rollout import deferred_opd

    monkeypatch.setattr(deferred_opd, "deferred_opd_active", lambda args: True)
    switch = AsyncMock()
    args = SimpleNamespace(
        _inference_phase_coordinator=SimpleNamespace(switch=SimpleNamespace(remote=switch)),
        use_opd=True,
        opd_type="sglang",
        teacher_hf_checkpoint="teacher",
        colocate=True,
        resource={"actor": [1, 1], "rollout": [1, 1], "teacher": [1, 1]},
    )
    manager = SimpleNamespace(
        schema_opd_transfer_data=lambda: ["teacher_log_probs"],
        prepare_teacher_inputs=AsyncMock(),
        score_teacher=AsyncMock(return_value=[False]),
        needs_student_prefill=True,
        score_student_at_teacher=AsyncMock(),
    )
    sample = Sample(index=1, tokens=[1, 2], response_length=1, teacher_log_probs=[-0.1])
    publish = AsyncMock()
    session = DeferredOpdSession.maybe_create(args, 0, None, manager, publish=publish)
    await session.transfer(args, [[sample]], 0, 0, None, is_last=True)
    with pytest.raises(RuntimeError, match=r"1 sample\(s\) were not scored"):
        await session.flush()
    manager.score_teacher.assert_awaited_once_with([sample])
    manager.score_student_at_teacher.assert_not_awaited()
    publish.assert_not_awaited()
    assert [call.args for call in switch.await_args_list] == [("teacher",), ("inference",)]
