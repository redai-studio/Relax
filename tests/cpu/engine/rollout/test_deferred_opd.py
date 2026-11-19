# Copyright (c) 2026 Relax Authors. All Rights Reserved.

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
    from relax.utils.utils import post_process_rewards

    args = SimpleNamespace(
        _genrm_instances_resolved={"judge": {}},
        defer_reward_to_post_process=True,
        _genrm_colocate_with_rollout=True,
        custom_reward_post_process_path="test.hook",
        reward_key=None,
        _inference_phase_coordinator=object(),
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
        events.append("publish")
        assert post_process_rewards(args, payload[0]) == [0.25]
        assert is_last
        return PublicationReceipt(operation_id, PublicationState.CONFIRMED)

    monkeypatch.setattr(scoring_phase, "async_scoring_phase", phase)
    monkeypatch.setattr(misc, "load_function", lambda path: hook)
    session = DeferredOpdSession.maybe_create(args, 0, None, None, publish=publish)
    assert session is not None
    await session.transfer(args, [[Sample(index=1, response_length=1)]], 0, 0, None, is_last=True)
    if fail:
        with pytest.raises(RuntimeError, match="did not publish"):
            await session.flush()
        assert events == ["enter", "score", "leave"]
    else:
        await session.flush()
        assert events == ["enter", "score", "leave", "publish"]


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


@pytest.mark.parametrize("distributed_post", [False, True])
def test_legacy_genrm_hook_keeps_local_swap(monkeypatch, distributed_post):
    import ray

    from examples.generate_reward_model import post_process_genrm_swap
    from relax.distributed.ray import rollout
    from relax.distributed.ray.inference_handle import use_managed_rollout
    from relax.utils import misc
    from relax.utils.utils import post_process_rewards

    events = []

    async def score(samples):
        events.append("score")
        return [0.75]

    def legacy_hook(args, samples):
        return post_process_genrm_swap.custom_reward_post_process(args, samples)

    judge = SimpleNamespace(
        onload=SimpleNamespace(remote=lambda: events.append("judge onload")),
        offload=SimpleNamespace(remote=lambda: events.append("judge offload")),
    )
    monkeypatch.setattr(ray, "get", lambda value: value)
    monkeypatch.setattr(ray, "get_actor", lambda name: judge)
    monkeypatch.setattr(
        rollout,
        "get_local_rollout_manager",
        lambda: SimpleNamespace(
            _offload_local=lambda: events.append("rollout offload"),
        ),
    )
    monkeypatch.setattr(post_process_genrm_swap, "_score_all", score)
    monkeypatch.setattr(misc, "load_function", lambda path: legacy_hook)
    monkeypatch.setattr("relax.utils.utils.load_function", lambda path: legacy_hook)
    args = SimpleNamespace(
        _genrm_instances_resolved={"__default__": {}},
        _genrm_colocate_with_rollout=True,
        defer_reward_to_post_process=True,
        custom_reward_post_process_path="test.legacy_hook",
        use_distributed_post=distributed_post,
        advantage_estimator="grpo",
        rewards_normalization=False,
    )
    assert not use_managed_rollout(args)
    assert DeferredOpdSession.maybe_create(args, 0, None, None, publish=None) is None
    assert post_process_rewards(args, [Sample(response_length=1)]) == [0.75]
    assert events == ["rollout offload", "judge onload", "score", "judge offload"]
