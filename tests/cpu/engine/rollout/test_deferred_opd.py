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
