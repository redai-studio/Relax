# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.engine.rollout import on_policy_distillation as opd
from relax.engine.rollout import sglang_rollout
from relax.utils.opd.opd_main_worker import LogprobResponse
from relax.utils.types import Sample


@pytest.fixture
def rollout(monkeypatch):
    args = Namespace(
        use_opd=True,
        opd_type="sglang",
        group_rm=False,
        partial_rollout=False,
        custom_generate_function_path=None,
        opd_token_selection="student_sampled",
        opd_teacher_timeout_s=10,
        opd_teacher_key="data_source",
        opd_teacher_routes_map={"A": ["http://a/generate"], "B": ["http://b/generate"]},
    )
    manager = opd.OpdManager(args)
    state = SimpleNamespace(
        aborted=False, opd_manager=manager, dp_rank_context=nullcontext, semaphore=asyncio.Semaphore(3)
    )
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _: state)
    return args, manager


def _counts(manager, teacher):
    metrics = manager.response_stats.metrics()
    prefix = f"mopd/by_teacher/{teacher}/"
    return metrics[prefix + "responses"], metrics[prefix + "awaiting_submission"]


async def _wait_for_generation(started, task):
    done, _ = await asyncio.wait((started, task), timeout=5, return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        task.result()
    assert started in done, "Timed out waiting for Student generation to start"


@pytest.mark.asyncio
async def test_out_of_order_generation_keeps_bound_teacher(monkeypatch, rollout):
    args, manager = rollout
    a1 = Sample(group_index=0, index=0, metadata={"data_source": "A"})
    a2 = Sample(group_index=0, index=1, metadata={"data_source": "A"})
    b = Sample(group_index=1, index=2, metadata={"data_source": "B"})
    started_a = set()
    both_a_started = asyncio.get_running_loop().create_future()
    finish_a = asyncio.Event()
    retry = False
    calls = []

    async def generate(args, sample, sampling_params, evaluation):
        source = sample.metadata["data_source"]
        assert sample.metadata.get("teacher_id") == source, f"Expected teacher_id={source!r} before Student generation"
        if source == "A" and not retry:
            started_a.add(sample.index)
            if len(started_a) == 2:
                both_a_started.set_result(None)
            await finish_a.wait()
            if sample is a2:
                raise sglang_rollout.GenerationAborted()
        if retry:
            assert sample is a2
            assert _counts(manager, "A") == (2, 1)
        sample.status = Sample.Status.COMPLETED
        sample.tokens = [10, 20]
        sample.response_length = 1
        return sample

    async def reward(args, sample, *, evaluation=False):
        assert evaluation is False
        return 1.0

    async def post(session, url, payload, sample, tag, expected_ids):
        calls.append((sample.index, url))
        if sample is b:
            assert _counts(manager, "A") == (2, 2)
            assert _counts(manager, "B") == (1, 0)
        score = {"http://a/generate": -0.25, "http://b/generate": -0.75}[url]
        return LogprobResponse({"meta_info": {"input_token_logprobs": [[None, 10], [score, 20]]}})

    monkeypatch.setattr(sglang_rollout, "generate", generate)
    monkeypatch.setattr(sglang_rollout, "async_rm", reward)
    monkeypatch.setattr(manager, "_post_logprob", post)
    task_a = asyncio.create_task(sglang_rollout.generate_and_rm_group(args, [a1, a2], {}))
    try:
        await _wait_for_generation(both_a_started, task_a)
        await asyncio.wait_for(sglang_rollout.generate_and_rm_group(args, [b], {}), timeout=5)
        assert calls == [(2, "http://b/generate")]
        finish_a.set()
        await asyncio.wait_for(task_a, timeout=5)
    finally:
        finish_a.set()
        task_a.cancel()
        await asyncio.gather(task_a, return_exceptions=True)

    assert calls == [(2, "http://b/generate"), (0, "http://a/generate")]
    assert [s.teacher_log_probs for s in (a1, a2, b)] == [[-0.25], None, [-0.75]]
    assert a2.status == Sample.Status.ABORTED
    assert _counts(manager, "A") == (2, 0)

    retry = True
    admit = Mock(wraps=manager.response_stats.admit)
    monkeypatch.setattr(manager.response_stats, "admit", admit)
    await sglang_rollout.generate_and_rm_group(args, [a1, a2], {})
    admit.assert_called_once_with(a2.index, "A")
    assert calls == [(2, "http://b/generate"), (0, "http://a/generate"), (1, "http://a/generate")]
    assert a2.teacher_log_probs == [-0.25]
    assert _counts(manager, "A") == (2, 0)


@pytest.mark.asyncio
async def test_group_retry_scores_both_responses_but_only_regenerates_aborted_member(monkeypatch, rollout):
    args, manager = rollout
    args.group_rm = True
    group = [Sample(group_index=0, index=i, metadata={"data_source": "A"}) for i in (0, 1)]

    generated = []

    async def generate(args, sample, sampling_params, evaluation):
        generated.append(sample.index)
        if sample.index == 1 and sample.status == Sample.Status.PENDING:
            raise sglang_rollout.GenerationAborted()
        sample.response = "answer"
        sample.tokens = [10, 20]
        sample.response_length = 1
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(sglang_rollout, "generate", generate)
    result = await sglang_rollout.generate_and_rm_group(args, group, {})
    assert [s.status for s in result] == [Sample.Status.COMPLETED, Sample.Status.ABORTED]
    assert all(s.reward is None and s.teacher_log_probs is None for s in group)
    assert _counts(manager, "A") == (2, 0)

    async def group_reward(args, samples, *, evaluation=False):
        assert evaluation is False
        assert _counts(manager, "A") == (2, 2)
        return [1.0, 1.0]

    monkeypatch.setattr(sglang_rollout, "batched_async_rm", group_reward)
    teacher_reply = LogprobResponse({"meta_info": {"input_token_logprobs": [[None, 10], [-0.25, 20]]}})
    monkeypatch.setattr(manager, "_post_logprob", AsyncMock(return_value=teacher_reply))
    generated.clear()
    await sglang_rollout.generate_and_rm_group(args, group, {})
    assert generated == [1]
    assert [sample.teacher_log_probs for sample in group] == [[-0.25], [-0.25]]
    assert _counts(manager, "A") == (2, 0)


@pytest.mark.asyncio
async def test_empty_response_leaves_awaiting_submission_at_prefill(rollout, monkeypatch):
    _, manager = rollout
    sample = Sample(index=0, metadata={"teacher_id": "A"}, response_length=0)
    post = AsyncMock()
    monkeypatch.setattr(manager, "_post_logprob", post)

    manager.response_stats.admit(sample.response_key, "A")
    assert _counts(manager, "A") == (1, 1)
    await manager.prefill(sample)
    assert _counts(manager, "A") == (1, 0)
    post.assert_not_awaited()


def test_response_window_carries_only_pending_responses(rollout):
    _, manager = rollout
    stats = manager.response_stats
    stats.admit(0, "A")
    stats.admit(1, "B")
    stats.finish(0)
    stats.new_window()
    assert _counts(manager, "A") == (0, 0)
    assert _counts(manager, "B") == (1, 1)
