# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from relax.engine.rollout import on_policy_distillation as opd
from relax.engine.rollout import sglang_rollout
from relax.utils.opd.opd_main_worker import LogprobResponse
from relax.utils.types import Sample


def _args(**overrides) -> Namespace:
    args = {
        "group_rm": False,
        "partial_rollout": False,
        "custom_generate_function_path": None,
        "opd_token_selection": "student_sampled",
        "opd_teacher_timeout_s": 10,
        "opd_teacher_routes_map": {"A": ["http://a/generate"], "B": ["http://b/generate"]},
        "opd_teacher_url": "http://single/generate",
    }
    return Namespace(**(args | overrides))


def _sample(source: str, group: int, index: int) -> Sample:
    return Sample(group_index=group, index=index, metadata={"data_source": source, "_source_mixture_batch_id": 37})


def _counts(manager: opd.OpdManager, teacher: str) -> tuple[int, ...]:
    metrics = manager.response_stats.metrics()
    return tuple(metrics[f"mopd/by_teacher/{teacher}/{field}"] for field in ("responses", "awaiting_submission"))


@pytest.fixture
def rollout(monkeypatch):
    args = _args()
    manager = opd.OpdManager(args)
    state = SimpleNamespace(
        aborted=False, opd_manager=manager, dp_rank_context=nullcontext, semaphore=asyncio.Semaphore(3)
    )
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _: state)
    return args, manager


@pytest.mark.asyncio
async def test_out_of_order_teacher_scores_cancel_and_retry(monkeypatch, rollout):
    args, manager = rollout
    all_started, finish_a = asyncio.Event(), asyncio.Event()
    retry_started = asyncio.Event()
    labels_at_generation = {}
    cancel_second_a = True

    async def generate(args, sample, sampling_params, evaluation):
        labels_at_generation[sample.index] = sample.metadata.get("teacher_id")
        if len(labels_at_generation) == 3:
            all_started.set()
        await all_started.wait()
        if sample.metadata["data_source"] == "A":
            if not cancel_second_a:
                retry_started.set()
            await finish_a.wait()
            if sample.index == 1 and cancel_second_a:
                raise sglang_rollout.GenerationAborted()
        sample.status = Sample.Status.COMPLETED
        sample.tokens = [10, 20 + sample.index]
        sample.response_length = 1
        return sample

    calls = []

    async def post(session, url, payload, sample, tag, expected_ids):
        calls.append((sample.index, url))
        if sample.index == 2:
            assert _counts(manager, "B") == (1, 0)
        score = {"http://a/generate": -0.25, "http://b/generate": -0.75}[url]
        return LogprobResponse({"meta_info": {"input_token_logprobs": [[None, 10], [score, sample.tokens[-1]]]}})

    async def reward(args, sample):
        return 1.0

    monkeypatch.setattr(sglang_rollout, "generate", generate)
    monkeypatch.setattr(sglang_rollout, "async_rm", reward)
    monkeypatch.setattr(manager, "_post_logprob", post)

    a = [_sample("A", 0, 0), _sample("A", 0, 1)]
    b = [_sample("B", 1, 2)]
    a_task = asyncio.create_task(sglang_rollout.generate_and_rm_group(args, a, {}))
    try:
        await asyncio.wait_for(sglang_rollout.generate_and_rm_group(args, b, {}), timeout=5)
        assert labels_at_generation == {0: "A", 1: "A", 2: "B"}
        assert _counts(manager, "A") == (2, 2)
        assert _counts(manager, "B") == (1, 0)
    finally:
        all_started.set()
        finish_a.set()
        await asyncio.wait_for(a_task, timeout=5)

    assert calls == [(2, "http://b/generate"), (0, "http://a/generate")]
    assert [s.teacher_log_probs for s in (*a, *b)] == [[-0.25], None, [-0.75]]
    assert a[1].status == Sample.Status.ABORTED
    assert _counts(manager, "A") == (2, 0)
    assert _counts(manager, "B") == (1, 0)
    for sample, expected_source in zip((*a, *b), ("A", "A", "B"), strict=True):
        assert sample.metadata["teacher_id"] == sample.metadata["data_source"] == expected_source
        assert sample.metadata["_source_mixture_batch_id"] == 37
    assert [(s.group_index, s.index) for s in (*a, *b)] == [(0, 0), (0, 1), (1, 2)]

    # Resume this prompt's aborted response within the same rollout window.
    cancel_second_a = False
    finish_a.clear()
    retry = asyncio.create_task(sglang_rollout.generate_and_rm_group(args, a, {}))
    try:
        await asyncio.wait_for(retry_started.wait(), timeout=5)
        assert _counts(manager, "A") == (2, 1)
    finally:
        finish_a.set()
        await asyncio.wait_for(retry, timeout=5)
    assert calls == [(2, "http://b/generate"), (0, "http://a/generate"), (1, "http://a/generate")]
    assert a[1].teacher_log_probs == [-0.25]
    assert _counts(manager, "A") == (2, 0)


@pytest.mark.asyncio
async def test_cancelled_response_no_longer_awaits_submission(monkeypatch, rollout):
    args, manager = rollout
    entered = asyncio.Event()

    async def generate(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(sglang_rollout, "generate", generate)
    task = asyncio.create_task(sglang_rollout.generate_and_rm_group(args, [_sample("A", 0, 0)], {}))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert _counts(manager, "A") == (1, 1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert _counts(manager, "A") == (1, 0)


def test_single_teacher_keeps_existing_route():
    args = _args(opd_teacher_routes_map={})
    sample = Sample(index=0)
    assert opd.OpdManager(args).response_stats is None
    assert opd._pick_teacher_url(args, sample) == "http://single/generate"
    assert "teacher_id" not in sample.metadata


@pytest.mark.asyncio
async def test_group_reward_closes_aborted_response_while_sibling_is_generating(monkeypatch, rollout):
    args, manager = rollout
    args.group_rm = True
    sibling_started, aborted, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def generate(args, sample, sampling_params, evaluation):
        if sample.index == 0:
            await sibling_started.wait()
            aborted.set()
            raise sglang_rollout.GenerationAborted()
        sibling_started.set()
        await finish.wait()
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(sglang_rollout, "generate", generate)
    group = [_sample("A", 0, 0), _sample("A", 0, 1)]
    task = asyncio.create_task(sglang_rollout.generate_and_rm_group(args, group, {}))
    try:
        await asyncio.wait_for(aborted.wait(), timeout=5)
        assert _counts(manager, "A") == (2, 1)
    finally:
        finish.set()
        await asyncio.wait_for(task, timeout=5)
    assert _counts(manager, "A") == (2, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected_url"),
    [
        ({"teacher_id": "B"}, "http://b/generate"),
        ({"teacher_id": "B", "data_source": "A"}, "http://b/generate"),
        ({"data_source": "A"}, None),
    ],
)
async def test_teacher_prefill_routes_from_sample_without_response_stats(monkeypatch, rollout, metadata, expected_url):
    _, manager = rollout
    manager.response_stats = None
    sample = Sample(index=3, tokens=[10, 20], response_length=1, metadata=metadata)

    async def post(session, url, payload, sample, tag, expected_ids):
        assert url == expected_url
        return LogprobResponse({"meta_info": {"input_token_logprobs": [[None, 10], [-0.5, 20]]}})

    monkeypatch.setattr(manager, "_post_logprob", post)
    if expected_url is None:
        with pytest.raises(KeyError, match="teacher_id"):
            await manager.prefill(sample)
    else:
        await manager.prefill(sample)
        assert sample.teacher_log_probs == [-0.5]


@pytest.mark.asyncio
async def test_group_reward_counts_replacement_sample_against_original_response(monkeypatch, rollout):
    args, manager = rollout
    args.group_rm = True
    replacement = Sample(index=99, tokens=[10, 20], response_length=1, status=Sample.Status.COMPLETED)

    async def generate(args, sample, sampling_params, evaluation):
        assert sample.metadata["teacher_id"] == "A"
        return replacement

    async def reward(args, group):
        return [1.0]

    async def post(session, url, payload, sample, tag, expected_ids):
        assert url == "http://a/generate"
        return LogprobResponse({"meta_info": {"input_token_logprobs": [[None, 10], [-0.5, 20]]}})

    monkeypatch.setattr(sglang_rollout, "generate", generate)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", reward)
    monkeypatch.setattr(manager, "_post_logprob", post)
    (result,) = await sglang_rollout.generate_and_rm_group(args, [_sample("A", 0, 1)], {})
    assert result is replacement
    assert result.metadata["teacher_id"] == result.metadata["data_source"] == "A"
    assert result.teacher_log_probs == [-0.5]
    assert _counts(manager, "A") == (1, 0)
