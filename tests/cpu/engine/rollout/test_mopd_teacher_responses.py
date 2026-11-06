# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import base64
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
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
@pytest.mark.parametrize("group_rm", [False, True])
@pytest.mark.parametrize("replace", [False, True], ids=["mutate-index", "replace-sample"])
async def test_scoring_uses_original_response_key(monkeypatch, rollout, group_rm, replace):
    args, manager = rollout
    args.group_rm = group_rm
    original = _sample("A", 0, 1)

    async def generate(args, sample, sampling_params, evaluation):
        assert sample.metadata["teacher_id"] == "A"
        result = Sample() if replace else sample
        result.index = 99
        result.tokens = [10, 20]
        result.response_length = 1
        result.status = Sample.Status.COMPLETED
        result.reward = 1.0
        return result

    async def reward(args, group):
        return [1.0]

    async def post(session, url, payload, sample, tag, expected_ids):
        assert url == "http://a/generate"
        assert _counts(manager, "A") == (1, 0)
        return LogprobResponse({"meta_info": {"input_token_logprobs": [[None, 10], [-0.5, 20]]}})

    monkeypatch.setattr(sglang_rollout, "generate", generate)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", reward)
    monkeypatch.setattr(manager, "_post_logprob", post)
    (result,) = await sglang_rollout.generate_and_rm_group(args, [original], {})
    assert (result is original) == (not replace)
    assert result.index == 99
    assert result.metadata["teacher_id"] == result.metadata["data_source"] == "A"
    assert result.teacher_log_probs == [-0.5]
    assert _counts(manager, "A") == (1, 0)
    manager.response_stats.new_window()
    assert _counts(manager, "A") == (0, 0)


def _shared_manager(monkeypatch, score, **args):
    locator = {"name": "shared", "namespace": "test"}
    manager = opd.OpdManager(_args(opd_shared_teacher=True, opd_shared_teacher_actor=locator, **args))

    def get_actor(name, namespace):
        assert (name, namespace) == ("shared", "test")
        return SimpleNamespace(score=SimpleNamespace(remote=score))

    monkeypatch.setattr("ray.get_actor", get_actor)
    return manager


@pytest.mark.asyncio
async def test_shared_teacher_returns_scores_to_bound_responses_out_of_order(monkeypatch):
    a, b = _sample("A", 10, 0), _sample("B", 20, 1)
    a_entered, release_a = asyncio.Event(), asyncio.Event()

    async def score(teacher, payload):
        sample = {2: a, 3: b}[payload["input_ids"][-1]]
        assert teacher == sample.metadata["teacher_id"]
        assert payload["input_ids"] == sample.tokens
        if teacher == "A":
            a_entered.set()
            await release_a.wait()
        logprob = -0.25 if teacher == "A" else -0.75
        return {
            "_relax_teacher_route": teacher,
            "meta_info": {"input_token_logprobs": [[None, 1], [logprob, sample.tokens[-1]]]},
        }

    manager = _shared_manager(monkeypatch, score)
    for sample in (a, b):
        manager.bind_teacher(sample)
        manager.response_stats.admit(sample.index, sample.metadata["teacher_id"])
        # The bound teacher survives a later metadata projection.
        del sample.metadata["data_source"]
        sample.tokens, sample.response_length = [1, 2 + sample.index], 1

    task = asyncio.create_task(manager.prefill(a))
    try:
        await asyncio.wait_for(a_entered.wait(), 5)
        await asyncio.wait_for(manager.prefill(b), 5)
        assert _counts(manager, "A") == _counts(manager, "B") == (1, 0)
        assert a.teacher_log_probs is None
        assert b.teacher_log_probs == [-0.75]
    finally:
        release_a.set()
        await asyncio.wait_for(task, 5)
    assert a.teacher_log_probs == [-0.25]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_error", [None, RuntimeError("actor unavailable")])
async def test_cancelling_shared_prefill_cancels_only_its_ray_score(monkeypatch, cancel_error):
    # Like an ObjectRef, cancelling the local await leaves the remote work running.
    remote_work = asyncio.get_running_loop().create_future()
    score_ref = asyncio.shield(remote_work)
    submitted = asyncio.Event()

    def score(teacher, payload):
        submitted.set()
        return score_ref

    cancel = Mock(side_effect=cancel_error)
    monkeypatch.setattr("ray.cancel", cancel)
    manager = _shared_manager(monkeypatch, score)
    sample = _sample("A", 10, 0)
    manager.bind_teacher(sample)
    sample.tokens, sample.response_length = [1, 2], 1

    task = asyncio.create_task(manager.prefill(sample))
    try:
        await asyncio.wait_for(submitted.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        cancel.assert_called_once_with(score_ref, recursive=False)
        assert not remote_work.done()
        assert sample.teacher_log_probs is None
    finally:
        remote_work.set_result(None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bad_data", "error"),
    [
        ({"_relax_teacher_route": "A"}, "different route"),
        ({"_relax_teacher_route": "B", "meta_info": {"input_token_logprobs": []}}, "complete logprobs"),
    ],
)
async def test_bad_shared_response_rejects_batch_with_successful_sibling(monkeypatch, bad_data, error):
    async def score(teacher, payload):
        if teacher == "B":
            return bad_data
        return {"_relax_teacher_route": teacher, "meta_info": {"input_token_logprobs": [[None, 1], [-0.5, 2]]}}

    manager = _shared_manager(monkeypatch, score)
    a, b = _sample("A", 10, 0), _sample("B", 20, 1)
    for sample in [a, b]:
        manager.bind_teacher(sample)
        sample.tokens, sample.response_length = [1, 2], 1
    with pytest.raises(RuntimeError, match=error):
        await manager.prefill([a, b])
    assert b.teacher_log_probs is None


@pytest.mark.asyncio
async def test_shared_teacher_topk_queries_student_via_original_router(monkeypatch):
    score = AsyncMock(
        return_value={
            "_relax_teacher_route": "A",
            "meta_info": {"input_token_logprobs": [[None, 1], [-0.5, 2]], "input_top_logprobs": [None, [[-0.2, 3]]]},
        }
    )
    manager = _shared_manager(
        monkeypatch,
        score,
        opd_token_selection="teacher_topk",
        opd_log_prob_top_k=1,
        opd_kl_coef=1.0,
        opd_loss_coef=0.0,
        sglang_router_ip="student",
        sglang_router_port=80,
    )
    direct = AsyncMock(return_value=LogprobResponse({"meta_info": {"input_token_ids_logprobs": [None, [[-0.25, 3]]]}}))
    monkeypatch.setattr(manager, "_post_logprob", direct)
    sample = _sample("A", 10, 0)
    manager.bind_teacher(sample)
    sample.tokens, sample.response_length = [1, 2], 1
    await manager.prefill(sample)
    assert score.await_count == direct.await_count == 1
    assert direct.await_args.args[1] == "http://student:80/generate"
    assert sample.student_at_teacher_topk_log_probs.tolist() == [[-0.25]]


@pytest.mark.asyncio
@pytest.mark.parametrize("shared", [False, True])
async def test_text_topk_accepts_legacy_response_tail_with_either_transport(monkeypatch, shared):
    def packed(values, dtype):
        return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode()

    # Older text engines include prefix rows, and cross-topk may return values alone.
    result = {
        "_relax_teacher_route": "A",
        "meta_info": {
            "input_token_logprobs": [[None, 1], [-0.5, 2]],
            "input_top_logprobs_val_b64": packed([-1, -2, -0.1, -0.2], "float32"),
            "input_top_logprobs_idx_b64": packed([99, 98, 0, 5], "int32"),
            "input_token_ids_logprobs_val_b64": packed([-1, -2, -0.3, -0.4], "float32"),
        },
    }
    config = dict(opd_token_selection="union", opd_log_prob_top_k=2, opd_kl_coef=1.0, opd_loss_coef=0.0)
    if shared:
        manager = _shared_manager(monkeypatch, AsyncMock(return_value=result), **config)
    else:
        manager = opd.OpdManager(_args(**config))
        monkeypatch.setattr(manager, "_post_logprob", AsyncMock(return_value=LogprobResponse(result)))
    sample = _sample("A", 10, 0)
    manager.bind_teacher(sample)
    sample.tokens, sample.response_length = [1, 2], 1
    sample.student_topk_token_ids = np.array([[5, 7]], dtype=np.int32)
    assert await manager._teacher_prefill(sample, None)
    np.testing.assert_array_equal(sample.teacher_topk_token_ids, [[0, 5]])
    np.testing.assert_allclose(sample.teacher_topk_log_probs, [[-0.1, -0.2]])
    np.testing.assert_allclose(sample.teacher_at_student_topk_log_probs, [[-0.3, -0.4]])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [None, [None, [[-0.5, -1]]], [None, [[float("nan"), 5]]]],
)
async def test_invalid_shared_text_topk_rejects_sample_before_writeback(monkeypatch, rows):
    data = {
        "_relax_teacher_route": "A",
        "meta_info": {"input_token_logprobs": [[None, 1], [-0.5, 2]], "input_top_logprobs": rows},
    }
    manager = _shared_manager(
        monkeypatch,
        AsyncMock(return_value=data),
        opd_token_selection="teacher_topk",
        opd_log_prob_top_k=1,
        opd_kl_coef=1.0,
        opd_loss_coef=0.0,
    )
    sample = _sample("A", 10, 0)
    manager.bind_teacher(sample)
    sample.tokens, sample.response_length = [1, 2], 1
    with pytest.raises(ValueError, match="input_top_logprobs"):
        await manager._teacher_prefill(sample, None)
    assert sample.teacher_topk_token_ids is None and sample.teacher_topk_log_probs is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selection", "meta", "expected_error"),
    [
        (
            "student_topk",
            {
                "input_token_ids_logprobs_shape": [1, 2],
                "input_token_ids_logprobs_null_prefix": 1,
                "input_token_ids_logprobs_val_flat": [-0.1, -0.3],
                "input_token_ids_logprobs_idx_flat": [7, 5],
            },
            "query token IDs differ",
        ),
        ("teacher_topk", {"input_top_logprobs": [None, [[-0.2, 5]]]}, "incomplete"),
        ("teacher_topk", {"input_top_logprobs": [[[-0.1, 5], [-0.2, 7]], None]}, "incomplete"),
    ],
)
async def test_shared_text_topk_does_not_reorder_or_pad_candidates(monkeypatch, selection, meta, expected_error):
    reply = {"_relax_teacher_route": "A", "meta_info": {"input_token_logprobs": [[None, 1], [-0.5, 2]], **meta}}
    manager = _shared_manager(
        monkeypatch,
        AsyncMock(return_value=reply),
        opd_token_selection=selection,
        opd_log_prob_top_k=2,
        opd_kl_coef=1.0,
        opd_loss_coef=0.0,
    )
    sample = _sample("A", 10, 0)
    manager.bind_teacher(sample)
    sample.tokens, sample.response_length = [1, 2], 1
    sample.student_topk_token_ids = np.array([[5, 7]], dtype=np.int32)
    with pytest.raises(ValueError, match=expected_error):
        await manager._teacher_prefill(sample, None)
    assert sample.teacher_at_student_topk_log_probs is None and sample.teacher_topk_token_ids is None
