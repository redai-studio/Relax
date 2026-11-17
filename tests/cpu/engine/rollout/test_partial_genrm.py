# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Judge carry contracts, with only tokenizer/Ray discovery/HTTP boundaries
faked."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from relax.engine.rollout import partial_genrm
from relax.engine.rollout.partial_genrm import PartialGenRM, PartialGenRMAborted
from relax.utils.types import Sample


MESSAGES = [{"role": "user", "content": "Score this response"}]


@pytest.fixture
def judge(monkeypatch):
    args = Namespace(genrm_model_path="judge", genrm_sampling_config={"max_response_len": 5})
    state = SimpleNamespace(abort_event=asyncio.Event())
    helper = PartialGenRM(args, state)
    helper.tokenizers["__default__"] = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: [10, 11], eos_token_id=99
    )
    monkeypatch.setattr(helper, "_get_engine_urls", AsyncMock(return_value=["http://judge"]))
    return helper


def _chunk(text, ids, reason):
    return {"text": text, "output_ids": ids, "meta_info": {"finish_reason": {"type": reason}}}


@pytest.mark.asyncio
async def test_partial_genrm_replays_completed_calls_and_resumes_prefix(judge, monkeypatch):
    post = AsyncMock(side_effect=[_chunk("first", [1], "stop"), _chunk("pre", [2, 3], "abort")])
    monkeypatch.setattr(partial_genrm, "post", post)
    sample = Sample(status=Sample.Status.TRUNCATED)
    generate = judge.bind(sample)
    assert await generate(MESSAGES) == "first"
    with pytest.raises(PartialGenRMAborted):
        await generate(MESSAGES, {"temperature": 0})

    sample.status = Sample.Status.ABORTED
    assert judge.restore(sample)
    assert sample.status == Sample.Status.TRUNCATED
    # Exercise a new rollout's helper; carry must live on Sample, not the helper.
    resumed = PartialGenRM(judge.args, judge.state)
    resumed.tokenizers = judge.tokenizers
    monkeypatch.setattr(resumed, "_get_engine_urls", judge._get_engine_urls)
    post.reset_mock(side_effect=True)
    post.return_value = _chunk("fix", [4], "stop")
    generate = resumed.bind(sample)
    assert await generate(MESSAGES) == "first"
    assert post.await_count == 0
    assert await generate(MESSAGES, {"temperature": 0}) == "prefix"
    payload = post.call_args.args[1]
    assert payload["input_ids"] == [10, 11, 2, 3]
    assert payload["sampling_params"]["max_new_tokens"] == 3
    assert payload["sampling_params"]["temperature"] == 0
    resumed.clear(sample)
    assert not resumed.restore(sample)


@pytest.mark.asyncio
async def test_partial_genrm_rejects_changed_resume_input(judge, monkeypatch):
    post = AsyncMock(return_value=_chunk("pre", [1], "abort"))
    monkeypatch.setattr(partial_genrm, "post", post)
    sample = Sample(status=Sample.Status.COMPLETED)
    with pytest.raises(PartialGenRMAborted):
        await judge.bind(sample)(MESSAGES)
    with pytest.raises(ValueError, match="same order"):
        await judge.bind(sample)([{"role": "user", "content": "changed"}])
    assert post.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ignore_eos,aborted", [(False, False), (True, True)])
async def test_partial_genrm_honors_eos_on_abort(judge, monkeypatch, ignore_eos, aborted):
    post = AsyncMock(return_value=_chunk("done", [99], "abort"))
    monkeypatch.setattr(partial_genrm, "post", post)
    generate = judge.bind(Sample(status=Sample.Status.COMPLETED))
    if aborted:
        with pytest.raises(PartialGenRMAborted):
            await generate(MESSAGES, {"ignore_eos": ignore_eos})
    else:
        assert await generate(MESSAGES, {"ignore_eos": ignore_eos}) == "done"


@pytest.mark.asyncio
async def test_partial_genrm_budget_exhaustion_does_not_redispatch(judge, monkeypatch):
    # Older SGLang versions expose token IDs only through output_token_logprobs.
    post = AsyncMock(
        return_value={
            "text": "done",
            "meta_info": {"finish_reason": {"type": "abort"}, "output_token_logprobs": [[0, 7, None]]},
        }
    )
    monkeypatch.setattr(partial_genrm, "post", post)
    sample = Sample(status=Sample.Status.COMPLETED)
    with pytest.raises(PartialGenRMAborted):
        await judge.bind(sample)(MESSAGES, {"max_new_tokens": 1})
    assert await judge.bind(sample)(MESSAGES, {"max_new_tokens": 1}) == "done"
    assert post.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,message,expected",
    [
        (499, "Request abc was aborted", PartialGenRMAborted),
        (499, "upstream closed", httpx.HTTPStatusError),
        (503, "unavailable", httpx.HTTPStatusError),
    ],
)
async def test_partial_genrm_only_maps_strict_abort_499(judge, monkeypatch, status, message, expected):
    response = httpx.Response(
        status, request=httpx.Request("POST", "http://judge/generate"), json={"error": {"message": message}}
    )
    post = AsyncMock(side_effect=httpx.HTTPStatusError("failed", request=response.request, response=response))
    monkeypatch.setattr(partial_genrm, "post", post)
    with pytest.raises(expected):
        await judge.bind(Sample(status=Sample.Status.COMPLETED))(MESSAGES)
    assert not judge.active


@pytest.mark.asyncio
async def test_partial_genrm_stops_dispatch_after_discovery_abort(judge, monkeypatch):
    async def discover(_key):
        judge.state.abort_event.set()
        return ["http://judge"]

    monkeypatch.setattr(judge, "_get_engine_urls", discover)
    post = AsyncMock()
    monkeypatch.setattr(partial_genrm, "post", post)
    with pytest.raises(PartialGenRMAborted):
        await judge.bind(Sample(status=Sample.Status.COMPLETED))(MESSAGES)
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_genrm_abort_drains_late_http_dispatch(judge, monkeypatch):
    entered, released = asyncio.Event(), asyncio.Event()
    aborts = 0

    async def post(url, payload, **kwargs):
        nonlocal aborts
        if url.endswith("/abort_request"):
            aborts += 1
            # The first abort arrives before generate reaches the scheduler.
            if aborts == 2:
                released.set()
            return {}
        entered.set()
        await released.wait()
        return _chunk("pre", [1], "abort")

    monkeypatch.setattr(partial_genrm, "post", post)
    task = asyncio.create_task(judge.bind(Sample(status=Sample.Status.COMPLETED))(MESSAGES))
    await asyncio.wait_for(entered.wait(), timeout=2)
    judge.state.abort_event.set()
    await asyncio.wait_for(judge.abort_engines(), timeout=3)
    with pytest.raises(PartialGenRMAborted):
        await task
    assert aborts == 2
    assert not judge.active


def test_partial_genrm_requires_explicit_async_opt_in(monkeypatch):
    async def legacy(args, sample, **kwargs):
        return 1

    args = Namespace(partial_rollout=True, custom_rm_path="reward", genrm_model_path="judge")
    monkeypatch.setattr(partial_genrm, "load_function", lambda path: legacy)
    assert PartialGenRM.from_args(args, None) is None

    def sync(args, sample, genrm_generate=None):
        return 1

    monkeypatch.setattr(partial_genrm, "load_function", lambda path: sync)
    with pytest.raises(ValueError, match="async"):
        PartialGenRM.from_args(args, None)

    async def reward(args, sample, genrm_generate=None):
        return 1

    monkeypatch.setattr(partial_genrm, "load_function", lambda path: reward)
    assert PartialGenRM.from_args(args, None) is not None
    args.group_rm = True
    with pytest.raises(ValueError, match="per-sample"):
        PartialGenRM.from_args(args, None)
    args.partial_rollout = False
    assert PartialGenRM.from_args(args, None) is None
