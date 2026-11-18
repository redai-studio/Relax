# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Judge carry contracts, with only tokenizer/Ray discovery/HTTP boundaries
faked."""

import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from relax.engine.rollout import partial_genrm
from relax.engine.rollout.partial_genrm import PartialGenRM, PartialGenRMAborted
from relax.utils.types import Sample


MESSAGES = [{"role": "user", "content": "Score this response"}]


@pytest.fixture
async def judge(monkeypatch):
    args = Namespace(genrm_model_path="judge", genrm_sampling_config={"max_response_len": 5})
    state = SimpleNamespace(abort_event=asyncio.Event())
    helper = PartialGenRM(args, state)
    helper.tokenizers["__default__"] = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: [10, 11],
        eos_token_id=99,
        decode=lambda ids, **kwargs: "".join(
            {1: "first", 2: "pr", 3: "e", 4: "fix", 7: "done", 99: "done"}[i] for i in ids
        ),
    )
    monkeypatch.setattr(helper, "_get_engine_urls", AsyncMock(return_value=["http://judge"]))
    yield helper
    await helper.aclose()


def _chunk(text, ids, reason):
    return {"text": text, "output_ids": ids, "meta_info": {"finish_reason": {"type": reason}}}


@pytest.fixture
def byte_tokenizer():
    backend = Tokenizer(
        models.BPE({char: index for index, char in enumerate(pre_tokenizers.ByteLevel.alphabet())}, [])
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", chat_template="{{ messages[0]['content'] }}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["__default__", "quality"])
async def test_partial_genrm_discovers_named_singleton_manager(judge, monkeypatch, key):
    judge.args._genrm_instances_resolved = {key: {"model_path": "judge", "sampling_config": {}}}
    name = "relax_genrm_manager" if key == "__default__" else "relax_genrm_manager_quality"
    manager = SimpleNamespace(get_engine_hosts_ports=SimpleNamespace(remote=AsyncMock(return_value=[("judge", 8000)])))
    registered = {name: manager}
    monkeypatch.setattr(partial_genrm.ray, "get_actor", registered.__getitem__)
    assert await PartialGenRM._get_engine_urls(judge, key) == ["http://judge:8000"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stop", "length", "abort"])
async def test_partial_genrm_normalizes_completed_replayed_and_budget_results(judge, monkeypatch, reason):
    post = AsyncMock(return_value=_chunk("\n1 \n", [1], reason))
    monkeypatch.setattr(partial_genrm, "_post", post)
    sample = Sample(status=Sample.Status.COMPLETED)
    params = {"max_new_tokens": 1}
    if reason == "abort":
        with pytest.raises(PartialGenRMAborted):
            await judge.bind(sample)(MESSAGES, params)
    else:
        assert await judge.bind(sample)(MESSAGES, params) == "1"
    assert await judge.bind(sample)(MESSAGES, params) == "1"
    assert await judge.bind(sample)(MESSAGES, params) == "1"
    assert post.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stop,no_stop_trim,skip_special,expected",
    [
        ("END", False, True, "€1"),
        ("END", True, True, "€1END"),
        ("<eos>", False, False, "€1"),
        ("<eos>", True, False, "€1<eos>"),
        ("<eos>", True, True, "€1"),
    ],
)
async def test_partial_genrm_resumes_multibyte_text_and_preserves_stop_options(
    judge, monkeypatch, byte_tokenizer, stop, no_stop_trim, skip_special, expected
):
    judge.tokenizers["__default__"] = byte_tokenizer
    ids = byte_tokenizer.encode("€1" + stop, add_special_tokens=False)
    assert byte_tokenizer.decode(ids[:1]) == "�"
    first = _chunk("�", ids[:1], "abort")
    last = _chunk("1" + (stop if no_stop_trim and not skip_special else ""), ids[1:], "stop")
    last["meta_info"]["finish_reason"]["matched"] = byte_tokenizer.eos_token_id if stop == "<eos>" else stop
    post = AsyncMock(side_effect=[first, last])
    monkeypatch.setattr(partial_genrm, "_post", post)
    sample = Sample(status=Sample.Status.COMPLETED)
    params = {"max_new_tokens": 20, "no_stop_trim": no_stop_trim, "skip_special_tokens": skip_special}
    with pytest.raises(PartialGenRMAborted):
        await judge.bind(sample)(MESSAGES, params)
    assert await judge.bind(sample)(MESSAGES, params) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("no_stop_trim,expected", [(False, "OK"), (True, "OKEND")])
async def test_partial_genrm_trims_characters_after_stop_inside_one_token(
    judge, monkeypatch, byte_tokenizer, no_stop_trim, expected
):
    byte_tokenizer.add_tokens(["END!"])
    stop_token = byte_tokenizer.convert_tokens_to_ids("END!")
    assert byte_tokenizer.decode([stop_token]) == "END!"
    prefix = byte_tokenizer.encode("OK", add_special_tokens=False)
    judge.tokenizers["__default__"] = byte_tokenizer
    last = _chunk("END" if no_stop_trim else "", [stop_token], "stop")
    last["meta_info"]["finish_reason"]["matched"] = "END"
    post = AsyncMock(side_effect=[_chunk("OK", prefix, "abort"), last])
    monkeypatch.setattr(partial_genrm, "_post", post)
    sample = Sample(status=Sample.Status.COMPLETED)
    params = {"max_new_tokens": 20, "no_stop_trim": no_stop_trim}
    with pytest.raises(PartialGenRMAborted):
        await judge.bind(sample)(MESSAGES, params)
    assert await judge.bind(sample)(MESSAGES, params) == expected
    assert await judge.bind(sample)(MESSAGES, params) == expected
    assert post.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"stop": "END"}, {"stop": ["END"]}, {"stop_regex": "END[.!]"}])
async def test_partial_genrm_rejects_request_local_stop_conditions_before_dispatch(judge, monkeypatch, params):
    post = AsyncMock(return_value=_chunk("END!", [1], "length"))
    monkeypatch.setattr(partial_genrm, "_post", post)
    with pytest.raises(ValueError, match="stop/stop_regex.*stop_token_ids"):
        await judge.bind(Sample(status=Sample.Status.COMPLETED))(MESSAGES, params)
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_genrm_replays_completed_calls_and_resumes_prefix(judge, monkeypatch):
    post = AsyncMock(side_effect=[_chunk("first", [1], "stop"), _chunk("pre", [2, 3], "abort")])
    monkeypatch.setattr(partial_genrm, "_post", post)
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
    payload = post.call_args.args[2]
    assert payload["input_ids"] == [10, 11, 2, 3]
    assert payload["sampling_params"]["max_new_tokens"] == 3
    assert payload["sampling_params"]["temperature"] == 0
    resumed.clear(sample)
    assert not resumed.restore(sample)
    await resumed.aclose()


@pytest.mark.asyncio
async def test_partial_genrm_rejects_changed_resume_input(judge, monkeypatch):
    post = AsyncMock(return_value=_chunk("pre", [1], "abort"))
    monkeypatch.setattr(partial_genrm, "_post", post)
    sample = Sample(status=Sample.Status.COMPLETED)
    with pytest.raises(PartialGenRMAborted):
        await judge.bind(sample)(MESSAGES)
    with pytest.raises(ValueError, match="same order"):
        await judge.bind(sample)([{"role": "user", "content": "changed"}])
    assert post.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params,aborted",
    [
        ({"ignore_eos": False}, False),
        ({"ignore_eos": True}, True),
        ({"ignore_eos": True, "stop_token_ids": [99], "stop": None, "stop_regex": []}, True),
        ({"ignore_eos": False, "stop_token_ids": [7]}, False),
        ({"ignore_eos": True, "stop_token_ids": [7]}, True),
    ],
)
async def test_partial_genrm_honors_eos_on_abort(judge, monkeypatch, params, aborted):
    token_id = (params.get("stop_token_ids") or [99])[0]
    post = AsyncMock(side_effect=[_chunk("done", [token_id], "abort"), _chunk("fix", [4], "length")])
    monkeypatch.setattr(partial_genrm, "_post", post)
    sample = Sample(status=Sample.Status.COMPLETED)
    generate = judge.bind(sample)
    if aborted:
        with pytest.raises(PartialGenRMAborted):
            await generate(MESSAGES, params)
        assert await judge.bind(sample)(MESSAGES, params) == "donefix"
        payload = post.call_args.args[2]
        assert payload["input_ids"] == [10, 11, token_id]
        assert payload["sampling_params"]["max_new_tokens"] == 4
        assert await judge.bind(sample)(MESSAGES, params) == "donefix"
        assert post.await_count == 2
    else:
        assert await generate(MESSAGES, params) == "done"
        assert await judge.bind(sample)(MESSAGES, params) == "done"
        assert post.await_count == 1


@pytest.mark.asyncio
async def test_partial_genrm_budget_exhaustion_does_not_redispatch(judge, monkeypatch):
    # Older SGLang versions expose token IDs only through output_token_logprobs.
    post = AsyncMock(
        return_value={
            "text": "done",
            "meta_info": {"finish_reason": {"type": "abort"}, "output_token_logprobs": [[0, 7, None]]},
        }
    )
    monkeypatch.setattr(partial_genrm, "_post", post)
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
    monkeypatch.setattr(partial_genrm, "_post", post)
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
    monkeypatch.setattr(partial_genrm, "_post", post)
    with pytest.raises(PartialGenRMAborted):
        await judge.bind(Sample(status=Sample.Status.COMPLETED))(MESSAGES)
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_genrm_abort_drains_late_http_dispatch(judge, monkeypatch):
    entered, released = asyncio.Event(), asyncio.Event()
    aborts = 0

    async def post(client, url, payload, **kwargs):
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

    monkeypatch.setattr(partial_genrm, "_post", post)
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
