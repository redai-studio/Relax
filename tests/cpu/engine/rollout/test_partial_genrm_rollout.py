# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise Judge carry through the real rollout and async reward executor."""

import asyncio
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from aiohttp import web

from relax.engine.rewards import RewardExecutor
from relax.engine.rollout import partial_genrm
from relax.engine.rollout.partial_genrm import PartialGenRM
from relax.utils import http_utils
from relax.utils.types import Sample


sglang_rollout = pytest.importorskip("relax.engine.rollout.sglang_rollout", exc_type=ModuleNotFoundError)


@pytest.mark.asyncio
async def test_partial_genrm_rollout_resumes_reward_without_regenerating_policy(monkeypatch):
    reward_calls = []
    module = ModuleType("partial_genrm_test_reward")

    async def reward(args, sample, genrm_generate=None):
        reward_calls.append(genrm_generate is not None)
        if genrm_generate is None:
            return 2  # The business's normal eval/client path.
        try:
            response = await genrm_generate([{"role": "user", "content": sample.response}])
        except Exception:
            return -1  # Ordinary business fallback must not swallow partial aborts.
        return len(response)

    module.reward = reward
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(RewardExecutor, "_instance", RewardExecutor(max_concurrency=1))
    args = Namespace(
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
        group_rm=False,
        custom_rm_path=f"{module.__name__}.reward",
        genrm_model_path="judge",
        genrm_sampling_config={"max_response_len": 5},
    )
    state = SimpleNamespace(abort_event=asyncio.Event())
    state.partial_genrm = PartialGenRM.from_args(args, state)
    state.partial_genrm.tokenizers["__default__"] = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: [10],
        eos_token_id=99,
        decode=lambda ids, **kwargs: "".join({1: "pre", 2: "fix"}[i] for i in ids),
    )
    monkeypatch.setattr(state.partial_genrm, "_get_engine_urls", AsyncMock(return_value=["http://judge"]))
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout.opd, "is_teacher_routing_enabled", lambda *args, **kwargs: False)
    monkeypatch.setattr(sglang_rollout.opd, "is_opd_enabled", lambda *args, **kwargs: False)

    async def policy(state, args, sample, params, **kwargs):
        sample.response = "policy"
        sample.tokens = [7, 8]
        sample.response_length = 2
        sample.rollout_log_probs = [-0.1, -0.2]
        sample.status = Sample.Status.COMPLETED
        return sample

    dispatch = AsyncMock(side_effect=policy)
    monkeypatch.setattr(sglang_rollout, "_dispatch_generate", dispatch)
    post = AsyncMock(
        side_effect=[
            {"text": "pre", "output_ids": [1], "meta_info": {"finish_reason": {"type": "abort"}}},
            {"text": "fix", "output_ids": [2], "meta_info": {"finish_reason": {"type": "stop"}}},
        ]
    )
    monkeypatch.setattr(partial_genrm, "_post", post)
    sample = Sample()
    assert await sglang_rollout.generate_and_rm(args, sample, {}) is sample
    assert sample.status == Sample.Status.ABORTED
    assert sample.reward is None
    assert await sglang_rollout.generate_and_rm(args, sample, {}) is sample
    assert sample.status == Sample.Status.COMPLETED
    assert sample.reward == 6
    assert sample.tokens == [7, 8] and sample.rollout_log_probs == [-0.1, -0.2]
    assert sample.loss_mask == [0, 0]
    assert not state.partial_genrm.restore(sample)
    assert dispatch.await_count == 1
    assert reward_calls == [True, True]

    # Eval uses the business fallback even if the training abort flag is set.
    state.abort_event.set()
    evaluated = await sglang_rollout.generate_and_rm(args, Sample(), {}, evaluation=True)
    assert evaluated.reward == 2
    assert reward_calls == [True, True, False]
    assert post.await_count == 2
    await state.partial_genrm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,close_error",
    [(RuntimeError("scoring failed"), RuntimeError("close failed")), (asyncio.CancelledError(), None)],
)
async def test_partial_genrm_closes_transport_without_hiding_rollout_failure(monkeypatch, error, close_error):
    helper = PartialGenRM(Namespace(), None)
    client = AsyncMock()
    client.aclose.side_effect = close_error
    helper._client = client
    state = SimpleNamespace(partial_genrm=helper)
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "_generate_rollout_async", AsyncMock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        await sglang_rollout.generate_rollout_async(Namespace(), 0, None, None)
    assert caught.value is error
    client.aclose.assert_awaited_once()
    assert helper._client is None


@pytest.mark.asyncio
@pytest.mark.parametrize("distributed", [False, True])
async def test_partial_genrm_abort_does_not_wait_for_generation_transport(monkeypatch, distributed):
    entered = asyncio.Event()
    requests = set()
    control_requests = []

    async def generate(request):
        released = asyncio.Event()
        requests.add(released)
        entered.set()
        try:
            await released.wait()
            return web.json_response(
                {"text": "pre", "output_ids": [1], "meta_info": {"finish_reason": {"type": "abort"}}}
            )
        finally:
            requests.discard(released)

    async def abort_request(request):
        control_requests.append(request.path)
        if request.path == "/abort_request":
            for released in requests:
                released.set()
        return web.json_response({})

    async def workers(request):
        control_requests.append(request.path)
        return web.json_response({"workers": [{"url": f"{base}/policy"}], "urls": [f"{base}/policy"]})

    app = web.Application()
    app.router.add_post("/generate", generate)
    app.router.add_post("/abort_request", abort_request)
    app.router.add_post("/policy/abort_request", abort_request)
    app.router.add_get("/workers", workers)
    app.router.add_get("/list_workers", workers)
    runner = web.AppRunner(app, shutdown_timeout=1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    args = Namespace(
        genrm_model_path="judge",
        genrm_sampling_config={},
        reward_max_concurrency=1,
        partial_rollout=True,
        use_slime_router=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=port,
    )
    state = SimpleNamespace(
        abort_event=asyncio.Event(), aborted=False, evaluating=0, protected_pendings=set(), pendings=set()
    )
    helper = PartialGenRM(args, state)
    helper.engine_urls["__default__"] = [base]
    helper.tokenizers["__default__"] = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: [10], eos_token_id=99
    )
    state.partial_genrm = helper
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    limits = httpx.Limits(max_connections=1)
    async with (
        httpx.AsyncClient(limits=limits, timeout=None) as policy_client,
        httpx.AsyncClient(limits=limits, timeout=None) as actor_client,
    ):
        # Model the external Ray boundary with one actor execution slot, while
        # exercising the real public dispatcher and real HTTP connection pools.
        actor_slot = asyncio.Semaphore(1)

        async def remote_post(*args):
            async with actor_slot:
                return await http_utils._post_with_error_envelope(actor_client, *args)

        actor = SimpleNamespace(do_post=SimpleNamespace(remote=lambda *args: asyncio.create_task(remote_post(*args))))
        monkeypatch.setattr(http_utils, "_http_client", policy_client)
        monkeypatch.setattr(http_utils, "_distributed_post_enabled", distributed)
        monkeypatch.setattr(http_utils, "_post_actors", [actor] if distributed else [])
        monkeypatch.setattr(http_utils, "_post_actor_idx", 0)
        samples = [Sample(status=Sample.Status.COMPLETED) for _ in range(2)]

        async def score(sample):
            try:
                await helper.bind(sample)([{"role": "user", "content": "score"}])
            except partial_genrm.PartialGenRMAborted:
                sample.status = Sample.Status.ABORTED
            return [sample]

        tasks = [asyncio.create_task(score(sample)) for sample in samples]
        state.pendings.update(tasks)
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            assert helper.active[base] == 2
            carried, protected = await asyncio.wait_for(sglang_rollout.abort(args, 1), timeout=3)
            assert len(carried) == 2 and not protected
            assert all(sample.status == Sample.Status.ABORTED and sample.abort_count == 1 for sample in samples)
            assert "/abort_request" in control_requests and "/policy/abort_request" in control_requests
        finally:
            for released in requests:
                released.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if getattr(helper, "_client", None) is not None:
                await helper.aclose()
            await runner.cleanup()
