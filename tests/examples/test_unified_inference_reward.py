# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest


pytest.importorskip("ray")

from relax.engine.rewards import RewardExecutionError, RewardExecutor, batched_async_rm  # noqa: E402
from relax.utils.types import Sample  # noqa: E402
from scripts.training.hpc import unified_inference_reward as reward  # noqa: E402


async def test_deferred_training_reward_batch_preserves_order_and_limits_requests(monkeypatch):
    active = 0
    peak = 0
    requests = []

    async def backend(request):
        nonlocal active, peak
        payload = json.loads(request.content)
        index = int(payload["text"].rsplit(" ", 1)[1])
        requests.append(index)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01 * (4 - index))
        active -= 1
        return httpx.Response(200, json={"meta_info": {"output_token_logprobs": [[-index, 1, "a"], [-2, 2, "b"]]}})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        reward.httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(backend))
    )
    monkeypatch.setattr(reward, "get_serve_url", lambda role: f"http://{role}")
    monkeypatch.setattr(RewardExecutor, "_instance", None)
    args = SimpleNamespace(
        custom_rm_path="scripts.training.hpc.unified_inference_reward.score", reward_max_concurrency=2
    )
    samples = [Sample(index=i, response=str(i), response_length=1, teacher_log_probs=[-1.0]) for i in range(4)]
    result = await batched_async_rm(args, samples)
    assert result == [{"score": -(i + 2) / 2} for i in range(4)]
    assert sorted(requests) == list(range(4))
    assert peak == 2

    samples[0].teacher_log_probs = None
    with pytest.raises(RewardExecutionError, match="completed Teacher writeback"):
        await batched_async_rm(args, [samples[0]])
    assert len(requests) == 4
