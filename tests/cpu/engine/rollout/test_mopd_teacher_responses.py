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


@pytest.mark.asyncio
async def test_out_of_order_generation_keeps_bound_teacher(monkeypatch):
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
        aborted=False, opd_manager=manager, dp_rank_context=nullcontext, semaphore=asyncio.Semaphore(2)
    )
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _: state)
    a = Sample(group_index=0, index=0, metadata={"data_source": "A"})
    b = Sample(group_index=1, index=1, metadata={"data_source": "B"})
    started_a = asyncio.get_running_loop().create_future()
    finish_a = asyncio.Event()
    calls = []

    async def generate(args, sample, sampling_params, evaluation):
        source = sample.metadata["data_source"]
        assert sample.metadata.get("teacher_id") == source, f"Expected teacher_id={source!r} before Student generation"
        if sample is a:
            started_a.set_result(None)
            await finish_a.wait()
        sample.status = Sample.Status.COMPLETED
        sample.tokens = [10, 20]
        sample.response_length = 1
        return sample

    async def reward(args, sample):
        return 1.0

    async def post(session, url, payload, sample, tag, expected_ids):
        calls.append((sample.index, url))
        score = {"http://a/generate": -0.25, "http://b/generate": -0.75}[url]
        return LogprobResponse({"meta_info": {"input_token_logprobs": [[None, 10], [score, 20]]}})

    monkeypatch.setattr(sglang_rollout, "generate", generate)
    monkeypatch.setattr(sglang_rollout, "async_rm", reward)
    monkeypatch.setattr(manager, "_post_logprob", post)
    task_a = asyncio.create_task(sglang_rollout.generate_and_rm_group(args, [a], {}))
    try:
        done, _ = await asyncio.wait((started_a, task_a), timeout=5, return_when=asyncio.FIRST_COMPLETED)
        if task_a in done:
            task_a.result()
        assert started_a in done, "Sample A did not enter Student generation"
        await asyncio.wait_for(sglang_rollout.generate_and_rm_group(args, [b], {}), timeout=5)
        assert calls == [(1, "http://b/generate")]
        finish_a.set()
        await asyncio.wait_for(task_a, timeout=5)
    finally:
        task_a.cancel()
        await asyncio.gather(task_a, return_exceptions=True)

    assert calls == [(1, "http://b/generate"), (0, "http://a/generate")]
    assert [a.teacher_log_probs, b.teacher_log_probs] == [[-0.25], [-0.75]]
