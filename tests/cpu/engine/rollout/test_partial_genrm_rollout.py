# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise Judge carry through the real rollout and async reward executor."""

import asyncio
import sys
from argparse import Namespace
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from relax.engine.rewards import RewardExecutor
from relax.engine.rollout import partial_genrm
from relax.engine.rollout.partial_genrm import PartialGenRM
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
        apply_chat_template=lambda *args, **kwargs: [10], eos_token_id=99
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
    monkeypatch.setattr(partial_genrm, "post", post)
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
