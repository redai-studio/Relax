# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Generative reward manager: local scoring, finally-offload, CPU-no-CUDA,
remote guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from relax.distributed.ray.generative_reward import (
    GenerativeRewardManager,
    LocalGenerativeRewardManager,
)


# A module-level fake scorer reachable by dotpath for the manager's loader.
_EVENTS = []


class FakeScorer:
    required_tracks = ("image",)

    def __init__(self, args):
        self.args = args

    def onload(self):
        _EVENTS.append("onload")

    def offload(self):
        _EVENTS.append("offload")

    def score_batch(self, requests):
        _EVENTS.append("score")
        return {"pickscore": [1.0 for _ in requests]}


class FailingScorer(FakeScorer):
    def score_batch(self, requests):
        _EVENTS.append("score")
        raise RuntimeError("boom")


_SCORER_PATH = "tests.distributed.ray.test_generative_reward.FakeScorer"
_FAILING_PATH = "tests.distributed.ray.test_generative_reward.FailingScorer"


def _args(**kw):
    base = dict(reward_scorer_path=_SCORER_PATH, reward_runtime="cpu", reward_endpoint=None)
    base.update(kw)
    return SimpleNamespace(**base)


def setup_function(_):
    _EVENTS.clear()


def test_manager_local_returns_local_manager():
    mgr = GenerativeRewardManager.local(_args())
    assert isinstance(mgr, LocalGenerativeRewardManager)


def test_local_score_runs_onload_score_offload():
    mgr = LocalGenerativeRewardManager(_args())
    out = mgr.score([{}, {}])
    assert out == {"pickscore": [1.0, 1.0]}
    assert _EVENTS == ["onload", "score", "offload"]


def test_local_score_offloads_even_on_failure():
    mgr = LocalGenerativeRewardManager(_args(reward_scorer_path=_FAILING_PATH))
    with pytest.raises(RuntimeError):
        mgr.score([{}])
    assert _EVENTS[-1] == "offload"  # finally-offload guarantee


def test_local_requires_scorer_path():
    mgr = LocalGenerativeRewardManager(_args(reward_scorer_path=None))
    with pytest.raises(ValueError):
        mgr.score([{}])


def test_cpu_scorer_does_not_initialize_cuda():
    # design §14.1(6): a CPU-runtime scorer must not touch CUDA.
    from relax.engine.rewards.pickscore import PickScoreScorer

    s = PickScoreScorer(SimpleNamespace(reward_runtime="cpu", reward_model_path=None))
    assert s.device.type == "cpu"


def test_remote_runtime_requires_endpoint():
    mgr = GenerativeRewardManager(_args(reward_runtime="remote", reward_endpoint=None))
    with pytest.raises(ValueError):
        mgr.score([{}])
