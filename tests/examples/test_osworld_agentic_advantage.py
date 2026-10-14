# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest

from examples.osworld_agentic.app.reward_post_process import compute_advantages, filter_outcome_group


def test_osworld_advantage_normalizes_sessions_not_export_rows(monkeypatch):
    monkeypatch.setenv("OSWORLD_PER_TRAJ_NORM", "0")
    monkeypatch.delenv("OSWORLD_REWARD_DUMP", raising=False)
    sessions = [
        {"turn_000": {"outcome": 1.0, "agent_turns": 1}},
        {f"turn_{i:03d}": {"outcome": 0.0, "agent_turns": 3} for i in range(3)},
    ]
    assert compute_advantages(sessions) == [
        {"turn_000": 1.0},
        {f"turn_{i:03d}": -1.0 for i in range(3)},
    ]


def test_osworld_advantage_zeroes_all_fail_group(monkeypatch):
    monkeypatch.delenv("OSWORLD_REWARD_DUMP", raising=False)
    sessions = [{"turn_000": {"outcome": 0.0, "agent_turns": turns}} for turns in (1, 30)]
    assert compute_advantages(sessions) == [{"turn_000": 0.0}, {"turn_000": 0.0}]


def test_osworld_advantage_rejects_inconsistent_outcomes():
    with pytest.raises(ValueError, match="disagree"):
        compute_advantages([{"a": {"outcome": 0.0}, "b": {"outcome": 1.0}}])


@pytest.mark.parametrize("outcomes,keep", [([0.0] * 8, False), ([1.0] * 8, False), ([0.0] * 7 + [1.0], True)])
def test_osworld_filter_handles_rewardless_exports(monkeypatch, outcomes, keep):
    monkeypatch.delenv("OSWORLD_ACC_LO", raising=False)
    monkeypatch.delenv("OSWORLD_ACC_HI", raising=False)
    # Many exports from one session must not change the eight-trajectory accuracy.
    samples = [
        SimpleNamespace(session_id=str(i), metadata={"outcome": outcome}, reward=None)
        for i, outcome in enumerate(outcomes)
        for _ in range(100 if i == 0 else 1)
    ]
    result = filter_outcome_group(SimpleNamespace(reward_key="score"), samples)
    assert result.keep is keep
    assert (result.reason is None) is keep


def test_osworld_filter_rejects_inconsistent_outcomes():
    samples = [SimpleNamespace(session_id="a", metadata={"outcome": x}) for x in (0.0, 1.0)]
    with pytest.raises(ValueError, match="disagree"):
        filter_outcome_group(SimpleNamespace(), samples)
