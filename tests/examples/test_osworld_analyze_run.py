# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression coverage for logical Group/Session accounting in reward dumps."""

import pytest

from examples.osworld_agentic.scripts.analyze_run import summarize_run


def _row(step, session, **updates):
    row = {
        "run_tag": "run-a",
        "step": step,
        "group_index": 0,
        "session": session,
        "group_kept": True,
        "group_std": 0.5,
        "outcome": 1,
        "turns": 2,
        "records": 2,
        "shaped": 0.99,
        "stop_reason": "agent_done",
        "ts": "2026-10-11T12:00:00+00:00",
    }
    row.update(updates)
    return row


def test_group_counter_defines_rollout_and_session_weighted_metrics():
    rows = [
        _row(1, "a", outcome=0, turns=30, records=31, group_kept=False, group_std=0.2),
        _row(1, "b", outcome=0, group_kept=False, group_std=0.2),
        _row(2, "c", group_std=0),
        _row(2, "d", group_std=0),
        _row(3, "e", stop_reason="max_turns", ts="2026-10-11T12:05:00+00:00"),
    ]
    first, second = summarize_run(rows, groups_per_rollout=2, samples_per_group=2)
    assert first["rollout"] == 0 and first["complete"]
    assert first["success_rate"] == 0.5
    assert first["filtered_groups"] == 1  # group std is deliberately nonzero
    assert first["records"] == 37 and first["turns"] == 36
    assert first["extra_calls"] == 1
    assert first["budget_hits_approximate"] == 1
    assert second["rollout"] == 1 and not second["complete"]
    assert second["max_turns_explicit"] == 1
    assert second["completion_interval_min"] == 5


def test_run_selection_rejects_silent_mixing():
    rows = [_row(1, "a"), _row(1, "b", run_tag="run-b")]
    with pytest.raises(ValueError, match="multiple run tags"):
        summarize_run(rows)
    result = summarize_run(rows, run_tag="run-b", groups_per_rollout=1, samples_per_group=1)
    assert len(result) == 1 and result[0]["sessions"] == 1
    with pytest.raises(ValueError, match="no reward records"):
        summarize_run(rows, run_tag="absent")


def test_partial_group_and_missing_group_are_incomplete():
    result = summarize_run([_row(1, "a"), _row(3, "b")], groups_per_rollout=3, samples_per_group=1)
    assert not result[0]["complete"]
    result = summarize_run([_row(1, "a")], groups_per_rollout=1, samples_per_group=2)
    assert not result[0]["complete"]


def test_duplicate_session_rejected():
    with pytest.raises(ValueError, match="duplicate Session"):
        summarize_run([_row(1, "a"), _row(1, "a")])


def test_legacy_budget_threshold_is_configurable():
    row = _row(1, "a", turns=40, records=40)
    assert summarize_run([row], max_turns=50)[0]["budget_hits_approximate"] == 0
    assert summarize_run([row], max_turns=30)[0]["budget_hits_approximate"] == 1
