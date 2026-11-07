# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from examples.weight_version_audit.demo import run_demo


@pytest.mark.parametrize(
    "scenario,span,age,missing",
    [
        ("complete", 2, 2.2, 0),
        ("missing", 0, 3, 0.4),
        ("lagging", 0, 3, 0),
    ],
)
async def test_real_session_two_update_reproduction(scenario, span, age, missing):
    result = await run_demo(scenario)
    sample = result["sample"]
    assert sample["observed_generation_span"] == span
    assert sample["token_weighted_age"] == age
    assert sample["missing_token_rate"] == missing
    assert sample["post_generation_published_advance"] == 1
    events = result["events"]
    begin = next(i for i, event in enumerate(events) if event["event"] == "tool_wait_begin")
    end = next(i for i, event in enumerate(events) if event["event"] == "tool_wait_end")
    assert [event["published_seq"] for event in events[begin + 1 : end]] == [11, 12]
