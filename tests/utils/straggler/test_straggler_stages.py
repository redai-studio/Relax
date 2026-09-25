# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the coarse stage-group taxonomy.

The mapping is pinned against the timer names that actually exist in the
Megatron version Relax uses, and the attention/MoE capability is pinned as
schema-only so it can never be reported as supported by accident.
"""

import pytest

from relax.utils.straggler.stages import (
    GROUP_BACKWARD,
    GROUP_COMMUNICATION,
    GROUP_DATA,
    GROUP_EVAL,
    GROUP_FORWARD,
    GROUP_OPTIMIZER,
    GROUP_OTHER,
    SCHEMA_ONLY_GROUPS,
    STAGE_GROUPS,
    coverage,
    group_names,
    group_of,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("forward-backward", GROUP_FORWARD),
        ("forward-compute", GROUP_FORWARD),
        ("backward-compute", GROUP_BACKWARD),
        ("optimizer", GROUP_OPTIMIZER),
        ("optimizer-inner-step", GROUP_OPTIMIZER),
        ("batch-generator", GROUP_DATA),
        ("params-all-gather", GROUP_COMMUNICATION),
        ("all-grads-sync", GROUP_COMMUNICATION),
        ("embedding-grads-all-reduce", GROUP_COMMUNICATION),
        ("forward-send-forward-recv", GROUP_COMMUNICATION),
        ("eval-time", GROUP_EVAL),
        ("load-checkpoint", "setup"),
    ],
)
def test_known_timer_names_map_to_their_group(name: str, expected: str) -> None:
    assert group_of(name) == expected


@pytest.mark.parametrize(
    "name",
    ["mystery-stage", "", "  ", "forward-ish", "attention-core", "moe-dispatch"],
)
def test_unknown_names_are_not_guessed_into_a_group(name: str) -> None:
    assert group_of(name) == GROUP_OTHER


def test_communication_suffix_rule_covers_unlisted_names() -> None:
    assert group_of("expert-parallel-all-reduce") == GROUP_COMMUNICATION
    assert group_of("pipeline-forward-recv") == GROUP_COMMUNICATION
    assert group_of("optimizer-something-new") == GROUP_OPTIMIZER


def test_attention_and_moe_are_schema_only() -> None:
    """Megatron core 0.19 exposes no attention/MoE timer, so C2 claims
    neither."""
    assert SCHEMA_ONLY_GROUPS == ("attention", "moe")
    assert "attention" not in STAGE_GROUPS.values()
    assert "moe" not in STAGE_GROUPS.values()


def test_group_names_buckets_preserve_order() -> None:
    buckets = group_names(["forward-compute", "batch-generator", "backward-compute"])

    assert buckets[GROUP_FORWARD] == ["forward-compute"]
    assert buckets[GROUP_BACKWARD] == ["backward-compute"]
    assert buckets[GROUP_DATA] == ["batch-generator"]


def test_coverage_reports_missing_and_unclassified_groups() -> None:
    report = coverage(["forward-backward", "optimizer", "unknown-thing"])

    assert report["measured_groups"] == [GROUP_FORWARD, GROUP_OPTIMIZER]
    assert GROUP_COMMUNICATION in report["missing_groups"]
    assert report["unclassified"] == ["unknown-thing"]
    assert report["schema_only_groups"] == ["attention", "moe"]
    assert "attention" not in report["groups"]
