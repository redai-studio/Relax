# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the coarse stage-group taxonomy.

The mapping is pinned against the timer names that actually exist in the
Megatron version Relax uses, and the attention/MoE capability is pinned as
schema-only so it can never be reported as supported by accident.

The table below is the independent audit's full result, not a sample: all 23
names Relax's core path passes to ``config.timers``. A name added to
``OBSERVED_TIMER_NAMES`` without a group, or a group changed away from the
audited value, fails here instead of reaching a verdict mislabelled.
"""

from typing import Any, Dict, List

import pytest

from relax.utils.straggler import reporter
from relax.utils.straggler.stages import (
    GROUP_BACKWARD,
    GROUP_COMMUNICATION,
    GROUP_DATA,
    GROUP_FORWARD,
    GROUP_OPTIMIZER,
    GROUP_OTHER,
    MEASURED_GROUPS,
    OBSERVED_TIMER_NAMES,
    SCHEMA_ONLY_GROUPS,
    STAGE_GROUPS,
    coverage,
    group_names,
    group_of,
    with_stage_group,
)


#: Every timer name Relax's core path really emits, with the coarse stage it
#: must map to. Kept as the single source of truth for the expected mapping.
OBSERVED_TO_GROUP: Dict[str, str] = {
    "forward-backward": GROUP_FORWARD,
    "forward-compute": GROUP_FORWARD,
    "backward-compute": GROUP_BACKWARD,
    "forward-send": GROUP_COMMUNICATION,
    "forward-recv": GROUP_COMMUNICATION,
    "backward-send": GROUP_COMMUNICATION,
    "backward-recv": GROUP_COMMUNICATION,
    "forward-send-forward-recv": GROUP_COMMUNICATION,
    "forward-send-backward-recv": GROUP_COMMUNICATION,
    "backward-send-forward-recv": GROUP_COMMUNICATION,
    "backward-send-backward-recv": GROUP_COMMUNICATION,
    "forward-backward-send-forward-backward-recv": GROUP_COMMUNICATION,
    "all-grads-sync": GROUP_COMMUNICATION,
    "non-tensor-parallel-grads-all-reduce": GROUP_COMMUNICATION,
    "embedding-grads-all-reduce": GROUP_COMMUNICATION,
    "conditional-embedder-grads-all-reduce": GROUP_COMMUNICATION,
    "params-all-gather": GROUP_COMMUNICATION,
    "optimizer-inner-step": GROUP_OPTIMIZER,
    "optimizer-copy-to-main-grad": GROUP_OPTIMIZER,
    "optimizer-unscale-and-check-inf": GROUP_OPTIMIZER,
    "optimizer-copy-main-to-model-params": GROUP_OPTIMIZER,
    "optimizer-clip-main-grad": GROUP_OPTIMIZER,
    "optimizer-count-zeros": GROUP_OPTIMIZER,
}


def test_observed_timer_set_is_exactly_the_audited_23() -> None:
    """The observed set is the audited inventory, with no duplicates."""
    assert len(OBSERVED_TIMER_NAMES) == 23
    assert len(set(OBSERVED_TIMER_NAMES)) == 23
    assert set(OBSERVED_TIMER_NAMES) == set(OBSERVED_TO_GROUP)


def test_every_observed_name_has_an_explicit_group_entry() -> None:
    """A name added to the observed set without a mapping fails here."""
    for name in OBSERVED_TIMER_NAMES:
        assert name in STAGE_GROUPS, f"{name!r} is observed but has no explicit group"
        assert STAGE_GROUPS[name] == OBSERVED_TO_GROUP[name]


@pytest.mark.parametrize(("name", "expected"), sorted(OBSERVED_TO_GROUP.items()))
def test_known_timer_names_map_to_their_group(name: str, expected: str) -> None:
    assert group_of(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "mystery-stage",
        "",
        "  ",
        "forward-ish",
        "attention-core",
        "moe-dispatch",
        # Names that merely *look* like a measured family. The taxonomy is a
        # closed exact-match table: guessing a group from a suffix is not
        # allowed, because a wrong stage label is worse than an unclassified
        # one.
        "expert-parallel-all-reduce",
        "pipeline-forward-recv",
        "optimizer-something-new",
        "some-optimizer-thing",
        "totally-made-up-all-gather",
    ],
)
def test_unknown_names_are_never_guessed_into_a_group(name: str) -> None:
    assert group_of(name) == GROUP_OTHER


def test_group_of_never_raises_on_unexpected_input() -> None:
    assert group_of(None) == GROUP_OTHER  # type: ignore[arg-type]
    assert group_of(123) == GROUP_OTHER  # type: ignore[arg-type]


def test_attention_and_moe_are_schema_only() -> None:
    """Megatron core 0.19 exposes no attention/MoE timer, so C2 claims
    neither."""
    assert SCHEMA_ONLY_GROUPS == ("attention", "moe")
    assert "attention" not in STAGE_GROUPS.values()
    assert "moe" not in STAGE_GROUPS.values()
    assert set(SCHEMA_ONLY_GROUPS).isdisjoint(MEASURED_GROUPS)
    for name in ("attention", "moe", "moE", "attention-core", "moe-dispatch"):
        assert group_of(name) == GROUP_OTHER
        assert group_of(name) not in MEASURED_GROUPS


def test_schema_only_groups_never_appear_as_a_measured_group() -> None:
    """Neither the taxonomy nor a full observed-name audit may report
    attention/MoE as measured."""
    assert not (set(SCHEMA_ONLY_GROUPS) & set(STAGE_GROUPS.values()))
    assert not (set(SCHEMA_ONLY_GROUPS) & set(MEASURED_GROUPS))

    report = coverage(OBSERVED_TIMER_NAMES)

    assert report["schema_only_groups"] == ["attention", "moe"]
    assert report["unclassified"] == []
    assert set(report["measured_groups"]) == {GROUP_FORWARD, GROUP_BACKWARD, GROUP_COMMUNICATION, GROUP_OPTIMIZER}
    for group in SCHEMA_ONLY_GROUPS:
        assert group not in report["groups"]
        assert group not in report["measured_groups"]


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


def test_with_stage_group_puts_the_group_next_to_the_raw_stage() -> None:
    facts: Dict[str, Any] = {"stage": "backward-compute", "rank": 1}

    annotated = with_stage_group(facts)

    assert annotated["stage"] == "backward-compute"
    assert annotated["stage_group"] == GROUP_BACKWARD
    # The verdict's own facts are not mutated.
    assert facts == {"stage": "backward-compute", "rank": 1}


def test_with_stage_group_does_not_label_missing_or_schema_only_stages() -> None:
    assert with_stage_group({"rank": 1}) == {"rank": 1}
    assert with_stage_group({"stage": ""}) == {"stage": ""}
    # An attention/MoE-looking name is unclassified, never promoted to a
    # measured group.
    assert with_stage_group({"stage": "attention-core"})["stage_group"] == GROUP_OTHER
    assert with_stage_group({"stage": "moe-dispatch"})["stage_group"] == GROUP_OTHER


class _StubRuntime:
    """Minimal runtime exposing the two methods the reporter reads."""

    def __init__(self, verdicts: List[Any]) -> None:
        self._verdicts = list(verdicts)

    def summary(self) -> Dict[str, Any]:
        return {}

    def drain_verdicts(self) -> List[Any]:
        return list(self._verdicts)


def test_reporter_logs_the_raw_name_and_coarse_stage_together(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The group reaches a reader beside the raw name, without a wire
    change."""
    verdict = {
        "name": "backward-compute",
        "deviation": 0.5,
        "rank": 1,
        "facts": {"stage": "backward-compute"},
    }

    with caplog.at_level("INFO"):
        metrics = reporter.build_metrics(_StubRuntime([verdict]))

    assert "name=backward-compute coarse_stage=backward" in caplog.text
    # Metrics stay numeric: a string must not leak into a metrics backend.
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in metrics.values())


def test_reporter_labels_an_unclassified_stage_as_other(
    caplog: pytest.LogCaptureFixture,
) -> None:
    verdict = {
        "name": "attention-core",
        "deviation": 0.2,
        "rank": 0,
        "facts": {"stage": "attention-core"},
    }

    with caplog.at_level("INFO"):
        reporter.build_metrics(_StubRuntime([verdict]))

    assert "coarse_stage=other" in caplog.text
    assert "coarse_stage=attention" not in caplog.text


def test_reporter_logs_nothing_for_a_verdict_without_a_stage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A verdict stub without a stage name must not fabricate a label."""
    with caplog.at_level("INFO"):
        reporter.build_metrics(_StubRuntime([{"deviation": 0.1, "rank": 2}]))

    assert "coarse_stage=" not in caplog.text
