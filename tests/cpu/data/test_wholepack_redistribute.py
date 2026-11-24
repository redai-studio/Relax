# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Preserve first-improvement domain scheduling and its fallback decisions."""

import copy
import json
from pathlib import Path

import pytest

from relax.utils.data.wholepack import redistribute


REFERENCE = json.loads(Path(__file__).with_name("wholepack_redistribute_reference.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_domain_scoring_preserves_reference_assignment(case):
    """Reference plans include secondary-score accepts and long-wave fallback.

    Expected outputs come from the previous full-layout evaluator. They protect
    the scan order and immediate updates after accepted swaps.
    """
    bins = []
    identity = 0
    pad = case["kwargs"]["tp_size"] * 1024
    for lengths in case["lengths"]:
        group = []
        for length in lengths:
            padding = (-length) % pad
            group.append(
                dict(
                    ids=[identity],
                    t=length,
                    T=length + padding,
                    Q=length * length + padding * padding,
                    dummy=False,
                    oversize=length > case["kwargs"]["capacity"],
                )
            )
            identity += 1
        bins.append(group)
    before = copy.deepcopy(bins)
    actual, evidence = redistribute(bins, **case["kwargs"])
    assert bins == before
    assert [[[pack["ids"][0], pack["dummy"]] for pack in group] for group in actual] == case["expected_assignment"]
    assert evidence == case["expected_evidence"]
