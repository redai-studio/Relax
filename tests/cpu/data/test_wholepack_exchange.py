# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep optimized candidate scoring equivalent to the prior full evaluator."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from relax.utils.data.wholepack_exchange import refine


REFERENCE = json.loads(Path(__file__).with_name("wholepack_exchange_reference.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["name"])
def test_exchange_preserves_full_reference_plan(case):
    """Golden outputs were produced by the former full D*K tensor evaluator.

    Cover same-domain swaps, dummy sources, oversize wave movement, the K16
    boundary, completely locked packs, tied maxima and the 1e-12 score
    threshold.
    """
    bins = []
    for rank, lengths in enumerate(case["lengths"]):
        group = []
        for wave, length in enumerate(lengths):
            padding = (-length) % 4096
            group.append(
                dict(
                    ids=[rank * len(lengths) + wave],
                    t=length,
                    T=length + padding,
                    Q=case["qualities"][rank][wave] + padding * padding,
                    dummy=False,
                    oversize=length > 262144,
                )
            )
        bins.append(group)
    for rank, source, target in case["dummy_sources"]:
        bins[rank][target] = {**bins[rank][source], "dummy": True}
    before = copy.deepcopy(bins)
    actual, evidence = refine(bins, 262144)
    evidence.pop("seconds")
    assert bins == before, "Optimization must not change a pack's contents"
    assert [[[pack["ids"][0], pack["dummy"]] for pack in group] for group in actual] == case["expected_assignment"]
    assert evidence == case["expected_evidence"]


def test_pruned_search_preserves_order_across_chunks_and_phases():
    """Fixed reference comes from the previous unpruned local-domain
    evaluator."""
    bins = []
    for rank in range(64):
        group = []
        for wave in range(16):
            length = 1000 + (rank * 7919 + wave * 104729 + (rank * wave + 17) * 1543) % 240000
            padding = (-length) % 4096
            group.append(
                dict(
                    ids=[rank * 16 + wave],
                    t=length,
                    T=length + padding,
                    Q=length * length + padding * padding,
                    dummy=False,
                    oversize=False,
                )
            )
        bins.append(group)
    before = copy.deepcopy(bins)
    actual, evidence = refine(bins, 262144)
    evidence.pop("seconds")
    expected = REFERENCE["chunked_reference"]
    assert bins == before
    assert hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() == expected["plan_sha256"]
    assert hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest() == expected["evidence_sha256"]
    assert evidence["accepted"] == 3
    assert evidence["trials"] == 1548288
