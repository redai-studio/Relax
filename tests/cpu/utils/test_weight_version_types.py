# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for WeightVersionSpan and Sample.weight_version_spans."""

from __future__ import annotations

import dataclasses

from relax.utils.types import Sample, WeightVersionSpan


def test_weight_version_span_is_frozen_with_expected_fields():
    span = WeightVersionSpan(version="3", token_start=0, token_end=5, attempt_seq=0)
    assert span.version == "3"
    assert span.token_start == 0
    assert span.token_end == 5
    assert span.attempt_seq == 0
    try:
        span.version = "4"  # type: ignore[misc]
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised, "WeightVersionSpan must be frozen"


def test_weight_version_span_allows_unknown_version():
    # None means the engine did not report a version: explicitly unknown, never 0 or newest.
    span = WeightVersionSpan(version=None, token_start=5, token_end=9, attempt_seq=1)
    assert span.version is None


def test_sample_has_default_empty_weight_version_spans():
    # Every Sample field has a default, so it constructs with no arguments.
    sample = Sample()
    assert sample.weight_version_spans == []
    # The legacy field is kept alongside.
    assert sample.weight_versions == []


def test_sample_carries_weight_version_spans():
    spans = [
        WeightVersionSpan(version="1", token_start=0, token_end=3, attempt_seq=0),
        WeightVersionSpan(version=None, token_start=3, token_end=4, attempt_seq=1),
    ]
    sample = Sample(weight_version_spans=spans)
    assert sample.weight_version_spans == spans
    assert sample.weight_versions == []
