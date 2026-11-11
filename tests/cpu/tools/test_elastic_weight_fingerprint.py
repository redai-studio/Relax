# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import json
import subprocess
import sys

import pytest

from tools.elastic_validate import probe_weight_fingerprint as probe


@pytest.mark.parametrize(
    ("a", "b"),
    [([], []), ([[1.0]], []), ([[]], [[]]), ([[float("nan")]], [[1.0]]), ([[float("inf")]], [[float("inf")]])],
)
def test_compare_rejects_incomplete_or_nonfinite_probes(a, b):
    assert probe.compare({"url": "a", "prompts": a}, {"url": "b", "prompts": b}) == 2


@pytest.mark.parametrize(("value", "expected"), [(1.01, 0), (2.0, 1)])
def test_compare_distinguishes_matching_and_different_logprobs(value, expected):
    assert probe.compare({"url": "a", "prompts": [[1.0]]}, {"url": "b", "prompts": [[value]]}) == expected


def test_compare_cli_returns_error_for_engine_count_mismatch(monkeypatch, tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"engines": []}))
    b.write_text(json.dumps({"engines": [{"url": "b", "prompts": [[1.0]]}]}))
    monkeypatch.setattr("sys.argv", ["probe", "--compare", str(a), str(b)])
    assert probe.main() == 2
    result = subprocess.run(
        [sys.executable, probe.__file__, "--compare", str(a), str(b)], capture_output=True, timeout=10
    )
    assert result.returncode == 2
