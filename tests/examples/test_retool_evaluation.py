# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Aggregation-only regressions: no model, executor, scoring or history
replay."""

import json
from types import SimpleNamespace

import pytest

from examples.retool.evaluation import summarize


def test_native_k8_metrics_use_complete_groups(tmp_path, monkeypatch):
    from relax.utils import tracking_utils

    monkeypatch.setattr(tracking_utils, "log", lambda *args, **kwargs: None)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(cases={"gsm8k": 1}, smoke=True)))
    monkeypatch.setenv("RETOOL_EVAL_MANIFEST", str(manifest))
    output = tmp_path / "results"
    output.mkdir()
    samples = []
    for index in range(8):
        sample = SimpleNamespace(
            index=index,
            status=SimpleNamespace(value="completed"),
            reward={"acc": index < 2},
            metadata={"tool_calls": int(index == 3), "tool_feedback_counts": {"ok": int(index == 3)}},
        )
        sample.to_dict = lambda index=index: {"index": index}
        samples.append(sample)
    args = SimpleNamespace(save=str(output), hf_checkpoint="model")
    assert summarize(0, args, {"gsm8k": {"samples": samples, "rewards": [1, 1, 0, 0, 0, 0, 0, 0]}})
    result = json.loads((output / "evaluation_complete.json").read_text())["panels"]["gsm8k"]
    assert result["pass8"] == 1 and result["mean_pass1"] == 0.25
    assert result["tool_call_rate"] == 0.125 and result["tool_submissions"] == 1
    assert len(list((output / "gsm8k").glob("case*.json"))) == 8


def test_incomplete_k8_is_not_accepted(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(cases={"gsm8k": 1}, smoke=True)))
    monkeypatch.setenv("RETOOL_EVAL_MANIFEST", str(manifest))
    with pytest.raises(AssertionError):
        summarize(0, SimpleNamespace(save=str(tmp_path)), {"gsm8k": {"samples": [], "rewards": []}})
    assert not (tmp_path / "evaluation_complete.json").exists()
