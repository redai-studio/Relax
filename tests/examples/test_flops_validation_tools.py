# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Failure-sensitive checks for offline reports, using artificial test logs."""

import json

import pytest
import torch

from examples.flops_balancing.compare_step_captures import compare_captures
from examples.flops_balancing.summarize_runs import read_window, summarize
from relax.utils.training.optimizer_step_capture import OptimizerStepCapture, capture_rollout_inputs
from tests.utils.test_optimizer_step_capture import _OptimizerContract


def _capture(root):
    capture_rollout_inputs(
        {
            "tokens": [torch.tensor([1, 2])],
            "loss_masks": [torch.tensor([0, 1])],
            "total_lengths": [2],
            "response_lengths": [2],
        },
        root,
        rank=0,
        topology=dict(dp=0, tp=0, pp=0, cp=0),
        normalization="token",
        global_batch_size=1,
        device=torch.device("cpu"),
        group=None,
    )
    optimizer = _OptimizerContract()
    optimizer.parameter.grad = torch.tensor([0.5, 0.75])
    capture = OptimizerStepCapture(optimizer, root / "rank-00000", group=None, device=torch.device("cpu"))
    with capture:
        optimizer.step()
    capture.finish({"loss": 1.0}, True)


def _compare(left, right):
    return compare_captures(
        left, right, loss_atol=0, loss_rtol=0, grad_atol=0, grad_rtol=0, update_atol=0, update_rtol=0
    )


def test_complete_capture_comparison_rejects_changed_update_and_initial_state(tmp_path):
    left, right = tmp_path / "A", tmp_path / "D"
    _capture(left)
    _capture(right)
    assert _compare(left, right)["passed"]
    path = right / "rank-00000/parameter-000000.pt"
    original = torch.load(path, weights_only=True)
    for field in ("initial", "gradient", "update"):
        changed = {key: value.clone() for key, value in original.items()}
        changed[field][0] += 0.1
        torch.save(changed, path)
        assert not _compare(left, right)["passed"]


def test_capture_comparison_rejects_different_effective_inputs(tmp_path):
    left, right = tmp_path / "A", tmp_path / "D"
    _capture(left)
    _capture(right)
    path = right / "inputs-rank-00000.json"
    changed = json.loads(path.read_text())
    changed["samples"][0] = "different mask digest"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Effective samples"):
        _compare(left, right)


def _log(seconds):
    return "\n".join(
        f"perf {index}: {
            {
                'perf/step_time': time,
                'perf/actor_train_tokens': tokens,
                'perf/step_resp_token_per_s': tokens / 2 / time,
            }
        }"
        for index, (time, tokens) in enumerate(seconds)
    )


def test_throughput_uses_total_tokens_over_total_time():
    result = read_window(_log([(1, 100), (3, 600)]), warmup=0, steps=2)
    assert result["input_tokens_per_s"] == 175  # Mean of step rates would be 150.
    assert result["supervised_tokens_per_s"] == 87.5


def test_throughput_rejects_duplicate_or_missing_steps():
    log = _log([(1, 100)])
    with pytest.raises(ValueError, match="Duplicate"):
        read_window(log + "\n" + log, 0, 1)
    with pytest.raises(ValueError, match="Missing"):
        read_window(log, 0, 2)


def _runs(tmp_path):
    directories = []
    for mode, seconds in (("A", 1.2), ("D", 1.0)):
        for repeat in range(3):
            directory = tmp_path / f"{mode}{repeat}"
            directory.mkdir()
            metadata = dict(
                mode=mode,
                capture=False,
                dry_run=False,
                invariant_command=["same args"],
                versions={},
                source_sha256="source",
                workload_sha256="workload",
                model_revision="model",
                data_revision="data",
                native_source_sha256="native source",
                devices=[{"name": "test fixture"}],
            )
            (directory / "run.json").write_text(json.dumps(metadata))
            (directory / "status.json").write_text(json.dumps({"status": "SUCCEEDED"}))
            (directory / "training.log").write_text(_log([(seconds, 120)]))
            directories.append(directory)
    return directories


def test_repeated_run_summary_checks_20_percent_boundary(tmp_path):
    report = summarize(_runs(tmp_path), warmup=0, steps=1)
    assert report["modes"]["D"]["reaches_20_percent"]


@pytest.mark.parametrize("change", ["different_config", "different_tokens", "capture", "failure", "same_run"])
def test_repeated_run_summary_rejects_unfair_or_incomplete_comparisons(tmp_path, change):
    directories = _runs(tmp_path)
    directory = directories[-1]
    path = directory / "run.json"
    metadata = json.loads(path.read_text())
    if change == "different_config":
        metadata["invariant_command"] = ["changed token budget"]
    elif change == "capture":
        metadata["capture"] = True
    elif change == "failure":
        (directory / "status.json").write_text(json.dumps({"status": "FAILED"}))
    elif change == "same_run":
        directories[-1] = directories[-2]
    else:
        (directory / "training.log").write_text(_log([(1, 100)]))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        summarize(directories, warmup=0, steps=1)
