"""CPU-only acceptance policy tests; no model forward or GPU allocation."""

import copy
import json
import sys

import torch
from experiments.evidence import assess_strict_acceptance, compare_shards


def comparison(*, loss=1.0, grad=1.0, delta=0.01):
    def quantity(reference, actual):
        return compare_shards(
            [torch.tensor([reference], dtype=torch.float32)], [torch.tensor([actual], dtype=torch.float32)]
        )

    return {
        "passed": True,
        "loss": quantity(1.0, loss),
        "gradients": {"weight": quantity(1.0, grad)},
        "delta": {"weight": quantity(0.01, delta)},
    }


def test_exact_baseline_and_cached_components_are_strictly_accepted():
    result = assess_strict_acceptance(comparison(), comparison())
    assert result["passed"] and result["strict_acceptance"]
    assert result["acceptance_status"] == "accepted"
    assert not result["needs_nondeterminism_review"]


def test_exact_baseline_does_not_allow_one_percent_cached_drift():
    result = assess_strict_acceptance(comparison(), comparison(grad=1.001))
    assert result["hard_numerical_limits_passed"]
    assert not result["passed"] and not result["strict_acceptance"]
    assert result["deterministic_mismatches"] == ["gradients:weight"]
    assert result["acceptance_status"] == "failed"


def test_baseline_noise_is_reported_without_inventing_absolute_tolerance():
    baseline = comparison(grad=1.001)
    original = copy.deepcopy(baseline)
    result = assess_strict_acceptance(baseline, comparison(grad=1.002))
    assert result["passed"] and not result["strict_acceptance"]
    assert result["needs_nondeterminism_review"]
    assert result["acceptance_status"] == "needs_nondeterminism_review"
    assert result["baseline_nondeterministic_components"] == ["gradients:weight"]
    assert result["baseline_nondeterminism_errors"]["gradients:weight"]["maximum_absolute_error"] > 1e-6
    assert baseline == original


def test_nondeterministic_gradient_does_not_relax_exact_loss_or_delta():
    result = assess_strict_acceptance(comparison(grad=1.001), comparison(grad=1.002, delta=0.01001))
    assert not result["passed"]
    assert result["deterministic_mismatches"] == ["delta:weight"]


def test_existing_hard_failure_cannot_become_review_or_pass():
    baseline = comparison(grad=1.001)
    baseline["passed"] = False
    result = assess_strict_acceptance(baseline, comparison(grad=1.002))
    assert not result["passed"] and not result["strict_acceptance"]
    assert result["acceptance_status"] == "failed"


def test_missing_component_cannot_be_strictly_accepted():
    candidate = comparison()
    candidate["delta"] = {}
    result = assess_strict_acceptance(comparison(), candidate)
    assert not result["component_sets_match"] and not result["passed"]


def run_compare(tmp_path, monkeypatch, *, baseline_grad, cached_grad):
    from experiments.compare import main

    common = {
        "rank": 0,
        "world": 1,
        "loss": 1.0,
        "delta": {"weight": torch.tensor([0.01])},
        "initial_parameters_sha256": "fixed-parameters",
        "optimizer_initial_sha256": "fixed-optimizer",
        "rng_before_sha256": "fixed-rng-before",
        "rng_after_sha256": "fixed-rng-after",
        "sample_ids": list(range(8)),
        "parameter_global_shapes": {"weight": [1]},
        "cache_audit": {"failed": 0, "checked_hits": 1},
    }
    argv = ["compare"]
    for name, gradient in (("a1", 1.0), ("a2", baseline_grad), ("bcold", cached_grad), ("bwarm", cached_grad)):
        directory = tmp_path / name
        directory.mkdir()
        row = {**common, "gradients": {"weight": torch.tensor([gradient])}}
        torch.save(row, directory / "trajectory-rank0000.pt")
        argv.extend(("--" + name, str(directory)))
    output = tmp_path / "compare.json"
    argv.extend(("--output", str(output)))
    monkeypatch.setattr(sys, "argv", argv)
    code = main()
    return code, json.loads(output.read_text())


def test_cli_collects_performance_without_claiming_acceptance_for_baseline_noise(tmp_path, monkeypatch):
    code, result = run_compare(tmp_path, monkeypatch, baseline_grad=1.001, cached_grad=1.002)
    assert code == 0 and result["passed"]
    assert not result["strict_acceptance"]
    assert result["acceptance_status"] == "needs_nondeterminism_review"
    assert result["baseline_nondeterministic_components"] == ["gradients:weight"]
    assert result["comparisons"]["a2"]["gradients"]["weight"]["maximum_absolute_error"] > 0


def test_cli_rejects_drift_when_online_repeat_was_exact(tmp_path, monkeypatch):
    code, result = run_compare(tmp_path, monkeypatch, baseline_grad=1.0, cached_grad=1.001)
    assert code == 2 and not result["passed"]
    assert not result["strict_acceptance"]
    assert result["acceptance_status"] == "failed"
    assert result["deterministic_mismatches"]["bcold"] == ["gradients:weight"]
