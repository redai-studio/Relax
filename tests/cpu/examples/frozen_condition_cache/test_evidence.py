import copy

import pytest
import torch
from experiments.evidence import compare_shards, compare_trajectories, payload_digest


def test_exact_payload_includes_bits_dtype_shape_and_nested_structure():
    assert payload_digest(torch.tensor(0.0)) != payload_digest(torch.tensor(-0.0))
    assert payload_digest(torch.ones(2)) != payload_digest(torch.ones(2, dtype=torch.bfloat16))
    x = {"values": (torch.randn(3, 2), [None, "key"])}
    assert payload_digest(x) == payload_digest(copy.deepcopy(x))


def test_global_tensor_norm_uses_all_shards_not_per_shard_relative_error():
    row = compare_shards([torch.zeros(1), torch.tensor([100.0])], [torch.tensor([0.5]), torch.tensor([100.0])])
    assert row["passed"] and row["relative_l2"] == 0.005


@pytest.mark.parametrize("actual", [1e-30, float("nan"), float("inf")])
def test_zero_reference_is_strict_and_nonfinite_never_passes(actual):
    assert not compare_shards([torch.zeros(1)], [torch.tensor([actual])])["passed"]


def test_small_update_is_compared_as_delta_not_large_poststep_parameter():
    assert not compare_shards([torch.tensor([1e-5])], [torch.tensor([1.02e-5])])["passed"]
    assert compare_shards([torch.zeros(0)], [torch.zeros(0)])["passed"]
    assert not compare_shards([None], [torch.zeros(1)])["passed"]


def trajectory():
    return {
        "rank": 0,
        "world": 1,
        "loss": 2.0,
        "gradients": {"x": torch.tensor([1.0])},
        "delta": {"x": torch.tensor([0.001])},
        "initial_parameters_sha256": "p",
        "optimizer_initial_sha256": "o",
        "rng_before_sha256": "r",
        "rng_after_sha256": "s",
        "sample_ids": list(range(8)),
        "parameter_global_shapes": {"x": [1]},
    }


def test_full_trajectory_rejects_rng_shift_or_missing_global_shard():
    reference = trajectory()
    actual = copy.deepcopy(reference)
    assert compare_trajectories([reference], [actual])["passed"]
    actual["rng_after_sha256"] = "changed"
    assert not compare_trajectories([reference], [actual])["passed"]
    actual = copy.deepcopy(reference)
    actual["parameter_global_shapes"]["x"] = [2]
    assert not compare_trajectories([reference], [actual])["passed"]
