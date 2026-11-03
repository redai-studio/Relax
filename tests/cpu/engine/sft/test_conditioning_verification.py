# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""One-update comparison restores the reference branch's training state."""

import copy
import json
import random

import numpy as np
import pytest
import torch

from relax.engine.sft.conditioning_verification import VerificationError, compare_sharded_tensors, verify_update


@pytest.mark.parametrize(
    "reference,actual,passed",
    [
        ([0.0, 0.0], [0.0, 0.0], True),
        ([0.0, 0.0], [1e-20, 0.0], False),
        ([1.0, -1.0], [1.005, -1.005], True),
        ([1.0, -1.0], [1.02, -1.02], False),
        ([float("inf"), 1.0], [float("inf"), 1.0], False),
        ([1.0, 1.0], [float("nan"), 1.0], False),
    ],
)
def test_per_tensor_norm_check_handles_zero_and_nonfinite(reference, actual, passed):
    report = compare_sharded_tensors({"parameter": torch.tensor(reference)}, {"parameter": torch.tensor(actual)})
    assert report["parameter"]["passed"] is passed
    json.dumps(report, allow_nan=False)


def test_norm_checks_each_parameter_instead_of_hiding_small_tensor_errors():
    reference = {"large": torch.ones(20) * 1e6, "small": torch.tensor([1.0])}
    actual = {"large": reference["large"].clone(), "small": torch.tensor([1.1])}
    report = compare_sharded_tensors(reference, actual)
    assert report["large"]["passed"]
    assert not report["small"]["passed"]
    assert report["small"]["max_absolute_error"] > 0.09


def test_exact_comparison_does_not_use_one_percent_allowance():
    report = compare_sharded_tensors(
        {"weight": torch.tensor([1.0])}, {"weight": torch.tensor([1.00001])}, relative_tolerance=0
    )
    assert not report["weight"]["passed"]


def test_verify_update_leaves_only_one_real_adamw_update_and_rng_stream(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Verify a non-empty optimizer state, not only Adam's empty first step.
    model(torch.ones(1, 3)).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    samples = [torch.tensor([[1.0, -2.0, 3.0]]), torch.tensor([[2.0, 1.0, -1.0]])]
    force = [False]
    calls = []

    def objective(current_model, sample):
        target = random.random() + float(np.random.rand()) + torch.rand(())
        return (current_model(sample) - target).square().mean()

    def forward(sample):
        calls.append(force[0])
        return objective(model, sample)

    def seed():
        random.seed(19)
        np.random.seed(19)
        torch.manual_seed(19)

    seed()
    for sample in samples:
        (objective(reference, sample) / len(samples)).backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
    reference_optimizer.step()
    expected_next_random = (random.random(), float(np.random.rand()), torch.rand(()))
    seed()
    report = verify_update(
        model,
        optimizer,
        list(model.named_parameters()),
        samples,
        forward,
        lambda value: force.__setitem__(0, value),
        device=torch.device("cpu"),
        group=None,
        gloo_group=None,
        loss_scale=1 / len(samples),
        output_dir=tmp_path,
        clip_fn=lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5),
        relative_tolerance=0,
    )
    assert report["passed"]
    assert not report["cache_branch_verified"]
    assert not report["cache_validation"]["statistics_provided"]
    assert report["actual_clip_norm"] == float(expected_norm)
    assert calls == [True, True, False, False]
    assert force == [False]
    assert report["update_quantity"] == "FP32 theta_after - theta_before"
    assert json.loads((tmp_path / "conditioning-verification.json").read_text())["passed"]
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.equal(actual, expected)
        assert actual.grad is None
    for state in optimizer.state.values():
        assert state["step"] == 2  # The reference pass did not advance Adam twice.
    assert random.random() == expected_next_random[0]
    assert float(np.random.rand()) == expected_next_random[1]
    assert torch.equal(torch.rand(()), expected_next_random[2])


@pytest.mark.parametrize("cache_hits", [False, True])
def test_cached_pass_must_cover_real_hits_when_statistics_are_supplied(tmp_path, cache_hits):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    initial = copy.deepcopy(model.state_dict())
    force = [False]
    counts = {"effective_hit": 0, "encoding_calls": 0}
    samples = [torch.ones(1, 1), torch.ones(1, 1) * 2]

    def forward(sample):
        if not force[0] and cache_hits:
            counts["effective_hit"] += 1
        else:
            counts["encoding_calls"] += 1
        return model(sample).square().mean()

    def verify():
        return verify_update(
            model,
            optimizer,
            list(model.named_parameters()),
            samples,
            forward,
            lambda value: force.__setitem__(0, value),
            device=torch.device("cpu"),
            group=None,
            gloo_group=None,
            loss_scale=1 / len(samples),
            output_dir=tmp_path,
            relative_tolerance=0,
            cache_statistics=lambda: dict(counts),
        )

    if cache_hits:
        report = verify()
        assert report["passed"] and report["cache_branch_verified"]
        assert report["cache_validation"]["actual"]["global_effective_hits"] == len(samples)
        assert report["cache_validation"]["actual"]["global_encoding_calls"] == 0
        assert report["cache_validation"]["reference"]["global_encoding_calls"] == len(samples)
    else:
        with pytest.raises(VerificationError) as error:
            verify()
        report = error.value.report
        assert report["loss"]["passed"]
        assert all(row["passed"] for row in report["gradients"].values())
        assert all(row["passed"] for row in report["parameter_updates"].values())
        assert not report["passed"] and not report["cache_branch_verified"]
        assert report["cache_validation"]["actual"]["local_encoding_calls"] == len(samples)
        assert report["cache_validation"]["actual"]["local_effective_hits"] == 0
        assert not optimizer.state
        assert all(torch.equal(model.state_dict()[name], value) for name, value in initial.items())
    assert force == [False]
    assert json.loads((tmp_path / "conditioning-verification.json").read_text())["cache_branch_verified"] is cache_hits


def test_failed_comparison_writes_report_and_restores_initial_state(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    initial = model.weight.detach().clone()
    force = [False]
    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    initial_torch_rng = torch.get_rng_state().clone()

    def forward(sample):
        torch.rand(())
        return model(sample).sum() * (1 if force[0] else 2)

    with pytest.raises(VerificationError) as error:
        verify_update(
            model,
            optimizer,
            list(model.named_parameters()),
            [torch.ones(1, 1)],
            forward,
            lambda value: force.__setitem__(0, value),
            device=torch.device("cpu"),
            group=None,
            gloo_group=None,
            loss_scale=1,
            output_dir=tmp_path,
        )
    assert not error.value.report["passed"]
    assert torch.equal(model.weight, initial)
    assert not optimizer.state
    assert torch.equal(torch.get_rng_state(), initial_torch_rng)
    assert force == [False]
    assert not json.loads((tmp_path / "conditioning-verification.json").read_text())["passed"]


def test_forward_exception_restores_weights_optimizer_and_force_flag(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    initial = copy.deepcopy(model.state_dict())
    force = [False]

    def forward(sample):
        if not force[0]:
            raise RuntimeError("cached loading failed")
        return model(sample).square().mean()

    with pytest.raises(RuntimeError, match="cached loading failed"):
        verify_update(
            model,
            optimizer,
            list(model.named_parameters()),
            [torch.ones(1, 1)],
            forward,
            lambda value: force.__setitem__(0, value),
            device=torch.device("cpu"),
            group=None,
            gloo_group=None,
            loss_scale=1,
            output_dir=tmp_path,
        )
    assert not optimizer.state
    assert force == [False]
    assert all(torch.equal(model.state_dict()[name], value) for name, value in initial.items())


def test_nonfinite_update_writes_valid_json_and_restores_parameters(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    initial = copy.deepcopy(model.state_dict())
    with pytest.raises(VerificationError):
        verify_update(
            model,
            optimizer,
            list(model.named_parameters()),
            [torch.ones(1, 1)],
            lambda sample: model(sample).sum() * float("nan"),
            lambda value: None,
            device=torch.device("cpu"),
            group=None,
            gloo_group=None,
            loss_scale=1,
            output_dir=tmp_path,
            clip_fn=lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1),
        )
    report = json.loads((tmp_path / "conditioning-verification.json").read_text())
    assert not report["loss"]["finite"]
    assert all(not row["finite"] for row in report["parameter_updates"].values())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in initial.items())


def test_update_verification_rejects_bf16_master_weights(tmp_path):
    model = torch.nn.Linear(1, 1).to(dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="FP32 optimizer master"):
        verify_update(
            model,
            optimizer,
            list(model.named_parameters()),
            [torch.ones(1, 1, dtype=torch.bfloat16)],
            lambda sample: model(sample).square().mean(),
            lambda value: None,
            device=torch.device("cpu"),
            group=None,
            gloo_group=None,
            loss_scale=1,
            output_dir=tmp_path,
        )
