# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import ArgumentParser, Namespace

import pytest
import torch

from relax.utils.opd.opd_utils import (
    add_opd_arguments,
    apply_opd_to_advantages,
    compute_policy_opd_loss,
    validate_opd_topk_gather,
)
from tests.cpu.relax_parser import parser_for_flags


def test_opd_topk_gather_uses_declared_allgather_cp() -> None:
    parser = parser_for_flags("--allgather-cp")
    token_ids = [torch.tensor([[1, 2]])]
    assert validate_opd_topk_gather(parser.parse_args([]), token_ids) is None
    with pytest.raises(NotImplementedError, match="allgather_cp=True"):
        validate_opd_topk_gather(parser.parse_args(["--allgather-cp"]), token_ids)


def test_opd_topk_gather_leaves_inactive_sparse_config_unread() -> None:
    assert validate_opd_topk_gather(Namespace(), None) is None


@pytest.mark.parametrize("normalize_topk", [False, True], ids=["tail-default", "normalize-explicit"])
def test_opd_normalization_applies_to_topk_advantages_and_loss(normalize_topk: bool) -> None:
    parser = add_opd_arguments(ArgumentParser())
    cli = ["--opd-token-selection", "teacher_topk", "--opd-log-prob-top-k", "2"]
    if normalize_topk:
        cli += ["--opd-norm-mode", "norm"]
    advantage_args = parser.parse_args(cli)
    loss_args = parser.parse_args([*cli, "--opd-kl-coef", "0", "--opd-loss-coef", "1"])

    student_probs = torch.tensor([[0.4, 0.2]])
    teacher_probs = torch.tensor([[0.3, 0.1]])
    student_lp, teacher_lp = student_probs.log(), teacher_probs.log()
    if normalize_topk:
        student = student_probs / student_probs.sum()
        teacher = teacher_probs / teacher_probs.sum()
    else:
        student = torch.cat((student_probs, 1 - student_probs.sum(dim=1, keepdim=True)), dim=1)
        teacher = torch.cat((teacher_probs, 1 - teacher_probs.sum(dim=1, keepdim=True)), dim=1)
    expected_kl = (student * (student.log() - teacher.log())).sum()

    advantages = [torch.ones(1)]
    apply_opd_to_advantages(
        advantage_args,
        {"opd_topk_student_log_probs": [student_lp], "opd_topk_teacher_log_probs": [teacher_lp]},
        advantages,
    )
    loss, metrics = compute_policy_opd_loss(
        args=loss_args,
        batch={"response_lengths": [1], "loss_masks": [torch.ones(1)], "opd_topk_teacher_log_probs": [teacher_lp]},
        metric_reducer=torch.mean,
        log_probs=torch.zeros(1),
        old_log_probs=torch.zeros(1),
        log_probs_and_entropy={"topk_log_probs": [student_lp]},
    )

    torch.testing.assert_close(advantages[0], (1 - expected_kl).reshape(1))
    torch.testing.assert_close(loss, expected_kl)
    assert not metrics


@pytest.mark.parametrize(
    ("cli", "expected_loss", "expected_metrics"),
    [
        ([], 2.0, {}),
        (["--opd-log-prob-min-clamp", "-2"], 1.0, {}),
        (
            ["--opd-log-prob-min-clamp", "-2", "--opd-per-token-clip", "0.75", "--opd-is-clip", "0.5"],
            0.375,
            {"opd_per_token_clip_frac": 1.0, "opd_is_clip_frac": 1.0},
        ),
    ],
    ids=["unclipped-default", "log-prob-clamp", "per-token-and-is-clips"],
)
def test_sampled_opd_clamp_and_clip_options_preserve_values_and_metrics(
    cli: list[str], expected_loss: float, expected_metrics: dict[str, float]
) -> None:
    parser = add_opd_arguments(ArgumentParser())
    args = parser.parse_args(["--opd-kl-coef", "0", "--opd-loss-coef", "1", *cli])
    log_probs = torch.tensor([-1.0])
    loss, metrics = compute_policy_opd_loss(
        args=args,
        batch={"response_lengths": [1], "loss_masks": [torch.ones(1)], "teacher_log_probs": [torch.tensor([-3.0])]},
        metric_reducer=torch.mean,
        log_probs=log_probs,
        old_log_probs=log_probs,
        log_probs_and_entropy={},
    )

    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    assert {name: value.item() for name, value in metrics.items()} == expected_metrics


@pytest.mark.parametrize("zero_threshold", [False, True], ids=["threshold-default", "threshold-zero"])
def test_entropy_gate_uses_teacher_topk_for_loss_and_sampled_advantages(zero_threshold: bool) -> None:
    parser = add_opd_arguments(ArgumentParser())
    cli = [
        "--opd-token-selection",
        "teacher_topk",
        "--opd-log-prob-top-k",
        "2",
        "--opd-loss-coef",
        "1",
        "--opd-fkl-entropy-gate",
    ]
    if zero_threshold:
        cli += ["--opd-fkl-entropy-threshold", "0"]
    args = parser.parse_args(cli)
    student = torch.tensor([[0.4, 0.2, 0.4]])
    teacher = torch.tensor([[0.2, 0.2, 0.6]])
    student_lp, teacher_lp = student[:, :2].log(), teacher[:, :2].log()
    expected_loss = (teacher * (teacher.log() - student.log())).sum() if zero_threshold else torch.tensor(0.0)

    advantages = [torch.ones(1)]
    apply_opd_to_advantages(
        args,
        {"rollout_log_probs": [torch.tensor([-0.3])], "teacher_log_probs": [torch.tensor([-0.8])]},
        advantages,
    )
    loss, metrics = compute_policy_opd_loss(
        args=args,
        batch={"response_lengths": [1], "loss_masks": [torch.ones(1)], "opd_topk_teacher_log_probs": [teacher_lp]},
        metric_reducer=torch.mean,
        log_probs=torch.zeros(1),
        old_log_probs=torch.zeros(1),
        log_probs_and_entropy={"topk_log_probs": [student_lp]},
    )

    torch.testing.assert_close(advantages[0], torch.tensor([0.5]))
    torch.testing.assert_close(loss, expected_loss)
    assert metrics["opd_fkl_gate_frac"].item() == float(zero_threshold)


def test_policy_loss_keeps_no_op_for_disabled_opd() -> None:
    args = add_opd_arguments(ArgumentParser()).parse_args(["--opd-loss-coef", "0"])
    assert compute_policy_opd_loss(
        args=args,
        batch={},
        metric_reducer=torch.mean,
        log_probs=torch.empty(0),
        old_log_probs=torch.empty(0),
        log_probs_and_entropy={},
    ) == (None, {})
