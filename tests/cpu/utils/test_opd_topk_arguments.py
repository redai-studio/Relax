# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse

import pytest
import torch

from relax.utils.opd.opd_main_worker import TopkWorker, restore_opd_topk_rollout_fields
from relax.utils.opd.opd_utils import add_opd_arguments


@pytest.mark.parametrize("top_k", [None, 0, 2])
def test_restore_opd_topk_reads_the_declared_configuration(top_k: int | None) -> None:
    parser = add_opd_arguments(argparse.ArgumentParser())
    cli = ["--opd-token-selection", "student_topk"]
    if top_k is not None:
        cli += ["--opd-log-prob-top-k", str(top_k)]
    args = parser.parse_args(cli)
    field = TopkWorker.TRANSFER_TOKEN_IDS
    rows = {field: [[1, 2, 3, 4], []]}

    restore_opd_topk_rollout_fields(rows, args, torch.device("cpu"))

    if top_k == 2:
        assert torch.equal(rows[field][0], torch.tensor([[1, 2], [3, 4]]))
        assert rows[field][1].shape == (0, 2)
    else:
        assert rows[field] == [[1, 2, 3, 4], []]


def test_restore_opd_topk_rejects_a_missing_declared_width() -> None:
    parser = add_opd_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--opd-token-selection", "student_topk"])
    del args.opd_log_prob_top_k

    with pytest.raises(AttributeError, match="opd_log_prob_top_k"):
        restore_opd_topk_rollout_fields({}, args, torch.device("cpu"))
