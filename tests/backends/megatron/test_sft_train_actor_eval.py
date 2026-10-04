# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""train_actor must trigger SFT eval every eval_interval steps when
configured."""

from argparse import Namespace

import pytest

from relax.engine.sft.runtime import should_run_sft_eval, should_run_sft_predict


def _mk_actor_args():
    return Namespace(
        loss_type="sft",
        compute_advantages_and_returns=False,
        eval_prompt_data=["eval", "/dev/null"],
        eval_size=None,
        eval_interval=10,
        advantage_estimator="grpo",
        save=None,
        save_interval=None,
        rotate_ckpt=False,
        offload_train=False,
        offload_rollout=False,
        num_rollout=20,
    )


@pytest.mark.parametrize("loss_type", ["sft", "dpo"])
def test_should_run_sft_eval_at_interval_boundary(loss_type):
    args = _mk_actor_args()
    args.loss_type = loss_type
    assert should_run_sft_eval(args, rollout_id=9) is True
    assert should_run_sft_eval(args, rollout_id=19) is True
    assert should_run_sft_eval(args, rollout_id=4) is False
    assert should_run_sft_eval(args, rollout_id=0) is False


def test_should_run_sft_eval_disabled_when_no_interval():
    args = _mk_actor_args()
    args.eval_interval = None
    assert should_run_sft_eval(args, rollout_id=9) is False


def test_should_run_sft_eval_disabled_when_no_eval_source():
    args = _mk_actor_args()
    args.eval_prompt_data = None
    assert should_run_sft_eval(args, rollout_id=9) is False


def test_should_run_sft_eval_disabled_for_non_sft():
    args = _mk_actor_args()
    args.loss_type = "policy_loss"
    assert should_run_sft_eval(args, rollout_id=9) is False


@pytest.mark.parametrize("loss_type", ["sft", "dpo"])
def test_should_run_sft_predict_only_runs_for_sft(loss_type):
    args = _mk_actor_args()
    args.loss_type = loss_type
    args.sft_predict_interval = 10

    assert should_run_sft_predict(args, rollout_id=0) is False
    assert should_run_sft_predict(args, rollout_id=8) is False
    assert should_run_sft_predict(args, rollout_id=9) is (loss_type == "sft")
