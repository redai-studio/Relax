# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""compute_ppl_metrics turns already-reduced sums (total negative log prob,
total tokens) into the eval/loss + eval/ppl + eval/num_tokens dict."""

import math
import sys
from argparse import Namespace
from types import ModuleType

import torch


def test_compute_ppl_metrics_aggregates_correctly():
    """Known totals → loss = sum/tokens, ppl = exp(loss)."""
    from relax.engine.sft.eval.ppl import compute_ppl_metrics

    metrics = compute_ppl_metrics(total_neg_log_prob=10.0, total_tokens=6)

    expected_loss = 10.0 / 6.0
    assert math.isclose(metrics["eval/loss"], expected_loss, rel_tol=1e-6)
    assert math.isclose(metrics["eval/ppl"], math.exp(expected_loss), rel_tol=1e-5)
    assert metrics["eval/num_tokens"] == 6


def test_compute_ppl_metrics_empty_returns_zero_safely():
    """Zero tokens → all zeros, no DivisionByZero."""
    from relax.engine.sft.eval.ppl import compute_ppl_metrics

    metrics = compute_ppl_metrics(total_neg_log_prob=0.0, total_tokens=0)
    assert metrics["eval/num_tokens"] == 0
    assert metrics["eval/loss"] == 0.0
    assert metrics["eval/ppl"] == 0.0


def test_compute_sft_eval_step_forwards_dynamic_cp_context(monkeypatch):
    """Eval must slice logits and masks with the microbatch's dynamic CP."""
    from relax.engine.sft.eval.ppl import compute_sft_eval_step

    calls = {}

    def fake_get_log_probs_and_entropy(logits, **kwargs):
        calls["log_probs"] = kwargs
        return torch.empty(0), {"log_probs": [torch.tensor([-2.0, -3.0])]}

    def fake_get_sum_of_sample_mean(*args, **kwargs):
        calls["sum"] = kwargs
        return lambda values: values.sum()

    cp_utils = ModuleType("relax.backends.megatron.cp_utils")
    cp_utils.get_sum_of_sample_mean = fake_get_sum_of_sample_mean
    loss = ModuleType("relax.backends.megatron.loss")
    loss.get_log_probs_and_entropy = fake_get_log_probs_and_entropy
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.cp_utils", cp_utils)
    monkeypatch.setitem(sys.modules, "relax.backends.megatron.loss", loss)

    _, metrics = compute_sft_eval_step(
        torch.empty(0),
        args=Namespace(qkv_format="thd"),
        unconcat_tokens=[torch.tensor([1, 2, 3])],
        total_lengths=[3],
        response_lengths=[2],
        loss_masks=[torch.ones(2)],
        dynamic_cp_size=2,
        dynamic_cp_rank=1,
    )

    assert calls["log_probs"]["dynamic_cp_size"] == 2
    assert calls["log_probs"]["dynamic_cp_rank"] == 1
    assert calls["sum"]["dynamic_cp_size"] == 2
    assert calls["sum"]["dynamic_cp_rank"] == 1
    assert metrics["sum_neg_log_prob"][0].item() == 5.0
    assert metrics["num_tokens"][0].item() == 2
