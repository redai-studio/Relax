# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real Advantages input/KL regression; no Serve deployment or TQ client."""

from argparse import Namespace

import pytest
import torch

from relax.components import advantages as component


@pytest.mark.parametrize("container", ["list", "dense", "jagged", "empty_token_row"])
def test_advantages_computes_nonzero_kl_without_tensor_truthiness(monkeypatch, container):
    lengths = (3, 3) if container == "dense" else (0, 2) if container == "empty_token_row" else (3, 2)
    rows = [torch.arange(n, dtype=torch.float32, device="cpu") for n in lengths]
    refs = [row - 0.25 for row in rows]
    log_probs = torch.stack(rows) if container == "dense" else rows
    ref_log_probs = torch.stack(refs) if container == "dense" else refs
    if container in {"jagged", "empty_token_row"}:
        log_probs = torch.nested.nested_tensor(rows, layout=torch.jagged, device="cpu")
        ref_log_probs = torch.nested.nested_tensor(refs, layout=torch.jagged, device="cpu")
    _check_result(monkeypatch, log_probs, ref_log_probs, rows, lengths, kl_coef=0.1, expected_kl=0.25)


@pytest.mark.parametrize("absent", [False, True])
def test_advantages_preserves_zero_kl_and_onpolicy_template(monkeypatch, absent):
    rows = [torch.ones(n, device="cpu") for n in (3, 2)]
    log_probs = None if absent else torch.nested.nested_tensor(rows, layout=torch.jagged, device="cpu")
    _check_result(monkeypatch, log_probs, None, rows, (3, 2), kl_coef=0.1 if absent else 0.0, expected_kl=0.0)


def _check_result(monkeypatch, log_probs, refs, rows, lengths, *, kl_coef, expected_kl):
    cls = component.Advantages.func_or_class
    owner = object.__new__(cls)
    owner.config = Namespace(
        use_rollout_logprobs=False, kl_coef=kl_coef, kl_loss_type="k1", advantage_estimator="grpo"
    )
    observed = []
    original = component.compute_advantages_and_returns

    def observe(*args, **kwargs):
        observed.extend(kwargs["kl"])
        return original(*args, **kwargs)

    monkeypatch.setattr(component, "compute_advantages_and_returns", observe)
    result = owner.compute_advantages_and_returns(
        dict(
            log_probs=log_probs,
            ref_log_probs=refs,
            rollout_log_probs=rows,
            rewards=[1.5, -2.0],
            loss_masks=[torch.ones(n, device="cpu") for n in lengths],
            response_lengths=list(lengths),
            total_lengths=[n + 2 for n in lengths],
        )
    )
    assert len(observed) == 2
    for actual, n in zip(observed, lengths, strict=True):
        torch.testing.assert_close(actual, torch.full((n,), expected_kl, device="cpu"))
    for key in ("advantages", "returns"):
        for actual, n, reward in zip(result[key].unbind(), lengths, (1.5, -2.0), strict=True):
            torch.testing.assert_close(actual, torch.full((n,), reward, device="cpu"))


def test_advantages_preserves_intermediate_pipeline_early_return():
    cls = component.Advantages.func_or_class
    owner = object.__new__(cls)
    owner.config = Namespace(use_rollout_logprobs=False)
    assert owner.compute_advantages_and_returns({}) is None
