# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Policy adapters preserve kernel arguments, gradients and metric
contracts."""

import ast
import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from relax.algorithms.policy import POLICY_LOSS_FNS, compute_policy_loss_for  # noqa: E402
from relax.algorithms.spec import ALGORITHM_SPECS, get_algorithm  # noqa: E402
from relax.utils.training.ppo_utils import (  # noqa: E402
    compute_cispo_loss,
    compute_policy_loss,
    compute_rloo_loss,
    compute_sapo_loss,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _args(estimator, **overrides):
    base = dict(
        advantage_estimator=estimator,
        eps_clip=0.15,
        eps_clip_high=0.3,
        m2po_kl2_budget=0.02,
        m2po_miniclip_low=0.25,
        m2po_miniclip_high=0.4,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _tensors():
    generator = torch.Generator().manual_seed(0)
    return tuple(torch.randn(8, generator=generator) for _ in range(3))


@pytest.mark.parametrize("estimator", ["grpo", "gspo", "ppo", "reinforce_plus_plus", "reinforce_plus_plus_baseline"])
def test_ppo_clip_adapters_pass_mains_arguments(estimator):
    log_probs, ppo_kl, advantages = _tensors()
    args = _args(estimator)
    loss, clipfrac, metrics = compute_policy_loss_for(args, log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    expected = compute_policy_loss(ppo_kl, advantages, 0.15, 0.3)
    assert torch.equal(loss, expected[0])
    assert torch.equal(clipfrac, expected[1])
    assert metrics == {}


@pytest.mark.parametrize("taus", [None, (1.3, 1.7)], ids=["defaults", "explicit"])
def test_sapo_adapter_preserves_defaults_and_explicit_parameters(taus):
    log_probs, ppo_kl, advantages = _tensors()
    overrides = {} if taus is None else dict(sapo_tau_pos=taus[0], sapo_tau_neg=taus[1])
    loss, clipfrac, metrics = compute_policy_loss_for(
        _args("sapo", **overrides), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages
    )
    tau_pos, tau_neg = (1.0, 1.05) if taus is None else taus
    expected = compute_sapo_loss(ppo_kl=ppo_kl, advantages=advantages, tau_pos=tau_pos, tau_neg=tau_neg)
    assert torch.equal(loss, expected[0])
    assert torch.equal(clipfrac, expected[1])
    assert metrics == {}


def test_cispo_adapter_passes_mains_arguments():
    log_probs, ppo_kl, advantages = _tensors()
    loss, clipfrac, metrics = compute_policy_loss_for(
        _args("cispo", eps_clip_high=9.0), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages
    )
    expected = compute_cispo_loss(
        log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages, eps_clip=0.15, eps_clip_high=9.0
    )
    assert torch.equal(loss, expected[0])
    assert torch.equal(clipfrac, expected[1])
    assert metrics == {}


def test_rloo_adapter_passes_mains_arguments_and_ignores_ppo_kl():
    log_probs, _, advantages = _tensors()
    expected = compute_rloo_loss(log_probs=log_probs, advantages=advantages)
    for ppo_kl in (torch.zeros_like(log_probs), torch.full_like(log_probs, 5.0)):
        loss, clipfrac, metrics = compute_policy_loss_for(
            _args("rloo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages
        )
        assert torch.equal(loss, expected[0])
        assert torch.equal(clipfrac, expected[1])
        assert metrics == {}


def _legacy_solve_tau_from_sorted_delta2(sorted_delta2, target_sum):
    """Frozen pre-refactor implementation used as a numerical oracle."""
    n = sorted_delta2.numel()
    total = float(sorted_delta2.sum().item())
    if target_sum >= total - 1e-12:
        return 100000.0, total / n
    if target_sum <= 1e-12:
        return 0.0, 0.0
    csum = torch.cumsum(sorted_delta2, dim=0)
    for k in range(n):
        left_sum = float(csum[k].item())
        rest = n - k - 1
        m2 = sorted_delta2[k].item() - 1e-12
        if m2 * rest + left_sum >= target_sum - 1e-12:
            if k == 0:
                return 0.0, float(csum[-1].item()) / n
            m2_after = (sorted_delta2[k - 1].item() * (rest + 1) + float(csum[k - 1].item())) / n
            return max(sorted_delta2[k - 1].item() - 1e-12, 0.0) ** 0.5, m2_after
    return 100000.0, total / n


def _legacy_compute_m2po_loss(ppo_kl, advantages, kl2_budget, miniclip_low, miniclip_high):
    """Frozen M2PO loss from main before the registry refactor."""
    ratio = (-ppo_kl).exp()
    pos_harmful = (advantages > 1e-12) & (ratio > 1.0 + 1e-12)
    neg_harmful = (advantages < -1e-12) & (ratio < 1.0 - 1e-12)
    tr_delta_sq = ppo_kl[pos_harmful | neg_harmful].pow(2)
    n = tr_delta_sq.numel()
    if n == 0:
        clip_low, clip_high, m2_now, m2_after = 0.0, 100000.0, 0.0, 0.0
    else:
        m2_now = float(tr_delta_sq.sum().detach().item() / n)
        if m2_now <= kl2_budget + 1e-12:
            clip_low, clip_high, m2_after = 0.0, 100000.0, m2_now
        else:
            sorted_delta2, _ = torch.sort(tr_delta_sq)
            tau, m2_after = _legacy_solve_tau_from_sorted_delta2(sorted_delta2, kl2_budget * float(n))
            clip_low, clip_high = math.exp(-tau), math.exp(tau)

    eps_low = max(1.0 - clip_low, miniclip_low)
    eps_high = max(clip_high - 1.0, miniclip_high)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * ratio.clamp(1.0 - eps_low, 1.0 + eps_high)
    pg_loss = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = (pg_losses2 > pg_losses1).float()
    return pg_loss, clipfrac, m2_now, m2_after, eps_low, eps_high


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("ppo_kl", "advantages", "budget"),
    [
        pytest.param([-0.8, -0.4, -0.1, 0.2, 0.7], [1.0, 1.0, -1.0, -1.0, -1.0], 0.02, id="clipped"),
        pytest.param([-0.1, -0.2, 0.1], [1.0, 1.0, -1.0], 1.0, id="within-budget"),
        pytest.param([0.8, -0.4, 0.0], [1.0, -1.0, 0.0], 0.02, id="no-harmful-tokens"),
        pytest.param([-1.0, -2.0, -3.0], [1.0, 1.0, 1.0], 1.0, id="exact-breakpoint"),
    ],
)
def test_m2po_dispatch_preserves_legacy_loss_metrics_and_gradients(dtype, ppo_kl, advantages, budget):
    actual_ppo_kl = torch.tensor(ppo_kl, dtype=dtype, requires_grad=True)
    expected_ppo_kl = actual_ppo_kl.detach().clone().requires_grad_()
    advantages = torch.tensor(advantages, dtype=dtype)
    actual = compute_policy_loss_for(
        _args("m2po", m2po_kl2_budget=budget),
        log_probs=-actual_ppo_kl,
        ppo_kl=actual_ppo_kl,
        advantages=advantages,
    )
    expected = _legacy_compute_m2po_loss(expected_ppo_kl, advantages, budget, 0.25, 0.4)
    actual[0].sum().backward()
    expected[0].sum().backward()
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert list(actual[2]) == list(get_algorithm("m2po").policy_scalar_metric_names)
    for metric, target in zip(actual[2].values(), expected[2:], strict=True):
        torch.testing.assert_close(metric, torch.tensor(target, dtype=torch.float32), rtol=0, atol=0)
        assert metric.shape == ()
        assert not metric.requires_grad
    torch.testing.assert_close(actual_ppo_kl.grad, expected_ppo_kl.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("metric_names", "value", "error"),
    [
        pytest.param((), 1.0, "returned 1 scalar metrics, but its spec declares 0", id="count"),
        pytest.param(("probe",), torch.ones(2), "metric 'probe' must be scalar", id="shape"),
        pytest.param(("probe",), torch.tensor(1 + 2j), "metric 'probe' must be real-valued", id="complex"),
    ],
)
def test_dispatch_rejects_invalid_scalar_metric_contract(monkeypatch, metric_names, value, error):
    log_probs, ppo_kl, advantages = _tensors()
    monkeypatch.setitem(
        ALGORITHM_SPECS,
        "grpo",
        dataclasses.replace(get_algorithm("grpo"), policy_scalar_metric_names=metric_names),
    )
    monkeypatch.setitem(
        POLICY_LOSS_FNS,
        "ppo_clip",
        lambda *args, **kwargs: (torch.zeros_like(ppo_kl), torch.zeros_like(ppo_kl), value),
    )
    with pytest.raises(ValueError, match=error):
        compute_policy_loss_for(_args("grpo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.tensor([1.25], dtype=torch.float64, requires_grad=True), 1.25),
        (torch.tensor(2, dtype=torch.int64), 2.0),
        (torch.tensor(True), 1.0),
    ],
)
def test_dispatch_normalizes_real_scalar_metrics_to_float32(monkeypatch, value, expected):
    log_probs, ppo_kl, advantages = _tensors()
    monkeypatch.setitem(
        ALGORITHM_SPECS,
        "grpo",
        dataclasses.replace(get_algorithm("grpo"), policy_scalar_metric_names=("probe",)),
    )
    monkeypatch.setitem(
        POLICY_LOSS_FNS,
        "ppo_clip",
        lambda *args, **kwargs: (torch.zeros_like(ppo_kl), torch.zeros_like(ppo_kl), value),
    )
    _, _, metrics = compute_policy_loss_for(_args("grpo"), log_probs=log_probs, ppo_kl=ppo_kl, advantages=advantages)
    assert metrics["probe"].dtype == torch.float32
    assert metrics["probe"].shape == ()
    assert not metrics["probe"].requires_grad
    assert metrics["probe"] == expected


def _is_raw_estimator_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "advantage_estimator"
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "advantage_estimator"
    )


def _contains_string_literal(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Constant) and isinstance(child.value, str) for child in ast.walk(node))


def _name_checks(source: str) -> list[str]:
    """Find raw estimator-vs-literal comparisons without matching spec
    fields."""
    lines = source.splitlines()
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        raw_estimator_positions = {i for i, operand in enumerate(operands) if _is_raw_estimator_expression(operand)}
        if raw_estimator_positions and any(
            _contains_string_literal(operand) for i, operand in enumerate(operands) if i not in raw_estimator_positions
        ):
            found.append(lines[node.lineno - 1].strip())
    return found


@pytest.mark.parametrize("path", ["relax/backends/megatron/loss.py", "relax/components/advantages.py"])
def test_algorithm_call_sites_do_not_branch_on_estimator_names(path):
    assert _name_checks((REPO_ROOT / path).read_text(encoding="utf-8")) == []


def test_estimator_name_guard_recognizes_comparisons_but_allows_spec_fields():
    for expression in (
        'args.advantage_estimator == "gspo"',
        'args.advantage_estimator != "ppo"',
        'args.advantage_estimator in ["grpo", "gspo"]',
        'args.advantage_estimator in {"reinforce_plus_plus"}',
        'self.config.advantage_estimator in ("ppo",)',
        'getattr(args, "advantage_estimator", None) == "m2po"',
        '"rloo" == args.advantage_estimator',
    ):
        source = "selected = " + expression
        assert _name_checks(source) == [source]
    assert _name_checks('selected = get_algorithm(args.advantage_estimator).kl_level == "sequence"') == []
