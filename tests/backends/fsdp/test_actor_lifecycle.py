# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Deterministic FlowGRPO update over a synthetic trajectory.

A synthetic model + adapter must complete a deterministic FlowGRPO optimizer
update, with on-policy ratio == 1 on the first step and gradients flowing to
every trainable parameter.
"""

from __future__ import annotations

import torch

from relax.backends.fsdp.actor import _plan_micro_batch_updates, _replay_logp, flow_grpo_update


class _SyntheticModel(torch.nn.Module):
    """Learnable velocity: ``noise_pred = scale * x_t``."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.3))


class _SyntheticAdapter:
    family = "synthetic"
    supported_tasks = ("t2i",)

    def replay_transition(self, model, batch, step_index):
        slot = batch["sde_indices"].tolist().index(int(step_index))
        return model.scale * batch["image_x_t"][:, slot]


def _make_batch(b=4, shape=(3, 4, 4)):
    torch.manual_seed(0)
    sde_indices = torch.tensor([0, 1])
    return {
        "sigmas": torch.linspace(1.0, 0.0, 4),  # 3 steps
        "sde_indices": sde_indices,
        "image_x_t": torch.randn(b, len(sde_indices), *shape),
        "image_x_next": torch.randn(b, len(sde_indices), *shape),
    }


def _run_update():
    model = _SyntheticModel()
    adapter = _SyntheticAdapter()
    batch = _make_batch()
    step_indices = [0, 1]
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

    with torch.no_grad():
        old_logp = _replay_logp(model, adapter, batch, step_indices, eta=0.7, sigma_max=0.99)

    result = flow_grpo_update(
        model,
        adapter,
        batch,
        advantages=advantages,
        old_logp=old_logp,
        step_indices=step_indices,
        eta=0.7,
        clip_eps=0.2,
    )
    return model, result


def test_logp_keeps_the_step_axis_for_per_step_clipping():
    """``_replay_logp`` must return [B, S], not a sum over S.

    Summing makes the PPO ratio the PRODUCT of the per-step ratios, so one
    ``clip_eps`` band has to hold the combined drift of every step and one step
    drifting out clips the whole sample. FlowGRPO clips per (sample, step).
    """
    out = _replay_logp(_SyntheticModel(), _SyntheticAdapter(), _make_batch(b=4), [0, 1], eta=0.7, sigma_max=0.99)
    assert out.shape == (4, 2)  # [B, S], one column per trained step


def test_clipping_is_per_step_not_on_the_product_of_ratios():
    """A single drifting step must not clip the other steps' gradient.

    Build an anchor that is off by +2*eps on step 0 only and exactly right on
    step 1. Per-step clipping engages on half the elements; product clipping
    would see one out-of-band ratio and clip 100% of the sample.
    """
    step_indices = [0, 1]
    model = _SyntheticModel()
    adapter = _SyntheticAdapter()
    batch = _make_batch(b=4)
    with torch.no_grad():
        anchor = _replay_logp(model, adapter, batch, step_indices, eta=0.7, sigma_max=0.99).clone()
    eps = 0.2
    anchor[:, 0] -= 3 * eps  # step 0 far out of band; step 1 exactly on-policy

    result = flow_grpo_update(
        model,
        adapter,
        batch,
        advantages=torch.ones(4),
        old_logp=anchor,
        step_indices=step_indices,
        eta=0.7,
        clip_eps=eps,
    )
    # 4 samples x 2 steps: exactly the 4 step-0 elements clip.
    assert abs(float(result["metrics"]["clip_fraction"]) - 0.5) < 1e-6


def test_deterministic_update():
    model, result = _run_update()
    assert torch.isfinite(result["loss"])
    # on-policy first update: ratio == 1
    assert abs(float(result["metrics"]["ratio_mean"]) - 1.0) < 1e-5
    # gradient reached the trainable parameter
    assert model.scale.grad is not None


def test_off_policy_ratio_deviates_after_weight_change():
    model = _SyntheticModel()
    adapter = _SyntheticAdapter()
    batch = _make_batch()
    step_indices = [0, 1]
    with torch.no_grad():
        anchor = _replay_logp(model, adapter, batch, step_indices, eta=0.7, sigma_max=0.99)
    # perturb weights -> replayed logp differs -> ratio != 1
    with torch.no_grad():
        model.scale.add_(0.5)
    result = flow_grpo_update(
        model,
        adapter,
        batch,
        advantages=torch.ones(4),
        old_logp=anchor,
        step_indices=step_indices,
        eta=0.7,
        clip_eps=0.2,
    )
    assert abs(float(result["metrics"]["ratio_mean"]) - 1.0) > 1e-3


# ---------------------------------------------------------------------------
# Gradient accumulation over the global batch
# ---------------------------------------------------------------------------


def _micro_batch(seed: int, b=4, shape=(3, 4, 4)):
    """One synthetic micro-batch (= one prompt group) with its advantages."""
    torch.manual_seed(seed)
    sde_indices = torch.tensor([0, 1])
    batch = {
        "sigmas": torch.linspace(1.0, 0.0, 4),
        "sde_indices": sde_indices,
        "image_x_t": torch.randn(b, len(sde_indices), *shape),
        "image_x_next": torch.randn(b, len(sde_indices), *shape),
    }
    return batch, torch.randn(b)


def _anchor(model, adapter, batch, step_indices):
    with torch.no_grad():
        return _replay_logp(model, adapter, batch, step_indices, eta=0.7, sigma_max=0.99)


def test_grad_accumulation_equals_mean_of_micro_batch_grads():
    """1/N-scaled accumulation == the mean of the per-micro-batch gradients.

    This is the invariant that makes --global-batch-size the real optimizer-step
    boundary: N micro-batches accumulated with loss_scale=1/N must produce the
    same gradient as averaging their individual gradients.
    """
    adapter = _SyntheticAdapter()
    step_indices = [0, 1]
    mbs = [_micro_batch(seed) for seed in (0, 1, 2)]

    # (a) each micro-batch's gradient in isolation (unscaled, fresh grads).
    per_mb_grads = []
    for batch, advantages in mbs:
        model = _SyntheticModel()
        model.zero_grad(set_to_none=True)
        old_logp = _anchor(model, adapter, batch, step_indices)
        flow_grpo_update(
            model,
            adapter,
            batch,
            advantages=advantages,
            old_logp=old_logp,
            step_indices=step_indices,
            eta=0.7,
            clip_eps=0.2,
            loss_scale=1.0,
        )
        per_mb_grads.append(model.scale.grad.clone())
    expected = torch.stack(per_mb_grads).mean(dim=0)

    # (b) one model, no zero_grad between micro-batches, each scaled by 1/N.
    model = _SyntheticModel()
    model.zero_grad(set_to_none=True)
    scale = 1.0 / len(mbs)
    for batch, advantages in mbs:
        old_logp = _anchor(model, adapter, batch, step_indices)
        flow_grpo_update(
            model,
            adapter,
            batch,
            advantages=advantages,
            old_logp=old_logp,
            step_indices=step_indices,
            eta=0.7,
            clip_eps=0.2,
            loss_scale=scale,
        )

    assert torch.allclose(model.scale.grad, expected, atol=1e-6), (
        f"accumulated grad {model.scale.grad} != mean-of-micro-batch grads {expected}"
    )


def test_num_updates_per_batch_plans_disjoint_equal_sample_updates():
    mbs = [{"advantages": torch.zeros(n)} for n in (1, 1, 1, 3)]

    assert _plan_micro_batch_updates(mbs, 1) == [[0, 1, 2, 3]]
    assert _plan_micro_batch_updates(mbs, 2) == [[0, 1, 2], [3]]


def test_num_updates_per_batch_rejects_crossing_micro_batch_boundary():
    mbs = [{"advantages": torch.zeros(n)} for n in (2, 2, 2)]

    try:
        _plan_micro_batch_updates(mbs, 2)
    except ValueError as exc:
        assert "crosses a num_updates_per_batch boundary" in str(exc)
    else:
        raise AssertionError("expected update-plan boundary error")


# ---------------------------------------------------------------------------
# Self-anchoring (old_logp=None) on the first mini-epoch
# ---------------------------------------------------------------------------


def test_self_anchor_matches_explicit_anchor_bit_for_bit():
    """``old_logp=None`` must reproduce the explicit anchor exactly.

    The explicit anchor is a second, no_grad replay of the same transitions at
    the same (pre-update) weights, so its value is by construction equal to the
    update's own ``new_logp``. Dropping it is a pure saving only if loss and
    gradient come out identical -- assert that, not just "close".
    """
    step_indices = [0, 1]
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

    def _run(explicit: bool):
        torch.manual_seed(7)
        model = _SyntheticModel()
        adapter = _SyntheticAdapter()
        batch = _make_batch()
        old_logp = None
        if explicit:
            with torch.no_grad():
                old_logp = _replay_logp(model, adapter, batch, step_indices, eta=0.7, sigma_max=0.99)
        model.zero_grad(set_to_none=True)
        result = flow_grpo_update(
            model,
            adapter,
            batch,
            advantages=advantages,
            old_logp=old_logp,
            step_indices=step_indices,
            eta=0.7,
            clip_eps=0.2,
        )
        return result, model.scale.grad.clone()

    explicit_result, explicit_grad = _run(True)
    self_result, self_grad = _run(False)

    assert torch.equal(self_result["loss"], explicit_result["loss"])
    assert torch.equal(self_grad, explicit_grad)
    assert torch.equal(self_result["new_logp"], explicit_result["new_logp"])
    # The on-policy invariants the anchor exists to guarantee.
    assert float(self_result["metrics"]["ratio_mean"]) == 1.0
    assert float(self_result["metrics"]["clip_fraction"]) == 0.0


def test_self_anchor_returned_logp_is_a_usable_anchor_for_later_epochs():
    """The returned ``new_logp`` must be detached and reusable as π_old.

    ``train()`` stashes it for u >= 1; if it were still attached to the graph
    the second mini-epoch would backprop through the first one's replay.
    """
    step_indices = [0, 1]
    model = _SyntheticModel()
    adapter = _SyntheticAdapter()
    batch = _make_batch()

    first = flow_grpo_update(
        model,
        adapter,
        batch,
        advantages=torch.ones(4),
        old_logp=None,
        step_indices=step_indices,
        eta=0.7,
        clip_eps=0.2,
    )
    anchor = first["new_logp"]
    assert not anchor.requires_grad

    with torch.no_grad():  # simulate the optimizer step between mini-epochs
        model.scale.add_(0.5)
    second = flow_grpo_update(
        model,
        adapter,
        batch,
        advantages=torch.ones(4),
        old_logp=anchor,
        step_indices=step_indices,
        eta=0.7,
        clip_eps=0.2,
    )
    assert abs(float(second["metrics"]["ratio_mean"]) - 1.0) > 1e-3
