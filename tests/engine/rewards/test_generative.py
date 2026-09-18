# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Generative reward: component normalization, group barrier, fail-fast,
degeneracy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relax.engine.rewards.generative import combine_group_advantages
from relax.utils.types import Sample


class SpreadScorer:
    """Module-level fake scorer, reachable by dotpath for the reward
    manager."""

    required_tracks = ("image",)

    def __init__(self, args):
        self.args = args

    def onload(self):
        pass

    def offload(self):
        pass

    def score_batch(self, requests):
        # Distinct per candidate so the group advantage is non-degenerate.
        return {"pickscore": [float(i) for i in range(len(requests))]}


class FlatScorer(SpreadScorer):
    """Identical score for every candidate → degenerate round."""

    def score_batch(self, requests):
        return {"pickscore": [5.0 for _ in requests]}


class ShortScorer(SpreadScorer):
    """Broken scorer that returns fewer rewards than requests."""

    def score_batch(self, requests):
        return {"pickscore": [1.0]}


def test_combine_group_advantages_weighted_two_components():
    # Multi-component weighting stays generic even though t2i ships one scorer.
    comp = {
        "pickscore": [1.0, 2.0, 3.0, 4.0],
        "aesthetic": [1.0, 4.0, 2.0, 3.0],
    }
    gi = [[0, 1], [2, 3]]
    adv, degenerate = combine_group_advantages(
        comp, gi, {"pickscore": 0.5, "aesthetic": 0.5}, required_components=["pickscore", "aesthetic"]
    )
    assert len(adv) == 4
    assert not degenerate


def test_missing_required_component_raises():
    with pytest.raises(ValueError):
        combine_group_advantages(
            {"aesthetic": [1.0, 2.0]}, [[0, 1]], {"pickscore": 1.0}, required_components=["pickscore"]
        )


def test_non_finite_reward_raises():
    with pytest.raises(ValueError):
        combine_group_advantages(
            {"pickscore": [1.0, float("nan")]}, [[0, 1]], {"pickscore": 1.0}, required_components=["pickscore"]
        )
    with pytest.raises(ValueError):
        combine_group_advantages(
            {"pickscore": [1.0, float("nan")]}, [[0, 1]], {"pickscore": 1.0}, required_components=[]
        )


def test_degenerate_advantage_flagged():
    # all rewards equal within each group → zero variance advantage
    comp = {"pickscore": [5.0, 5.0, 5.0, 5.0]}
    gi = [[0, 1], [2, 3]]
    adv, degenerate = combine_group_advantages(comp, gi, {"pickscore": 1.0}, required_components=["pickscore"])
    assert degenerate
    assert all(abs(a) < 1e-3 for a in adv)


def test_std_normalization_matches_the_text_path_semantics():
    """``std_normalization`` must mean what it means everywhere else in Relax.

    The text path (``post_process_rewards``) reads ``grpo_std_normalization``
    as "divide the group-centered reward by that group's std"; clearing it via
    ``--disable-grpo-std-normalization`` is Dr.GRPO's "do not divide at all".
    This path used to wire the same flag to ``use_global_std``, which inverted
    it: setting Dr.GRPO switched the divisor to per-group std, and clearing it
    divided by one batch-wide std no other estimator uses.
    """
    # Group 0 spread 2.0, group 1 spread 0.2. Per-group std normalization must
    # give both groups the SAME advantage magnitude; centering alone must not.
    comp = {"pickscore": [1.0, 3.0, 10.0, 10.2]}
    gi = [[0, 1], [2, 3]]

    normed, _ = combine_group_advantages(
        comp, gi, {"pickscore": 1.0}, required_components=["pickscore"], std_normalization=True
    )
    assert normed[1] == pytest.approx(normed[3], abs=1e-3)
    assert normed[1] == pytest.approx(0.7071, abs=1e-3)  # 1 / std([1, 3]), unbiased

    centered, _ = combine_group_advantages(
        comp, gi, {"pickscore": 1.0}, required_components=["pickscore"], std_normalization=False
    )
    assert centered == pytest.approx([-1.0, 1.0, -0.1, 0.1], abs=1e-5)


def test_batch_advantage_std_mode_matches_reference_global_std():
    """Reference diffusion recipes divide centered rewards by one batch std."""
    comp = {"pickscore": [0.0, 10.0, 0.0, 1.0]}
    gi = [[0, 1], [2, 3]]

    adv, degenerate = combine_group_advantages(
        comp,
        gi,
        {"pickscore": 1.0},
        required_components=["pickscore"],
        advantage_std_mode="batch",
    )

    assert not degenerate
    assert adv[1] / adv[3] == pytest.approx(10.0, rel=1e-4)


def test_explicit_advantage_std_mode_overrides_legacy_bool():
    comp = {"pickscore": [1.0, 3.0, 10.0, 10.2]}
    gi = [[0, 1], [2, 3]]

    adv, _ = combine_group_advantages(
        comp,
        gi,
        {"pickscore": 1.0},
        required_components=["pickscore"],
        std_normalization=True,
        advantage_std_mode="none",
    )

    assert adv == pytest.approx([-1.0, 1.0, -0.1, 0.1], abs=1e-5)


def test_std_normalization_off_is_finite_for_a_singleton_group():
    # arguments.py force-clears grpo_std_normalization when n_samples_per_prompt
    # == 1, which is exactly the case where a per-group std is undefined.
    adv, degenerate = combine_group_advantages(
        {"pickscore": [1.0, 2.0]},
        [[0], [1]],
        {"pickscore": 1.0},
        required_components=["pickscore"],
        std_normalization=False,
    )
    assert adv == [0.0, 0.0]
    assert degenerate


def test_compute_reward_metrics_varied_rewards():
    from relax.engine.rewards.generative import compute_reward_metrics

    comp = {"pickscore": [0.1, 0.9, 0.2, 0.8]}
    gi = [[0, 1], [2, 3]]
    adv = [-1.0, 1.0, -1.0, 1.0]
    m = compute_reward_metrics(comp, adv, gi, degenerate=False, primary="pickscore")
    assert m["reward/pickscore_min"] == pytest.approx(0.1)
    assert m["reward/pickscore_max"] == pytest.approx(0.9)
    assert m["reward/pickscore_std"] > 0.0
    assert m["reward/num_groups"] == 2.0
    assert m["reward/degenerate"] == 0.0
    # per-group std is the discriminating signal → clearly > 0 here
    assert m["reward/group_std_mean"] > 0.0
    assert m["reward/advantage_std"] > 0.0


def test_compute_reward_metrics_degenerate_zero_group_std():
    from relax.engine.rewards.generative import compute_reward_metrics

    # identical within each group → group_std_mean == 0, degenerate flag surfaced
    comp = {"pickscore": [5.0, 5.0, 5.0, 5.0]}
    gi = [[0, 1], [2, 3]]
    m = compute_reward_metrics(comp, [0.0, 0.0, 0.0, 0.0], gi, degenerate=True, primary="pickscore")
    assert m["reward/group_std_mean"] == pytest.approx(0.0)
    assert m["reward/degenerate"] == 1.0
    assert m["reward/pickscore_std"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# post_process hook
# ---------------------------------------------------------------------------


def _post_process_args(**kw):
    base = dict(
        reward_runtime="cpu",
        reward_scorer_path="tests.engine.rewards.test_generative.SpreadScorer",
        reward_model_path=None,
        reward_endpoint=None,
        reward_component_weights={"pickscore": 1.0},
        reward_required_components=["pickscore"],
        grpo_std_normalization=True,
        generative_advantage_std_mode=None,
        # tracking sink: absent backends are tolerated by _log_reward_metrics
        use_metrics_service=False,
        use_wandb=False,
        use_tensorboard=False,
        use_clearml=False,
        notify_urls=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_post_process_keeps_the_raw_reward_on_the_sample():
    """``Sample.reward`` must stay the scorer's output, not the advantage.

    Overwriting it made every downstream ``get_reward_value`` consumer report a
    group-centered number (mean ~0 by construction) as if it were a reward.
    """
    from relax.engine.rewards.generative import post_process

    samples = [Sample(prompt="p", group_index=0, index=i) for i in range(4)]
    for slot, s in enumerate(samples):
        s.train_metadata = {"group_index": 0, "trajectory_slot": slot, "manifest": {"outputs": []}}

    args = _post_process_args()
    raw_rewards, advantages = post_process(args, samples)

    assert raw_rewards == [0.0, 1.0, 2.0, 3.0]
    assert [s.reward for s in samples] == raw_rewards
    assert [s.get_reward_value(SimpleNamespace(reward_key=None)) for s in samples] == raw_rewards
    # The advantage is returned to the caller (it rides the TQ 'advantages'
    # column) and is also carried alongside the reward, never on top of it.
    assert [s.train_metadata["advantage"] for s in samples] == pytest.approx(advantages)
    assert advantages != raw_rewards


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, ["pickscore"]), ([], []), (["pickscore"], ["pickscore"])],
)
def test_post_process_preserves_explicit_required_components(monkeypatch, configured, expected):
    from relax.engine.rewards import generative

    captured = []

    def _combine(component_rewards, _groups, _weights, *, required_components, **_kwargs):
        captured.extend(required_components)
        return [0.0] * len(component_rewards["pickscore"]), False

    monkeypatch.setattr(generative, "combine_group_advantages", _combine)
    samples = [Sample(prompt="p", group_index=0, index=i) for i in range(2)]
    for sample in samples:
        sample.train_metadata = {"manifest": {"outputs": []}}

    generative.post_process(_post_process_args(reward_required_components=configured), samples)

    assert captured == expected


def test_post_process_uses_explicit_batch_advantage_mode():
    from relax.engine.rewards.generative import post_process

    samples = [Sample(prompt="p", group_index=i // 2, index=i) for i in range(4)]
    for slot, s in enumerate(samples):
        s.train_metadata = {"group_index": slot // 2, "manifest": {"outputs": []}}

    args = _post_process_args(generative_advantage_std_mode="batch")
    _, advantages = post_process(args, samples)

    expected = float(0.5 / torch.tensor([0.0, 1.0, 2.0, 3.0]).std())
    assert advantages == pytest.approx([-expected, expected, -expected, expected], abs=1e-5)


def test_post_process_flags_a_degenerate_round_even_without_train_metadata():
    """The skip flag must land on the sample.

    ``(s.train_metadata or {}).setdefault(...)`` mutated a throwaway dict when
    train_metadata was None, so a degenerate round silently ran the optimizer.
    """
    from relax.engine.rewards.generative import post_process

    samples = [Sample(prompt="p", group_index=0, index=i) for i in range(2)]
    args = _post_process_args(reward_scorer_path="tests.engine.rewards.test_generative.FlatScorer")

    post_process(args, samples)

    assert all(s.train_metadata["skip_optimizer_step"] for s in samples)


def test_post_process_rejects_mismatched_reward_lengths():
    from relax.engine.rewards.generative import post_process

    samples = [Sample(prompt="p", group_index=0, index=i) for i in range(2)]
    args = _post_process_args(reward_scorer_path="tests.engine.rewards.test_generative.ShortScorer")

    with pytest.raises(ValueError, match="length mismatch"):
        post_process(args, samples)
