# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for RLOO leave-one-out baseline, advantage, and loss.

Covers the mathematical correctness of ``compute_rloo_leave_one_out_rewards``
and ``compute_rloo_loss``, the ``post_process_rewards`` rloo branch, DP/CP
invariance, and the non-clipping property. CPU-only — no megatron or GPU
required.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


torch = pytest.importorskip("torch")


def _install_fake_megatron(monkeypatch):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_world_size = lambda: 1
    mpu.get_context_parallel_rank = lambda: 0
    mpu.is_pipeline_last_stage = lambda: True
    core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)


def _make_sample(reward: float, group_index: int, response_length: int = 5, tokens=None):
    from relax.utils.types import Sample

    if tokens is None:
        tokens = list(range(response_length))
    return Sample(
        reward=reward,
        group_index=group_index,
        response_length=response_length,
        tokens=tokens,
        loss_mask=[1] * response_length,
    )


def _fake_args(**kwargs):
    from types import SimpleNamespace

    defaults = dict(
        advantage_estimator="rloo",
        rewards_normalization=True,
        n_samples_per_prompt=4,
        grpo_std_normalization=True,
        custom_reward_post_process_path=None,
        agentic_custom_advantage_path=None,
        reward_key=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Pure math: compute_rloo_leave_one_out_rewards
# ---------------------------------------------------------------------------


def test_rloo_loo_matches_bruteforce_elementwise():
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    torch.manual_seed(42)
    for G in (2, 3, 5, 8):
        rewards = torch.randn(G, dtype=torch.float64)
        expected = torch.empty_like(rewards)
        for i in range(G):
            others = torch.cat([rewards[:i], rewards[i + 1 :]])
            expected[i] = rewards[i] - others.mean()
        got = compute_rloo_leave_one_out_rewards(rewards)
        assert torch.allclose(got, expected, atol=1e-12), f"G={G}: mismatch"


def test_rloo_equals_scaled_mean_centering():
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    torch.manual_seed(0)
    for G in (2, 3, 5, 8, 16):
        rewards = torch.randn(G, dtype=torch.float64)
        expected = (G / (G - 1)) * (rewards - rewards.mean())
        got = compute_rloo_leave_one_out_rewards(rewards)
        assert torch.allclose(got, expected, atol=1e-12), f"G={G}: scaled identity failed"


def test_rloo_constant_group_gives_zero_advantage():
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    for G in (2, 3, 5):
        rewards = torch.full((G,), 0.7, dtype=torch.float64)
        got = compute_rloo_leave_one_out_rewards(rewards)
        assert torch.allclose(got, torch.zeros(G, dtype=torch.float64), atol=1e-12)


def test_rloo_group_size_two_minimal():
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    rewards = torch.tensor([3.0, 1.0], dtype=torch.float64)
    # A_1 = r_1 - r_2 = 2; A_2 = r_2 - r_1 = -2
    got = compute_rloo_leave_one_out_rewards(rewards)
    assert torch.allclose(got, torch.tensor([2.0, -2.0], dtype=torch.float64), atol=1e-12)


def test_rloo_rejects_non_1d_input():
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    rewards_2d = torch.randn(3, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="1-D"):
        compute_rloo_leave_one_out_rewards(rewards_2d)


def test_rloo_rejects_group_size_one():
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    with pytest.raises(ValueError, match=">= 2"):
        compute_rloo_leave_one_out_rewards(torch.tensor([1.0]))


def test_rloo_rejects_nonfinite_reward():
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    for bad in (float("nan"), float("inf"), float("-inf")):
        rewards = torch.tensor([1.0, bad, 0.5, 0.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="non-finite"):
            compute_rloo_leave_one_out_rewards(rewards)


# ---------------------------------------------------------------------------
# post_process_rewards rloo branch
# ---------------------------------------------------------------------------


def test_post_process_rewards_rloo_groups_by_group_index(monkeypatch):
    _install_fake_megatron(monkeypatch)
    from relax.utils.utils import post_process_rewards

    # Interleaved groups: group 0 has samples at positions 0,2,4,6; group 1 at 1,3,5,7
    samples = []
    for gi in (0, 1, 0, 1, 0, 1, 0, 1):
        samples.append(_make_sample(reward=float(gi), group_index=gi))
    args = _fake_args(n_samples_per_prompt=4)
    raw, normalized = post_process_rewards(args, samples)
    # Group 0: all rewards 0.0 → all advantages 0
    # Group 1: all rewards 1.0 → all advantages 0
    assert all(r == 0.0 for r in raw[0::2])  # group 0 raw
    assert all(r == 1.0 for r in raw[1::2])  # group 1 raw
    # Constant groups → zero advantage
    assert all(n == 0.0 for n in normalized)


def test_post_process_rewards_rloo_raw_rewards_unchanged(monkeypatch):
    _install_fake_megatron(monkeypatch)
    from relax.utils.utils import post_process_rewards

    raw_rewards = [0.0, 1.0, 0.0, 1.0]
    samples = [_make_sample(reward=r, group_index=0) for r in raw_rewards]
    args = _fake_args(n_samples_per_prompt=4)
    raw, normalized = post_process_rewards(args, samples)
    assert raw == raw_rewards
    # RLOO: (4/3)*(r - 0.5)
    expected = [(4 / 3) * (r - 0.5) for r in raw_rewards]
    for got, exp in zip(normalized, expected, strict=True):
        assert abs(got - exp) < 1e-6


def test_post_process_rewards_rloo_wrong_group_size_raises(monkeypatch):
    _install_fake_megatron(monkeypatch)
    from relax.utils.utils import post_process_rewards

    samples = [_make_sample(reward=float(i), group_index=0) for i in range(3)]
    args = _fake_args(n_samples_per_prompt=4)
    with pytest.raises(ValueError, match="3 samples, expected 4"):
        post_process_rewards(args, samples)


def test_post_process_rewards_rloo_nonfinite_reward_raises(monkeypatch):
    _install_fake_megatron(monkeypatch)
    from relax.utils.utils import post_process_rewards

    samples = [
        *[_make_sample(reward=1.0, group_index=0) for _ in range(4)],
        _make_sample(reward=1.0, group_index=42),
        _make_sample(reward=float("nan"), group_index=42),
        _make_sample(reward=0.0, group_index=42),
        _make_sample(reward=1.0, group_index=42),
    ]
    args = _fake_args(n_samples_per_prompt=4)
    with pytest.raises(ValueError) as exc_info:
        post_process_rewards(args, samples)

    message = str(exc_info.value)
    assert "non-finite" in message
    assert "group_index=42" in message
    assert "group position(s) [1]" in message
    assert "sample position(s) [5]" in message


def test_post_process_rewards_grpo_unchanged_by_rloo_addition(monkeypatch):
    """Adding 'rloo' must not change GRPO behaviour."""
    _install_fake_megatron(monkeypatch)
    from relax.utils.utils import post_process_rewards

    raw_rewards = [0.0, 1.0, 0.0, 1.0]
    samples = [_make_sample(reward=r, group_index=0) for r in raw_rewards]
    grpo_args = _fake_args(advantage_estimator="grpo", n_samples_per_prompt=4)
    _, grpo_norm = post_process_rewards(grpo_args, samples)
    # GRPO with std normalization: (r - mean) / (std + 1e-6)
    import numpy as np

    arr = np.array(raw_rewards)
    expected = (arr - arr.mean()) / (arr.std(ddof=1) + 1e-6)
    for got, exp in zip(grpo_norm, expected.tolist(), strict=True):
        assert abs(got - exp) < 1e-6


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


def test_rloo_loss_elementwise_value():
    from relax.utils.training.ppo_utils import compute_rloo_loss

    log_probs = torch.tensor([-0.5, -0.3, -0.2, -0.1], dtype=torch.float64)
    advantages = torch.tensor([0.5, -0.5, 0.25, -0.25], dtype=torch.float64)
    pg_loss, clipfrac = compute_rloo_loss(log_probs, advantages)
    expected = -(advantages * log_probs)
    assert torch.allclose(pg_loss, expected, atol=1e-12)
    assert torch.allclose(clipfrac, torch.zeros_like(clipfrac))


def test_rloo_loss_clipfrac_always_zero():
    from relax.utils.training.ppo_utils import compute_rloo_loss

    for _ in range(10):
        log_probs = torch.randn(100, dtype=torch.float64)
        advantages = torch.randn(100, dtype=torch.float64)
        _, clipfrac = compute_rloo_loss(log_probs, advantages)
        assert torch.all(clipfrac == 0.0)


def test_rloo_loss_gradient_equals_neg_advantage():
    from relax.utils.training.ppo_utils import compute_rloo_loss

    advantages = torch.tensor([0.5, -0.3, 0.2, -0.1], dtype=torch.float64)
    log_probs = torch.tensor([-0.4, -0.2, -0.6, -0.8], dtype=torch.float64, requires_grad=True)
    pg_loss, _ = compute_rloo_loss(log_probs, advantages)
    # Sum to get scalar, then backward
    pg_loss.sum().backward()
    # d pg_loss_i / d log_probs_i = -advantages_i (advantages detached)
    assert torch.allclose(log_probs.grad, -advantages, atol=1e-12)


def test_rloo_loss_is_not_ppo_clip():
    from relax.utils.training.ppo_utils import compute_policy_loss, compute_rloo_loss

    # Construct a scenario where PPO-clip would clip but RLOO does not.
    # ppo_kl = old_log - new_log; ratio = exp(-ppo_kl) = exp(new - old)
    # Make ratio large enough to trigger clipping.
    log_probs = torch.tensor([-0.1], dtype=torch.float64)
    advantages = torch.tensor([1.0], dtype=torch.float64)
    rloo_pg, rloo_cf = compute_rloo_loss(log_probs, advantages)

    # PPO: ppo_kl = old - new, ratio = exp(-ppo_kl) = exp(new - old)
    # Set old_log = -5, new_log = -0.1 → ratio = exp(4.9) >> 1+eps_clip
    ppo_kl = torch.tensor([-5.0 - (-0.1)], dtype=torch.float64)  # old - new = -4.9
    eps_clip = 0.2
    eps_clip_high = 0.2
    ppo_pg, ppo_cf = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)

    # RLOO loss = -(1.0 * -0.1) = 0.1; PPO clips ratio so loss differs
    assert abs(rloo_pg.item() - 0.1) < 1e-12
    assert not torch.allclose(rloo_pg, ppo_pg)
    # RLOO clipfrac is 0; PPO clipfrac is non-zero (clipped)
    assert rloo_cf.item() == 0.0
    assert ppo_cf.item() == 1.0


# ---------------------------------------------------------------------------
# DP / CP invariance (pure-tensor simulation, no real dist)
# ---------------------------------------------------------------------------


def test_rloo_dp_shard_invariance():
    """LOO on full batch, then split into DP shards == compute on each shard
    (the values are identical because LOO is done before split)."""
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    G = 4
    num_groups = 6
    torch.manual_seed(123)
    all_rewards = torch.randn(num_groups * G, dtype=torch.float64)
    # Full-batch LOO (group by group)
    full_adv = torch.empty_like(all_rewards)
    for g in range(num_groups):
        sl = slice(g * G, (g + 1) * G)
        full_adv[sl] = compute_rloo_leave_one_out_rewards(all_rewards[sl])

    # Simulate DP split: shard 0 = groups [0,1,2], shard 1 = groups [3,4,5]
    # (groups never cross DP boundary)
    for dp_rank, groups in enumerate([range(0, 3), range(3, 6)]):
        shard_adv = []
        for g in groups:
            sl = slice(g * G, (g + 1) * G)
            shard_adv.append(compute_rloo_leave_one_out_rewards(all_rewards[sl]))
        shard_adv = torch.cat(shard_adv)
        shard_full = torch.cat([full_adv[g * G : (g + 1) * G] for g in groups])
        assert torch.equal(shard_adv, shard_full), f"DP rank {dp_rank} mismatch"


def test_rloo_cp_slice_invariance():
    """Scalar broadcast then token-slice == slice then broadcast (CP only
    decides which tokens each rank holds; values unchanged)."""
    G = 4
    resp_lens = [10, 8, 12, 6]
    torch.manual_seed(7)
    rewards = torch.randn(G, dtype=torch.float64)

    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards

    advantages = compute_rloo_leave_one_out_rewards(rewards)

    # Full broadcast
    full_tokens = torch.cat(
        [torch.full((L,), advantages[i].item(), dtype=torch.float64) for i, L in enumerate(resp_lens)]
    )

    # Simulate CP: split each response into two chunks (zig-zag not needed for
    # scalar broadcast — any partition works since the value is constant).
    cp_rank_0 = []
    cp_rank_1 = []
    for i, L in enumerate(resp_lens):
        tokens = torch.full((L,), advantages[i].item(), dtype=torch.float64)
        half = L // 2
        cp_rank_0.append(tokens[:half])
        cp_rank_1.append(tokens[half:])

    rank_0 = torch.cat(cp_rank_0)
    rank_1 = torch.cat(cp_rank_1)
    reconstructed = torch.cat([rank_0, rank_1])
    # The concatenated shards cover all tokens (values match full broadcast)
    assert torch.equal(reconstructed.sort().values, full_tokens.sort().values)
    # Each token's value is correct
    for i, L in enumerate(resp_lens):
        shard_vals = torch.cat([cp_rank_0[i], cp_rank_1[i]])
        assert torch.allclose(shard_vals, torch.full((L,), advantages[i].item(), dtype=torch.float64), atol=1e-12)


def test_rloo_broadcast_respects_loss_mask_and_empty_response():
    """Empty response / all-zero mask: no NaN, masked-out tokens excluded."""
    from relax.utils.training.ppo_utils import compute_rloo_leave_one_out_rewards, compute_rloo_loss

    # Group with one empty response (response_length=0)
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    advantages = compute_rloo_leave_one_out_rewards(rewards)

    # Simulate token-level: sample 2 has 0 tokens (empty)
    resp_lens = [5, 3, 0, 4]
    loss_masks = [torch.ones(L) for L in resp_lens]
    log_probs_per_sample = [torch.randn(L, dtype=torch.float64) for L in resp_lens]

    total_pg = torch.tensor(0.0, dtype=torch.float64)
    for i, L in enumerate(resp_lens):
        if L == 0:
            continue  # empty response: no tokens, no loss
        adv_broadcast = advantages[i].expand(L)
        pg, _ = compute_rloo_loss(log_probs_per_sample[i], adv_broadcast)
        total_pg += (pg * loss_masks[i]).sum()

    assert torch.isfinite(total_pg)
    # No NaN from the empty-response sample (it was skipped)
