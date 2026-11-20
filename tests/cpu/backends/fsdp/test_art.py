# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""ART window, normalization and two-update CPU acceptance, without Ray."""

import math
from dataclasses import replace

import pytest
import torch

from relax.backends.fsdp.art import ARTTrainingStep, art_backward, art_update_windows


def _steps(rows):
    return [
        ARTTrainingStep(
            state=torch.zeros(len(rows), 2),
            old_velocity=torch.zeros(len(rows), 2),
            timestep=torch.full((len(rows),), sigma * 1000),
            sigma=torch.full((len(rows),), sigma),
            ode_velocity_coefficient=torch.full((len(rows),), -0.1),
        )
        for sigma in (0.5, 0.25)
    ]


def test_explicit_prompt_windows_match_independent_row_oracle():
    result = art_update_windows([0] * 4 + [1] * 4, order=[6, 1, 4, 3, 0, 7, 2, 5])
    assert result == (((6, 1, 4, 3),), ((0, 7, 2, 5),))


@pytest.mark.parametrize("world", [1, 2, 8])
def test_global_windows_balance_ranks_without_replication_or_rng_mutation(world):
    groups = [0] * 8 + [1] * 8
    before = torch.random.get_rng_state().clone()
    plans = art_update_windows(groups, world_size=world, seed=13, rollout_id=2)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert plans == art_update_windows(groups, world_size=world, seed=13, rollout_id=2)
    seen = []
    for window in plans:
        assert {len(rank) for rank in window} == {8 // world}
        rows = [row for rank in window for row in rank]
        assert [sum(groups[row] == group for row in rows) for group in (0, 1)] == [4, 4]
        seen.extend(rows)
    assert sorted(seen) == list(range(16))
    single = art_update_windows(groups, seed=13, rollout_id=2)
    assert all({row for rank in window for row in rank} == set(single[i][0]) for i, window in enumerate(plans))


def test_invalid_group_and_rank_geometry_fail_instead_of_replicating():
    with pytest.raises(ValueError, match="even K"):
        art_update_windows([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="every rank"):
        art_update_windows([0] * 8, world_size=8)
    with pytest.raises(ValueError, match="permutation"):
        art_update_windows([0] * 4, order=[0, 1, 2, 2])


@pytest.mark.parametrize("learning_rate", [1e-3, 0.05])
def test_two_adamw_updates_match_scalar_oracle_including_all_masked_second_step(learning_rate):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=learning_rate, weight_decay=0.01, foreach=False)
    direction = torch.tensor([[1.0, -1.0]])
    groups = [0] * 4 + [1] * 4
    plans = art_update_windows(groups, order=list(range(8)))
    expected_w = first_moment = second_moment = 0.0
    for update, window in enumerate(plans, 1):
        rows = window[0]
        magnitudes = torch.tensor([float(groups[row] == 0) for row in rows])
        optimizer.zero_grad(set_to_none=True)
        metrics = art_backward(
            lambda state, time: parameter * direction.expand_as(state),
            _steps(rows),
            guidance_direction=direction.expand(len(rows), -1),
            guidance_magnitude=magnitudes,
            global_item_count=8,
        )
        expected_loss = expected_grad = 0.0
        for row in rows:
            for sigma in (0.5, 0.25):
                if groups[row] != 0:
                    continue
                coefficient = -0.1 * (1 + 0.8**2 / 2)
                variance = 0.8**2 * sigma / (1 - sigma) * 0.1
                energy = coefficient**2 * expected_w**2 / (2 * variance)
                alignment = expected_w * (-1 - expected_w)
                if energy >= 1e-5 and alignment > 0:
                    continue
                expected_loss += sigma * (1 + expected_w) ** 2 / 8
                expected_grad += 2 * sigma * (1 + expected_w) / 8
        torch.testing.assert_close(metrics["loss_sum"] / 8, torch.tensor(expected_loss))
        torch.testing.assert_close(parameter.grad, torch.tensor(expected_grad))
        first_moment = 0.9 * first_moment + 0.1 * expected_grad
        second_moment = 0.999 * second_moment + 0.001 * expected_grad**2
        expected_w *= 1 - learning_rate * 0.01
        expected_w -= (
            learning_rate
            * (first_moment / (1 - 0.9**update))
            / (math.sqrt(second_moment / (1 - 0.999**update)) + 1e-8)
        )
        optimizer.step()
        torch.testing.assert_close(parameter.detach(), torch.tensor(expected_w))
        state = optimizer.state[parameter]
        torch.testing.assert_close(state["exp_avg"], torch.tensor(first_moment))
        torch.testing.assert_close(state["exp_avg_sq"], torch.tensor(second_moment))
        assert state["step"].item() == update
        if update == 2 and learning_rate == 0.05:
            assert metrics["kept_items"].item() == 0
            assert parameter.grad.item() == 0
            assert parameter.item() < -learning_rate  # Adam momentum still performs the second update.


def test_rank_averaged_microbatch_grads_equal_global_oracle_with_uneven_validity():
    direction = torch.tensor([[1.0, -1.0]])
    all_valid = torch.tensor([False, True, False, True])
    oracle = torch.nn.Parameter(torch.tensor(0.0))
    art_backward(
        lambda state, time: oracle * direction.expand_as(state),
        _steps(range(4)),
        guidance_direction=direction.expand(4, -1),
        guidance_magnitude=torch.ones(4),
        global_item_count=8,
        sample_valid_mask=all_valid,
    )
    grads, calls = [], []
    for rank in range(2):
        parameter = torch.nn.Parameter(torch.tensor(0.0))
        calls.append(0)

        def predict(state, time):
            calls[-1] += 1
            return parameter * direction.expand_as(state)

        for row in range(rank, 4, 2):
            art_backward(
                predict,
                _steps([row]),
                guidance_direction=direction,
                guidance_magnitude=torch.ones(1),
                global_item_count=8,
                data_parallel_world_size=2,
                sample_valid_mask=all_valid[row : row + 1],
            )
        grads.append(parameter.grad.detach())
    torch.testing.assert_close(torch.stack(grads).mean(), oracle.grad)
    assert calls == [4, 4]  # An entirely invalid rank still executes every backward slot.


def test_bad_cache_is_rejected_before_any_backward():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    steps = _steps([0])
    malformed = replace(steps[1], state=torch.zeros(2, 2))
    with pytest.raises(ValueError, match="identical shapes"):
        art_backward(
            lambda state, time: parameter * state,
            [steps[0], malformed],
            guidance_direction=torch.ones(1, 2),
            guidance_magnitude=torch.ones(1),
            global_item_count=2,
        )
    assert parameter.grad is None
