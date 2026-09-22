# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.engine.sft.runtime import resolve_sft_split_indices


def test_resolve_sft_split_indices_is_deterministic_and_disjoint():
    train_indices, eval_indices = resolve_sft_split_indices(100, 0.2, seed=42)
    same_train, same_eval = resolve_sft_split_indices(100, 0.2, seed=42)
    other_train, other_eval = resolve_sft_split_indices(100, 0.2, seed=43)

    assert (train_indices, eval_indices) == (same_train, same_eval)
    assert (train_indices, eval_indices) != (other_train, other_eval)
    assert len(train_indices) == 80
    assert len(eval_indices) == 20
    assert set(train_indices).isdisjoint(eval_indices)
    assert set(train_indices) | set(eval_indices) == set(range(100))


def test_resolve_sft_split_indices_clamps_eval_to_leave_one_train_row():
    train_indices, eval_indices = resolve_sft_split_indices(4, 99, seed=7)

    assert len(train_indices) == 1
    assert len(eval_indices) == 3


def test_resolve_sft_split_indices_matches_ms_swift_rng_order():
    train_indices, eval_indices = resolve_sft_split_indices(100, 0.2, seed=42)

    assert train_indices[:8] == (49, 48, 46, 56, 12, 17, 0, 50)
    assert eval_indices[:8] == (91, 52, 40, 13, 32, 26, 65, 1)
