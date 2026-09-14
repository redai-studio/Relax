# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace

import pytest

from examples.osworld_agentic.train_fixed_tasks import configure_fixed_tasks


def _args(**overrides):
    values = dict(
        use_agentic_rollout=True,
        rollout_global_dataset=True,
        agentic_prelaunch=False,
        fully_async=False,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        colocate=True,
        max_staleness=0,
        dynamic_sampling_filter_path=None,
        buffer_filter_path=None,
        rollout_data_postprocess_path=None,
        over_sampling_batch_size=None,
        rollout_batch_size=74,
        n_samples_per_prompt=8,
        global_batch_size=592,
        use_streaming_dataset=True,
        rollout_shuffle=True,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("oversampling", [None, 74])
def test_configure_selects_fixed_recipe_without_changing_training_batch(oversampling):
    args = _args(over_sampling_batch_size=oversampling)

    assert configure_fixed_tasks(args) is args

    assert args.use_streaming_dataset is False
    assert args.rollout_shuffle is False
    assert args.data_source_path == "examples.osworld_agentic.fixed_task_data_source.FixedTaskDataSource"
    assert args.rollout_function_path == "examples.osworld_agentic.fixed_task_rollout.generate_rollout"
    assert args.custom_reward_post_process_path == "examples.osworld_agentic.fixed_task_rollout.post_process_rewards"
    assert (args.rollout_batch_size, args.n_samples_per_prompt, args.global_batch_size) == (74, 8, 592)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("use_agentic_rollout", False),
        ("rollout_global_dataset", False),
        ("agentic_prelaunch", True),
        ("fully_async", True),
        ("partial_rollout", True),
        ("use_dynamic_global_batch_size", True),
        ("colocate", False),
        ("max_staleness", 1),
        ("max_staleness", -1),
        ("dynamic_sampling_filter_path", "some.filter"),
        ("buffer_filter_path", "some.filter"),
        ("rollout_data_postprocess_path", "some.postprocess"),
        ("over_sampling_batch_size", 75),
        ("over_sampling_batch_size", 73),
        ("global_batch_size", 296),
        ("global_batch_size", 1184),
    ],
)
def test_configure_rejects_unsupported_modes_before_mutating_args(name, value):
    args = _args(**{name: value})
    original = vars(args).copy()

    with pytest.raises(ValueError):
        configure_fixed_tasks(args)

    assert vars(args) == original


def test_configure_native_three_batches_preserves_complete_rollout_and_batch_sizes():
    args = _args(
        rollout_batch_size=12,
        n_samples_per_prompt=8,
        global_batch_size=32,
        num_steps_per_rollout=3,
        over_sampling_batch_size=12,
    )

    assert configure_fixed_tasks(args, train_batches_per_rollout=3) is args

    assert (args.rollout_batch_size, args.n_samples_per_prompt, args.global_batch_size) == (12, 8, 32)
    assert args.num_steps_per_rollout == 3
    assert args.use_streaming_dataset is False
    assert args.rollout_shuffle is False
    assert args.data_source_path == "examples.osworld_agentic.fixed_task_data_source.FixedTaskDataSource"
    assert args.rollout_function_path == "examples.osworld_agentic.fixed_task_rollout.generate_rollout"


def test_three_batches_require_explicit_profile_opt_in():
    args = _args(rollout_batch_size=12, global_batch_size=32, num_steps_per_rollout=3)
    original = vars(args).copy()

    with pytest.raises(ValueError, match="exactly cover"):
        configure_fixed_tasks(args)

    assert vars(args) == original


@pytest.mark.parametrize("configured_batches", [0, -1, True, 3.0, "3"])
def test_configure_rejects_invalid_training_batch_count(configured_batches):
    args = _args()
    original = vars(args).copy()

    with pytest.raises(ValueError, match="positive integer"):
        configure_fixed_tasks(args, train_batches_per_rollout=configured_batches)

    assert vars(args) == original


@pytest.mark.parametrize("num_steps", [None, 1, 2, 4, True, 3.0])
def test_three_batch_profile_requires_matching_native_optimizer_steps(num_steps):
    args = _args(rollout_batch_size=12, global_batch_size=32, num_steps_per_rollout=num_steps)
    original = vars(args).copy()

    with pytest.raises(ValueError, match="num_steps_per_rollout"):
        configure_fixed_tasks(args, train_batches_per_rollout=3)

    assert vars(args) == original


def test_default_profile_rejects_conflicting_explicit_optimizer_steps():
    args = _args(num_steps_per_rollout=3)
    original = vars(args).copy()

    with pytest.raises(ValueError, match="num_steps_per_rollout"):
        configure_fixed_tasks(args)

    assert vars(args) == original


def test_three_batches_must_exactly_cover_the_rollout():
    args = _args(rollout_batch_size=12, global_batch_size=40, num_steps_per_rollout=3)
    original = vars(args).copy()

    with pytest.raises(ValueError, match="exactly cover"):
        configure_fixed_tasks(args, train_batches_per_rollout=3)

    assert vars(args) == original


def test_minibatches_cannot_split_grpo_groups():
    # The arithmetic covers all 24 sessions, but each batch of 12 would split
    # one eight-session task group between different optimizer updates.
    args = _args(rollout_batch_size=3, global_batch_size=12, num_steps_per_rollout=2)
    original = vars(args).copy()

    with pytest.raises(ValueError, match="complete GRPO groups"):
        configure_fixed_tasks(args, train_batches_per_rollout=2)

    assert vars(args) == original
