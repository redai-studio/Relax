# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""CPU checks for native fixed-task minibatches with evaluation disabled."""

import unittest
from argparse import Namespace

from examples.osworld_agentic.train_fixed12_minibatch3 import configure_native_minibatches
from examples.osworld_agentic.train_fixed_tasks import configure_fixed_tasks


def make_args(**overrides):
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
        rollout_batch_size=12,
        n_samples_per_prompt=8,
        global_batch_size=32,
        num_steps_per_rollout=3,
        use_streaming_dataset=True,
        rollout_shuffle=True,
        eval_interval=None,
        eval_prompt_data=None,
        eval_size=None,
        skip_eval_before_train=True,
        custom_megatron_init_path=None,
        custom_megatron_before_train_step_hook_path=None,
        true_on_policy_mode=False,
        advantage_estimator="grpo",
    )
    values.update(overrides)
    return Namespace(**values)


class TestNativeMinibatchProfile(unittest.TestCase):
    def test_native_twelve_task_profile_uses_only_fixed_rollout_hooks(self):
        args = make_args()

        self.assertIs(configure_native_minibatches(args), args)

        self.assertEqual((args.rollout_batch_size, args.n_samples_per_prompt), (12, 8))
        self.assertEqual((args.global_batch_size, args.num_steps_per_rollout), (32, 3))
        self.assertIsNone(args.custom_megatron_init_path)
        self.assertIsNone(args.custom_megatron_before_train_step_hook_path)
        self.assertEqual(args.data_source_path, "examples.osworld_agentic.fixed_task_data_source.FixedTaskDataSource")
        self.assertFalse(args.rollout_shuffle)
        self.assertFalse(args.use_streaming_dataset)

    def test_evaluation_or_replay_hooks_are_rejected_before_changes(self):
        for name, value in (
            ("eval_interval", 1),
            ("eval_prompt_data", "eval.parquet"),
            ("eval_size", 48),
            ("skip_eval_before_train", False),
            ("custom_megatron_init_path", "example.replay.install"),
            ("custom_megatron_before_train_step_hook_path", "example.replay.before_train"),
        ):
            args = make_args(**{name: value})
            original = vars(args).copy()
            with self.subTest(name=name), self.assertRaises(ValueError):
                configure_native_minibatches(args)
            self.assertEqual(vars(args), original)

    def test_wrong_batch_plan_or_training_path_is_rejected(self):
        for overrides in (
            {"rollout_batch_size": 10},
            {"global_batch_size": 96},
            {"num_steps_per_rollout": 1},
            {"num_steps_per_rollout": 2},
            {"true_on_policy_mode": True},
            {"advantage_estimator": "reinforce"},
        ):
            args = make_args(**overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                configure_native_minibatches(args)

    def test_existing_fixed_task_default_still_accepts_one_batch(self):
        args = make_args(rollout_batch_size=74, global_batch_size=592, num_steps_per_rollout=1)
        self.assertIs(configure_fixed_tasks(args), args)
        self.assertEqual((args.global_batch_size, args.num_steps_per_rollout), (592, 1))

    def test_three_batches_need_opt_in_and_matching_native_count(self):
        args = make_args()
        with self.assertRaises(ValueError):
            configure_fixed_tasks(args)
        for num_steps in (None, 1, 2, 4, True, 3.0):
            with self.subTest(num_steps=num_steps), self.assertRaises(ValueError):
                configure_fixed_tasks(make_args(num_steps_per_rollout=num_steps), train_batches_per_rollout=3)

    def test_incomplete_or_split_groups_are_rejected(self):
        cases = (
            ({"global_batch_size": 40}, 3),
            ({"rollout_batch_size": 3, "global_batch_size": 12, "num_steps_per_rollout": 2}, 2),
        )
        for overrides, batches in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                configure_fixed_tasks(make_args(**overrides), train_batches_per_rollout=batches)


if __name__ == "__main__":
    unittest.main()
