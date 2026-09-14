# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""OSWorld fixed-task diagnostic using three native optimizer batches per
rollout."""

from argparse import Namespace

from examples.osworld_agentic.train_fixed_tasks import configure_fixed_tasks


def configure_native_minibatches(args: Namespace) -> Namespace:
    if (args.rollout_batch_size, args.n_samples_per_prompt, args.global_batch_size) != (12, 8, 32):
        raise ValueError("This diagnostic requires 12 tasks x 8 samples and global_batch_size=32.")
    if args.num_steps_per_rollout != 3:
        raise ValueError("This diagnostic requires --num-steps-per-rollout 3.")
    for key in (
        "eval_interval",
        "eval_prompt_data",
        "eval_size",
        "custom_megatron_init_path",
        "custom_megatron_before_train_step_hook_path",
    ):
        if getattr(args, key, None) is not None:
            raise ValueError(f"The native minibatch diagnostic requires {key}=None.")
    if not args.skip_eval_before_train:
        raise ValueError("This diagnostic disables evaluation before training.")
    if getattr(args, "true_on_policy_mode", False) or args.advantage_estimator != "grpo":
        raise ValueError("Three updates require the ordinary clipped GRPO training path.")
    return configure_fixed_tasks(args, train_batches_per_rollout=3)


if __name__ == "__main__":
    from relax.entrypoints.train import main
    from relax.utils.arguments import parse_args

    main(configure_native_minibatches(parse_args()))
