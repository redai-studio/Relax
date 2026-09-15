# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Configure complete task-set rollouts for Relax's native training loop."""

from argparse import Namespace


def configure_training(args: Namespace) -> Namespace:
    """Validate sampling and update boundaries, then select the OSWorld
    hooks."""
    configured_updates = getattr(args, "num_steps_per_rollout", None)
    updates = 1 if configured_updates is None else configured_updates
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("num_steps_per_rollout must be a positive integer.")
    if not args.use_agentic_rollout or not args.rollout_global_dataset:
        raise ValueError("OSWorld task-set training requires agentic rollout with a global dataset.")
    for name in ("agentic_prelaunch", "fully_async", "partial_rollout", "use_dynamic_global_batch_size"):
        if getattr(args, name, False):
            raise ValueError(f"OSWorld task-set training does not support {name}.")
    if not args.colocate or args.max_staleness != 0:
        raise ValueError("OSWorld task-set training requires --colocate and --max-staleness 0.")
    for name in ("dynamic_sampling_filter_path", "buffer_filter_path", "rollout_data_postprocess_path"):
        if getattr(args, name, None) is not None:
            raise ValueError(f"OSWorld task-set training does not support {name}.")
    if getattr(args, "over_sampling_batch_size", None) not in (None, args.rollout_batch_size):
        raise ValueError("OSWorld task-set training does not support oversampling.")
    if args.global_batch_size * updates != args.rollout_batch_size * args.n_samples_per_prompt:
        raise ValueError("OSWorld training batches must exactly cover one complete rollout.")
    if updates > 1:
        if args.global_batch_size % args.n_samples_per_prompt:
            raise ValueError("Each OSWorld training batch must contain complete GRPO groups.")
        for name in (
            "eval_interval",
            "eval_prompt_data",
            "eval_size",
            "custom_megatron_init_path",
            "custom_megatron_before_train_step_hook_path",
        ):
            if getattr(args, name, None) is not None:
                raise ValueError(f"Multiple optimizer updates require {name}=None.")
        if not args.skip_eval_before_train:
            raise ValueError("Multiple optimizer updates require evaluation before training to be disabled.")
        if getattr(args, "true_on_policy_mode", False) or args.advantage_estimator != "grpo":
            raise ValueError("Multiple optimizer updates require the ordinary clipped GRPO training path.")
    args.use_streaming_dataset = False
    args.rollout_shuffle = False
    args.data_source_path = "examples.osworld_agentic.training.data_source.OSWorldTaskDataSource"
    args.rollout_function_path = "examples.osworld_agentic.training.rollout.generate_rollout"
    args.custom_reward_post_process_path = "examples.osworld_agentic.training.rollout.post_process_rewards"
    return args


if __name__ == "__main__":
    from relax.entrypoints.train import main
    from relax.utils.arguments import parse_args

    main(configure_training(parse_args()))
