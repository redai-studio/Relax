# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""OSWorld entry point for one complete task set per rollout step."""


def configure_fixed_tasks(args, *, train_batches_per_rollout: int = 1):
    """Select recipe hooks after the framework has parsed and validated its
    args."""
    if (
        isinstance(train_batches_per_rollout, bool)
        or not isinstance(train_batches_per_rollout, int)
        or train_batches_per_rollout <= 0
    ):
        raise ValueError("train_batches_per_rollout must be a positive integer.")
    if not args.use_agentic_rollout or not args.rollout_global_dataset:
        raise ValueError("Fixed OSWorld tasks require agentic rollout with a global dataset.")
    for name in ("agentic_prelaunch", "fully_async", "partial_rollout", "use_dynamic_global_batch_size"):
        if getattr(args, name, False):
            raise ValueError(f"Fixed OSWorld tasks do not support {name}.")
    if not args.colocate or args.max_staleness != 0:
        raise ValueError("Fixed OSWorld tasks require --colocate and --max-staleness 0.")
    for name in ("dynamic_sampling_filter_path", "buffer_filter_path", "rollout_data_postprocess_path"):
        if getattr(args, name, None) is not None:
            raise ValueError(f"Fixed OSWorld tasks do not support {name}.")
    if getattr(args, "over_sampling_batch_size", None) not in (None, args.rollout_batch_size):
        raise ValueError("Fixed OSWorld tasks do not support oversampling.")
    if args.global_batch_size * train_batches_per_rollout != args.rollout_batch_size * args.n_samples_per_prompt:
        raise ValueError("Fixed OSWorld training batches must exactly cover one complete rollout.")
    num_steps_per_rollout = getattr(args, "num_steps_per_rollout", None)
    if num_steps_per_rollout is not None and (
        isinstance(num_steps_per_rollout, bool)
        or not isinstance(num_steps_per_rollout, int)
        or num_steps_per_rollout != train_batches_per_rollout
    ):
        raise ValueError("num_steps_per_rollout must match train_batches_per_rollout.")
    if train_batches_per_rollout > 1:
        if num_steps_per_rollout is None:
            raise ValueError("Multiple fixed-task training batches require explicit num_steps_per_rollout.")
        if args.global_batch_size % args.n_samples_per_prompt:
            raise ValueError("Each fixed-task training batch must contain complete GRPO groups.")
    args.use_streaming_dataset = False
    args.rollout_shuffle = False
    args.data_source_path = "examples.osworld_agentic.fixed_task_data_source.FixedTaskDataSource"
    args.rollout_function_path = "examples.osworld_agentic.fixed_task_rollout.generate_rollout"
    args.custom_reward_post_process_path = "examples.osworld_agentic.fixed_task_rollout.post_process_rewards"
    return args


if __name__ == "__main__":
    from relax.entrypoints.train import main
    from relax.utils.arguments import parse_args

    main(configure_fixed_tasks(parse_args()))
