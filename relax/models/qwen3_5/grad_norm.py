# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Keep a pre-update finite check while testing bucket-check overhead."""

from relax.utils.training.nonfinite_grad import NonFiniteGradientNormError


def finite_norm(value: object) -> float:
    import math

    import torch

    # Replace the existing per-child threshold readbacks with one host value.
    result = value.item() if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(result):
        raise NonFiniteGradientNormError("Non-finite global gradient norm before optimizer update")
    return result


def install_norm_guard(args: object, optimizer: object) -> None:
    import torch.distributed as dist
    from megatron.core.optimizer.optimizer import (
        ChainedOptimizer,
        _get_param_grad_norm_group,
        _is_separate_grad_norm_group,
    )

    assert isinstance(optimizer, ChainedOptimizer), type(optimizer)
    assert args.bf16 and not args.fp16 and not args.check_for_nan_in_loss_and_grad
    world_ranks = list(range(dist.get_world_size(group=dist.group.WORLD)))
    stats_groups = []
    for child in optimizer.chained_optimizers:
        if getattr(child, "is_stub_optimizer", False):
            continue
        assert child.config.clip_grad > 0
        assert not getattr(child, "grad_scaler", None)
        assert all(not _is_separate_grad_norm_group(_get_param_grad_norm_group(p)) for p in child.get_parameters()), (
            "Separate gradient-norm groups need independent guards"
        )
        group = child.get_grad_stats_parallel_group()
        ranks = world_ranks if group is None else dist.get_process_group_ranks(group)
        assert ranks == world_ranks, "Global finite guard requires each norm stats group to cover WORLD"
        stats_groups.append(dict(optimizer=type(child).__name__, ranks=ranks))
    optimizer._relax_qwen35_optimized = True
    optimizer._qwen_norm_guarded = True
    optimizer._qwen_norm_stats_groups = stats_groups
