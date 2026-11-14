# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Freeze adaptive layer recomputation masks across forward and backward
replay."""

MASK_KEY = "_qwen_recompute_mask"
_state = {}


def checkpoint_mask(max_tokens: int) -> tuple[int, ...]:
    gdn = tuple(i for i in range(40) if (i + 1) % 4)
    return gdn[:16] if max_tokens <= 262144 else tuple(sorted((*gdn, 3, 7)))


def _install_decoder(model: object) -> None:
    while hasattr(model, "module"):
        model = model.module
    decoder = getattr(model, "language_model", model).decoder
    assert len(decoder.layers) == 40
    config = decoder.config
    assert config.recompute_granularity == "full" and config.recompute_method == "block"
    assert not config.fp8 and not config.fp4
    actual_gdn = tuple(
        i
        for i, layer in enumerate(decoder.layers)
        if any(
            "GatedDeltaNet" in type(m).__name__ or type(m).__name__ == "GDN2" for m in layer.self_attention.modules()
        )
    )
    assert actual_gdn == tuple(i for i in range(40) if (i + 1) % 4), actual_gdn


def before_step(
    args: object, rollout_id: int, step_id: int, model: list, optimizer: object, opt_param_scheduler: object
) -> None:

    assert (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.context_parallel_size,
        args.expert_model_parallel_size,
        args.expert_tensor_parallel_size,
    ) == (4, 1, 1, 8, 1)
    assert args.optimizer_offload_fraction == 0 and args.data_pad_size_multiplier == 1024
    if not _state:
        for chunk in model:
            _install_decoder(chunk)
        _state["installed"] = True
    _state.update(iterator=None)


def attach_mask(args, iterator, batch):
    """Attach per-wave CPU metadata in the foreground training data path."""
    from .recipe import enabled

    if not enabled(args):
        return
    import torch
    import torch.distributed as dist
    from megatron.core import mpu

    wave = iterator.offset - 1
    if _state.get("iterator") is not iterator:
        lengths = iterator.rollout_data["total_lengths"]
        local = [
            ((sum(lengths[i] for i in indices) + 4095) // 4096) * 4096 for indices in iterator.micro_batch_indices
        ]
        maxima = torch.tensor(local, dtype=torch.int64, device="cpu")
        dist.all_reduce(
            maxima, op=dist.ReduceOp.MAX, group=mpu.get_data_parallel_group_gloo(with_context_parallel=True)
        )
        _state.update(iterator=iterator, maxima=maxima.tolist())
    maximum = _state["maxima"][wave]
    assert batch["tokens"].shape[-1] <= maximum
    setattr(batch["packed_seq_params"], MASK_KEY, checkpoint_mask(maximum))
