# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Scheduling geometry for whole-pack Qwen3.5 training."""

GLOBAL_BATCH_SIZE = 1600
EXPERT_PARALLEL_SIZE = 8


def resolve_tp(dp_size: int, tp_size: int | None) -> int:
    """Preserve the old DP32/TP2 API default; production passes TP
    explicitly."""
    return (2 if dp_size == 32 else 4) if tp_size is None else tp_size


def sampler_padding(
    n_samples_per_prompt: int, dp_size: int, capacity: int, pad_size: int | None, tp_size: int | None = None
) -> int:
    tp_size = resolve_tp(dp_size, tp_size)
    if n_samples_per_prompt != 1 or tp_size not in (2, 4):
        raise ValueError("wholepack requires SFT n1 and TP2/4")
    if dp_size <= 0 or dp_size % (EXPERT_PARALLEL_SIZE // tp_size):
        raise ValueError("wholepack DP must form complete EP8 domains")
    expected = tp_size * 1024 if pad_size is None else pad_size
    if expected <= 0 or expected % tp_size or capacity <= 0 or capacity % expected:
        raise ValueError("wholepack padding/capacity mismatch")
    return expected


def enabled(args) -> bool:
    return bool(getattr(args, "sft_wholepack", False))


def config_contract(args) -> tuple[int, int]:
    """Same existing model/data arguments on Controller and train workers."""
    tp = int(args.tensor_model_parallel_size)
    cp = int(args.context_parallel_size)
    pp = int(args.pipeline_model_parallel_size)
    pad_size = tp * int(args.data_pad_size_multiplier)
    capacity = int(args.max_tokens_per_gpu)
    if tp not in (2, 4) or (cp, pp, args.n_samples_per_prompt) != (1, 1, 1):
        raise ValueError("wholepack requires TP2/4 CP1/PP1/n1")
    if args.expert_model_parallel_size != EXPERT_PARALLEL_SIZE or args.expert_tensor_parallel_size != 1:
        raise ValueError("wholepack requires EP8/ETP1")
    if args.qkv_format != "thd" or not args.use_dynamic_batch_size:
        raise ValueError("wholepack requires dynamic THD binpack")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise ValueError("wholepack does not support VPP")
    if getattr(args, "sft_async_prepack", False) or getattr(args, "dynamic_context_parallel", False):
        raise ValueError("wholepack draft does not support async prepack or dynamicCP")
    if args.global_batch_size <= 0:
        raise ValueError("wholepack requires positive GBS")
    sampler_padding(args.n_samples_per_prompt, EXPERT_PARALLEL_SIZE // tp, capacity, pad_size, tp)
    if getattr(args, "rollout_batch_size", args.global_batch_size) != args.global_batch_size:
        raise ValueError("wholepack requires one optimizer step per rollout")
    return capacity, pad_size
