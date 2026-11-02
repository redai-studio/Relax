# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit scheduling policy for the reviewed 64-GPU Qwen3.5 recipe."""

WORLD_SIZE = 64
GLOBAL_BATCH_SIZE = 1600
EXPERT_PARALLEL_SIZE = 8
SAMPLER_GEOMETRIES = {(32, 131072), (16, 131072), (16, 262144)}


def sampler_padding(n_samples_per_prompt: int, dp_size: int, capacity: int, pad_size: int | None) -> int:
    if n_samples_per_prompt != 1 or (dp_size, capacity) not in SAMPLER_GEOMETRIES:
        raise ValueError("wholepack requires SFT n1 and reviewed DP16/32 capacity")
    expected = (WORLD_SIZE // dp_size) * 1024
    if pad_size is not None and pad_size != expected:
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
    if tp not in (2, 4) or (cp, pp, args.global_batch_size, args.n_samples_per_prompt) != (1, 1, GLOBAL_BATCH_SIZE, 1):
        raise ValueError("wholepack requires TP2/4 CP1/PP1/GBS1600/n1")
    if args.expert_model_parallel_size != EXPERT_PARALLEL_SIZE or args.expert_tensor_parallel_size != 1:
        raise ValueError("wholepack requires EP8/ETP1")
    if args.qkv_format != "thd" or not args.use_dynamic_batch_size:
        raise ValueError("wholepack requires dynamic THD binpack")
    if getattr(args, "virtual_pipeline_model_parallel_size", None) not in (None, 1):
        raise ValueError("wholepack does not support VPP")
    if getattr(args, "sft_async_prepack", False) or getattr(args, "dynamic_context_parallel", False):
        raise ValueError("wholepack draft does not support async prepack or dynamicCP")
    sampler_padding(args.n_samples_per_prompt, WORLD_SIZE // tp, capacity, pad_size)
    if getattr(args, "rollout_batch_size", GLOBAL_BATCH_SIZE) != GLOBAL_BATCH_SIZE:
        raise ValueError("wholepack requires one GBS1600 optimizer step per rollout")
    return capacity, pad_size
