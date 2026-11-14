# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Register raw Qwen mappings against the existing converter's meta outputs."""

from relax.distributed.weight_sync.source_map import TensorMapping


def qwen_source_mapping(args, model_name, info, tp_size):
    import torch

    from ..weight_conversion import convert_to_hf

    # Other converters retain their complete export route until their
    # transformations have an explicit lossless mapping.
    if model_name not in {"qwen2", "qwen3", "qwen3moe", "qwen3vl", "qwen3vlmoe"}:
        return None
    if getattr(args, "q_lora_rank", None) is not None:
        return None
    sharded = bool(info.attrs.get("tensor_model_parallel")) and info.attrs.get("parallel_mode") != "duplicated"
    dimension = info.attrs.get("partition_dim") if sharded else None
    shape = tuple(info.shape)
    full_shape = list(shape)
    name = info.name.replace(".to_wrap.", ".")
    kind = "identity"
    if "vision_model" not in name:
        if "linear_fc1.weight" in name:
            kind = "gate_up"
        elif "linear_qkv.weight" in name or name.endswith("linear_qkv.bias"):
            kind = "qkv"
        if "linear_fc2.weight" in name and dimension == 0:
            dimension = 1
    if sharded:
        full_shape[dimension] *= tp_size
    outputs = convert_to_hf(
        args, model_name, info.name, torch.empty(full_shape, device="meta", dtype=info.dtype), None
    )
    crop = name.endswith(("embedding.word_embeddings.weight", "output_layer.weight"))
    head_dim = getattr(args, "kv_channels", None) or args.hidden_size // args.num_attention_heads
    return TensorMapping(
        shape=shape,
        tp_size=tp_size,
        partition_dim=dimension,
        outputs=tuple((key, tuple(value.shape)) for key, value in outputs),
        kind=kind,
        qkv_groups=args.num_query_groups,
        q_heads_per_group=args.num_attention_heads // args.num_query_groups,
        head_dim=head_dim,
        crop_vocabulary=crop,
    )
