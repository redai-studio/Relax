# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import re

import torch
from megatron.core import mpu

from relax.utils.logging_utils import get_logger
from relax.utils.misc import get_hf_config


logger = get_logger(__name__)


def load_weight(args, model, weights: list[tuple[str, torch.Tensor]]) -> None:
    """Load a list of (name, tensor) weights into the local model parameters.

    Handles expert index remapping and TP chunking to map the received full
    tensors into each rank's sharded parameter.
    """
    params_dict = dict(model[0].named_parameters())
    if args.num_experts:
        num_local_experts = args.num_experts // args.expert_model_parallel_size
        local_experts_range = list(
            range(
                num_local_experts * mpu.get_expert_model_parallel_rank(),
                num_local_experts * (mpu.get_expert_model_parallel_rank() + 1),
            )
        )

    for name, loaded_weight in weights:
        name = name.removeprefix("module.")
        if "mlp.experts" in name:
            m = re.search(r"weight(\d+)$", name)
            if int(m.group(1)) in local_experts_range:
                name = name[: m.start()] + f"weight{int(m.group(1)) % num_local_experts}"
            else:
                # not owned by this rank
                continue

        if name not in params_dict:
            logger.warning(f"{name} not in params_dict")
            continue

        param = params_dict.get(name)
        loaded_weight = chunk_param(args, name, param, loaded_weight)
        default_weight_loader(param, loaded_weight)


def chunk_param(
    args,
    name: str,
    target_param: torch.Tensor,
    full_param: torch.Tensor,
) -> torch.Tensor:
    """Slice a full parameter into the local tensor-parallel shard.

    This function is strictly symmetric to all_gather_param and handles
    special cases (GLU layout and grouped MoE partition bug) to ensure
    correct per-rank slicing.

    Args:
        name: Parameter name (used to detect special cases).
        target_param: The local module parameter describing partitioning.
        full_param: The gathered full tensor to be sliced.

    Returns:
        The shard tensor for the current tensor-parallel rank.
    """
    name = name.replace(".to_wrap.", ".")
    # 1. expert_bias is not sharded
    if "expert_bias" in name:
        return full_param

    # 2. If the parameter is not tensor-parallel, return full tensor
    if (
        not getattr(target_param, "tensor_model_parallel", False)
        or getattr(target_param, "parallel_mode", None) == "duplicated"
    ):
        return full_param

    # 3. Choose TP config: expert TP or regular TP
    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_rank = mpu.get_expert_tensor_parallel_rank()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()

    # 4. Verify stride
    # NOTE: Megatron-LM (megatron/core/transformer/mlp.py) sets partition_stride=2
    # for GLU/SwiGLU linear_fc1 layers. The rechunk logic below handles this correctly.
    partition_dim = target_param.partition_dim
    if "linear_fc1" not in name:
        assert getattr(target_param, "partition_stride", 1) == 1, (
            f"{name}: partition_stride={getattr(target_param, 'partition_stride', 1)} != 1 is not supported"
        )

    # 5. Workaround grouped MoE partition bug for linear_fc2.weight
    effective_partition_dim = partition_dim
    if "linear_fc2.weight" in name and partition_dim == 0 and "vision_model" not in name:
        # grouped MoE used an incorrect dim during merge; reverse it here
        effective_partition_dim = 1

    # 6. Qwen3.5 GDN: self_attention.conv1d.weight (inverse of all_gather_param)
    if "self_attention.conv1d.weight" in name:
        config = get_hf_config(args.hf_checkpoint).text_config
        qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
        v_dim = config.linear_value_head_dim * config.linear_num_value_heads

        q_full, k_full, v_full = torch.split(full_param, [qk_dim, qk_dim, v_dim], dim=0)
        shards = []
        for component in [q_full, k_full, v_full]:
            chunks = torch.chunk(component, tp_size, dim=0)
            shards.append(chunks[tp_rank])
        return torch.cat(shards, dim=0)

    # Qwen3.5 GDN: self_attention.in_proj.weight (inverse of all_gather_param)
    if "self_attention.in_proj.weight" in name:
        config = get_hf_config(args.hf_checkpoint).text_config
        qk_head_dim = config.linear_key_head_dim
        v_head_dim = config.linear_value_head_dim
        num_qk_heads = config.linear_num_key_heads
        num_v_heads = config.linear_num_value_heads
        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads

        segments = torch.split(full_param, [qk_dim, qk_dim, v_dim, v_dim, num_v_heads, num_v_heads], dim=0)
        shards = []
        for seg in segments:
            chunks = torch.chunk(seg, tp_size, dim=0)
            shards.append(chunks[tp_rank])
        return torch.cat(shards, dim=0)

    # 7. Special handling for GLU linear_fc1.weight layout
    if "linear_fc1.weight" in name and "vision_model" not in name:
        # merge pattern produced layout: [p0_a, p0_b, p1_a, p1_b, ...]
        # split into 2*tp_size chunks then recombine per-rank as [pa_i || pb_i]
        assert full_param.size(0) % (2 * tp_size) == 0, (
            f"linear_fc1.weight dim0 size {full_param.size(0)} not divisible by 2*tp_size={2 * tp_size}"
        )
        chunks = torch.chunk(full_param, 2 * tp_size, dim=0)
        part_a = chunks[tp_rank]
        part_b = chunks[tp_rank + tp_size]
        return torch.cat([part_a, part_b], dim=0)

    # 8. Regular partitioning (including special dim overrides above)
    assert full_param.size(effective_partition_dim) % tp_size == 0, (
        f"Param {name} dim{effective_partition_dim} size {full_param.size(effective_partition_dim)} "
        f"not divisible by tp_size={tp_size}"
    )
    partitions = torch.chunk(full_param, tp_size, dim=effective_partition_dim)
    return partitions[tp_rank]


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Copy a loaded tensor into a model parameter with sanity checks.

    Scalars are handled by filling the parameter. On mismatch, raises.
    """
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Scalar parameters: use fill to avoid shape issues
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()})"
            )
            param.data.copy_(loaded_weight)
    except Exception:
        # Keep exception to allow debugging at the callsite
        raise
