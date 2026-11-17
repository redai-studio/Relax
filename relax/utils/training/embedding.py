# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packed embedding pooling and InfoNCE primitives.

These functions neither construct an optimizer nor perform a training step. The
caller supplies the data-parallel group explicitly. A contrastive batch must
remain complete across the loss call; accumulating independent smaller InfoNCE
batches changes the negative set.
"""

import math
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


def validate_embedding_args(args: Namespace) -> None:
    """Validate the shared SFT embedding path before job creation."""
    if getattr(args, "loss_type", None) != "sft" or getattr(args, "train_backend", "megatron") != "megatron":
        raise ValueError("embedding requires --loss-type sft --train-backend megatron")
    if getattr(args, "megatron_to_hf_mode", "bridge") != "bridge":
        raise ValueError("embedding requires --megatron-to-hf-mode bridge")
    if not getattr(args, "custom_dataset_class_path", None):
        raise ValueError("embedding requires a pair-atomic --custom-dataset-class")
    if not getattr(args, "custom_loss_function_path", None):
        raise ValueError("embedding requires --custom-loss-function-path")
    temperature = float(getattr(args, "contrastive_temperature", 0.05))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("--contrastive-temperature must be finite and positive")
    if int(getattr(args, "global_batch_size", 0) or 0) <= 0:
        raise ValueError("embedding global_batch_size counts positive query/positive pairs")
    if int(getattr(args, "n_samples_per_prompt", 1)) != 1:
        raise ValueError("embedding requires n_samples_per_prompt=1: each TQ row is already a complete pair")
    if not getattr(args, "use_dynamic_batch_size", False) or int(getattr(args, "max_tokens_per_gpu", 0) or 0) <= 0:
        raise ValueError("embedding requires dynamic batching with a positive --max-tokens-per-gpu")
    if int(getattr(args, "data_pad_size_multiplier", 128)) <= 0:
        raise ValueError("embedding requires a positive data padding multiple")
    if getattr(args, "num_steps_per_rollout", 1) not in (None, 1):
        raise ValueError("embedding requires one complete pair batch per rollout")
    for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
        if int(getattr(args, name, 1)) != 1:
            raise ValueError("The initial embedding integration requires TP=PP=CP=1")
    unsupported = (
        "fully_async",
        "hybrid",
        "sequence_parallel",
        "dynamic_context_parallel",
        "allgather_cp",
        "calculate_per_token_loss",
        "sft_chunked_logits",
        "enable_mtp_training",
        "mtp_only_training",
        "sft_async_prepack",
        "sft_train_data_prefetch",
        "sft_image_preprocess_on_rank",
        "debug_train_only",
        "recompute_loss_function",
        "overlap_moe_expert_parallel_comm",
        "partial_rollout",
        "use_dynamic_global_batch_size",
        "use_opd",
    )
    enabled = [name for name in unsupported if getattr(args, name, False)]
    optional = (
        "virtual_pipeline_model_parallel_size",
        "sft_predict_interval",
        "eval_interval",
        "save_hf",
        "custom_model_provider_path",
        "mtp_num_layers",
        "eval_prompt_data",
    )
    enabled.extend(name for name in optional if getattr(args, name, None))
    enabled.extend(name for name in ("eval_size", "num_labels") if getattr(args, name, None) is not None)
    if enabled:
        raise ValueError(f"embedding does not support these options yet: {', '.join(enabled)}")
    if getattr(args, "qkv_format", "thd") != "thd":
        raise ValueError("embedding requires --qkv-format thd")
    if getattr(args, "sft_oversize_strategy", "keep") != "keep":
        raise ValueError("embedding requires --sft-oversize-strategy keep; pairs cannot be truncated or dropped")
    if not getattr(args, "use_gloo_process_groups", False):
        raise ValueError("embedding requires --use-gloo-process-groups for DP batch agreement")
    if getattr(args, "sft_training_mode", "sft") != "sft":
        raise ValueError("embedding requires the SFT pair producer, not CPT token packing")


def freeze_embedding_unused_heads(model: torch.nn.Module) -> None:
    """Keep unused vocabulary/embedding parameters out of DDP optimizer
    buckets."""
    owner = getattr(model, "language_model", model)
    output = getattr(owner, "output_layer", None)
    if output is not None:
        output.requires_grad_(False)
    embedding = getattr(owner, "embedding", None)
    if embedding is not None:
        embedding.requires_grad_(False)


def expand_embedding_pairs(rows: dict[str, Any]) -> dict[str, Any]:
    """Expand pair-atomic TQ payloads only after DP row assignment."""
    if "embedding_pair_count" in rows:
        return dict(rows)
    pairs = rows.get("multimodal_train_inputs")
    if not pairs:
        raise ValueError("embedding requires tokenized query/positive pairs in multimodal_train_inputs")
    output: dict[str, Any] = {"tokens": [], "loss_masks": [], "total_lengths": [], "response_lengths": []}
    multimodal = []
    positions = []
    for pair in pairs:
        pair = getattr(pair, "data", pair)
        if not isinstance(pair, dict) or set(pair) != {"query", "positive"}:
            raise ValueError("Each embedding TQ row must contain exactly query and positive branches")
        for side in ("query", "positive"):
            branch = pair[side]
            ids = torch.as_tensor(branch["input_ids"], dtype=torch.long)
            if ids.device.type != "cpu" or ids.ndim != 1 or ids.numel() == 0:
                raise ValueError("Embedding branches must provide nonempty one-dimensional CPU token tensors")
            output["tokens"].append(ids)
            output["loss_masks"].append(torch.ones_like(ids))
            output["total_lengths"].append(ids.numel())
            output["response_lengths"].append(ids.numel())
            positions.append(branch.get("embedding_position", ids.numel() - 1))
            multimodal.append(
                {key: value for key, value in branch.items() if key not in {"input_ids", "embedding_position"}} or None
            )
    if any(item is not None for item in multimodal):
        output["multimodal_train_inputs"] = multimodal
    output["embedding_pair_count"] = len(pairs)
    output["embedding_positions"] = positions
    if "dynamic_global_batch_size" in rows:
        output["dynamic_global_batch_size"] = rows["dynamic_global_batch_size"]
    return output


def pool_packed_embeddings(
    hidden_states: torch.Tensor,
    lengths: Sequence[int],
    *,
    positions: Sequence[int] | None = None,
    padded_lengths: Sequence[int] | None = None,
) -> torch.Tensor:
    """Pool one state per sequence, defaulting to its last real token."""
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("Embedding pooling requires packed hidden states shaped [1, tokens, hidden]")
    if not lengths:
        raise ValueError("Embedding pooling requires a nonempty sequence batch")
    if positions is None:
        positions = [length - 1 for length in lengths]
    if padded_lengths is None:
        padded_lengths = lengths
    offsets = []
    cursor = 0
    for length, position, padded_length in zip(lengths, positions, padded_lengths, strict=True):
        if not isinstance(position, int) or isinstance(position, bool) or not 0 <= position < length:
            raise ValueError("Embedding position must index a real token in its sequence")
        if padded_length < length:
            raise ValueError("Packed embedding lengths cannot truncate a sequence")
        offsets.append(cursor + position)
        cursor += padded_length
    if cursor > hidden_states.shape[1]:
        raise ValueError("Packed hidden states do not cover the complete embedding sequences")
    indices = torch.tensor(offsets, dtype=torch.long, device=hidden_states.device)
    return hidden_states[0].index_select(0, indices)


def gather_with_local_grad(features: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
    """Gather detached remote features, preserving only this rank's graph."""
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return features
    # Shape agreement is checked collectively before all_gather, so one rank's
    # partial batch cannot leave its peers hanging in a tensor collective.
    shape = torch.tensor(features.shape, device=features.device, dtype=torch.long)
    shapes = [torch.empty_like(shape) for _ in range(world_size)]
    dist.all_gather(shapes, shape, group=process_group)
    torch._assert_async(torch.stack(shapes).eq(shape).all(), "InfoNCE ranks must have equal complete feature batches")
    gathered = [torch.empty_like(features) for _ in range(world_size)]
    dist.all_gather(gathered, features.contiguous(), group=process_group)
    gathered[dist.get_rank(process_group)] = features
    return torch.cat(gathered, dim=0)


def infonce_loss(
    query: torch.Tensor,
    positive: torch.Tensor,
    *,
    temperature: float = 0.05,
    process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Native mean one-way InfoNCE, before the backend's gradient reduction.

    Every DP rank evaluates the global objective with only its own feature
    graph attached. The caller must preserve the reference 1/world_size
    gradient scale when adapting this mean to Megatron's loss normalization.
    """
    if query.ndim != 2 or query.shape != positive.shape or query.shape[0] == 0:
        raise ValueError("InfoNCE features must have matching nonempty [pairs, hidden] shapes")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("InfoNCE temperature must be finite and positive")
    if process_group is None and dist.is_initialized():
        raise ValueError("Distributed InfoNCE requires an explicit data-parallel process group")
    if process_group is not None:
        query = gather_with_local_grad(query, process_group)
        positive = gather_with_local_grad(positive, process_group)
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    scores = F.cosine_similarity(query.unsqueeze(1), positive.unsqueeze(0), dim=-1) / temperature
    return F.cross_entropy(scores, torch.arange(scores.shape[0], device=scores.device))


def scale_for_megatron_sample_mean(
    mean_loss: torch.Tensor, *, global_pairs: int, dp_size: int, num_microbatches: int = 1
) -> torch.Tensor:
    """Adapt a mean to the shared sample-sum loss wrapper, with one microbatch.

    The shared wrapper multiplies by DP/global_batch, then DDP averages
    gradients. Multiplying here by global_batch/DP cancels only the wrapper's
    scaling. It does not introduce an extra world-size multiplier into native
    InfoNCE.
    """
    if dp_size <= 0 or global_pairs <= 0 or global_pairs % dp_size:
        raise ValueError("The complete contrastive pair batch must divide evenly across data-parallel ranks")
    if num_microbatches != 1:
        raise ValueError(
            "InfoNCE requires one complete contrastive microbatch; independent InfoNCE accumulation differs"
        )
    return mean_loss * (global_pairs // dp_size)
