# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model and loss helpers for HF models wrapped by FSDPTurbo.

GRPO shares Relax's text policy clipping; SFT mirrors Megatron's masked NLL.
With Ulysses CP, callers feed complete batches while model forwards return
local sequence shards, and SFT statistics are reduced over the CP group.
"""

from __future__ import annotations

import os
import re
import types
from argparse import Namespace
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional

from relax.backends.fsdp_turbo.router_precision import enable_qwen4_router_fp32
from relax.engine.sft.runtime import is_sft_mode
from relax.utils.device import make_current_torch_device
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger
from relax.utils.training.ppo_utils import compute_policy_loss


logger = get_logger(__name__)


def resolve_auto_model_class(hf_config):
    """Pick the transformers auto class from the checkpoint architecture."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    architecture = (getattr(hf_config, "architectures", None) or [""])[0]
    if any(tag in architecture for tag in ("ForConditionalGeneration", "ForImageTextToText", "ForVision2Seq")):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def load_hf_model(model_path: str, torch_dtype: torch.dtype):
    """Load the HF training model for FSDPTurbo wrapping."""
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    auto_cls = resolve_auto_model_class(hf_config)
    logger.info(f"Loading training model from {model_path} via {auto_cls.__name__} ({torch_dtype})")
    return auto_cls.from_pretrained(model_path, dtype=torch_dtype, trust_remote_code=True)


def _local_shard_size_and_offset(placement, *args):
    method = getattr(placement, "local_shard_size_and_offset", None)
    if method is None:
        method = placement._local_shard_size_and_offset
    return method(*args)


def _pad_batch(
    tokens: Sequence[torch.Tensor],
    device: torch.device,
    pad_id_token: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad tokens with ``pad_id_token``, using zero when it is
    omitted."""
    max_len = max(int(t.numel()) for t in tokens)
    input_ids = torch.full(
        (len(tokens), max_len), 0 if pad_id_token is None else pad_id_token, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(len(tokens), max_len, dtype=torch.long, device=device)
    for row, tensor in enumerate(tokens):
        length = int(tensor.numel())
        input_ids[row, :length] = tensor.to(device=device, dtype=torch.long)
        attention_mask[row, :length] = 1
    if not torch.is_inference(attention_mask):
        attention_mask._relax_sft_prefix = (attention_mask._version, tuple(int(t.numel()) for t in tokens))
    return input_ids, attention_mask


_NLL_CHUNK_ROWS = 1024  # ~0.95 GiB per FP32 tensor at the Qwen 248k vocabulary.


def _fp32_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Recompute FP32 CE intermediates in backward instead of retaining a full
    FP32 vocabulary."""
    from torch.utils.checkpoint import checkpoint

    def compute(chunk_logits: torch.Tensor, chunk_targets: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=chunk_logits.device.type, enabled=False):
            return torch.nn.functional.cross_entropy(
                chunk_logits.float() / temperature, chunk_targets, reduction="none", ignore_index=-100
            )

    if torch.is_grad_enabled() and logits.requires_grad:
        return checkpoint(compute, logits, targets, use_reentrant=False, preserve_rng_state=False)
    return compute(logits, targets)


def _ulysses_cp_context() -> tuple[Any, int, int]:
    """Return ``(group, rank, size)``, or ``(None, 0, 1)`` when CP is
    unavailable.

    The lazy import keeps tests and driver processes free of FSDPTurbo.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return None, 0, 1
    try:
        from fsdp_turbo.distributed.context_parallel.ulysses.utils import get_ulysses_context
    except ImportError:
        return None, 0, 1
    return get_ulysses_context()


def _cp_local_mask(mask: torch.Tensor, group, local_positions: int) -> torch.Tensor:
    """Pad and contiguously shard a full supervised mask like the model's
    sequence."""
    from fsdp_turbo.distributed.context_parallel.batch import (
        pad_tensor_to_group,
        split_tensor_to_group,
    )

    padded_len = local_positions * dist.get_world_size(group)
    if mask.numel() < padded_len:
        mask = torch.nn.functional.pad(mask, (0, padded_len - mask.numel()), value=0.0)
    return split_tensor_to_group(pad_tensor_to_group(mask, group, dim=0, pad_value=0.0), group, dim=0)


def _cp_shard_targets(targets: torch.Tensor, group, local_positions: int) -> torch.Tensor:
    """Pad global next-token targets to the model's sequence length, then
    shard."""
    from fsdp_turbo.distributed.context_parallel.batch import split_tensor_to_group

    padded_length = local_positions * dist.get_world_size(group)
    if targets.size(1) > padded_length:
        raise ValueError(
            f"CP logits cover {padded_length} global positions, fewer than the {targets.size(1)} target positions."
        )
    targets = torch.nn.functional.pad(targets, (0, padded_length - targets.size(1)), value=-100)
    return split_tensor_to_group(targets, group, dim=1)


def _forward_token_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0,
    *,
    sft_targets: torch.Tensor | None = None,
    **model_kwargs,
) -> torch.Tensor:
    """Return next-token NLL as fp32 ``[B, T-1]`` using chunked fused CE.

    Chunking avoids retaining full ``[positions, vocab]`` fp32 intermediates.
    With Ulysses CP, shift targets on the complete sequence before applying the
    model's contiguous shard exactly once.
    """
    _ensure_mm_token_type_ids(model, input_ids, model_kwargs)
    # Route labels only to MTP; the main head uses the chunked NLL below.
    labels = model_kwargs.pop("labels", None)
    mtp_loss_mask = model_kwargs.pop("mtp_loss_mask", None)
    forward_kwargs = {"use_cache": False, **model_kwargs}
    if sft_targets is not None:
        return model(input_ids=input_ids, attention_mask=attention_mask, sft_targets=sft_targets, **forward_kwargs)
    # MTP-patched models (Qwen3.8/qwen4_exp) consume the final hidden states
    # in their forward hook; without this the MTP loss silently never runs.
    inner = model if getattr(model, "mtp", None) is not None else getattr(model, "model", model)
    if getattr(inner, "mtp", None) is not None:
        forward_kwargs["output_hidden_states"] = True
        if labels is not None:
            forward_kwargs["labels"] = labels
            forward_kwargs["mtp_loss_mask"] = mtp_loss_mask
            forward_kwargs["mtp_only_labels"] = True
    logits = model(input_ids=input_ids, attention_mask=attention_mask, **forward_kwargs).logits

    ulysses_group, _ulysses_rank, ulysses_size = _ulysses_cp_context()
    if ulysses_size > 1:
        batch_size = input_ids.size(0)
        # Build global next-token targets before sharding; -100 ignores padding and the last position.
        targets = torch.full(input_ids.shape, -100, dtype=torch.long, device=input_ids.device)
        targets[:, :-1] = input_ids[:, 1:]
        targets = _cp_shard_targets(targets, ulysses_group, logits.size(1))
        if logits.size(1) != targets.size(1):
            raise ValueError(
                "Ulysses CP shape mismatch: local logits cover "
                f"{logits.size(1)} positions but the target shard has {targets.size(1)}. "
                "The model forward must shard the fused sequence post-fusion across "
                "the ulysses group (qwen4_exp: built into fsdp_turbo's consolidated "
                "modeling_qwen4_exp forward; other recipes: via module_patches) — "
                "Relax feeds whole samples and never splits batches externally."
            )
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        nll = torch.empty(flat_logits.size(0), dtype=torch.float32, device=input_ids.device)
        chunk = _NLL_CHUNK_ROWS
        for start in range(0, flat_logits.size(0), chunk):
            end = min(start + chunk, flat_logits.size(0))
            chunk_nll = _fp32_token_cross_entropy(flat_logits[start:end], flat_targets[start:end], temperature)
            nll[start:end] = chunk_nll.to(torch.float32)
        return nll.reshape(batch_size, -1)

    shifted_logits = logits[:, :-1]  # view [B, T-1, V]
    batch_size, num_positions, vocab_size = shifted_logits.shape
    flat_logits = shifted_logits.reshape(batch_size * num_positions, vocab_size)
    flat_targets = input_ids[:, 1:].reshape(-1)
    nll = torch.empty(batch_size * num_positions, dtype=torch.float32, device=input_ids.device)
    chunk = _NLL_CHUNK_ROWS
    for start in range(0, batch_size * num_positions, chunk):
        end = min(start + chunk, batch_size * num_positions)
        chunk_nll = _fp32_token_cross_entropy(flat_logits[start:end], flat_targets[start:end], temperature)
        nll[start:end] = chunk_nll.to(torch.float32)
    return nll.reshape(batch_size, num_positions)


def _take_mtp_loss(model: torch.nn.Module) -> torch.Tensor | None:
    """Consume the MTP loss stashed on the module by the patched forward.

    FSDP2 rebuilds the output dataclass and drops dynamic attributes. Return
    None when no loss was produced, including ``no_grad`` evaluation.
    """
    inner = model if hasattr(model, "_last_mtp_loss") else getattr(model, "model", model)
    mtp_loss = getattr(inner, "_last_mtp_loss", None)
    if mtp_loss is None:
        return None
    inner._last_mtp_loss = None  # consume for the next microbatch
    if isinstance(mtp_loss, list):
        mtp_loss = [loss for loss in mtp_loss if loss is not None]
        if not mtp_loss:
            return None
        mtp_loss = torch.stack(mtp_loss).sum()
    return mtp_loss


def _forward_token_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    **model_kwargs,
) -> torch.Tensor:
    """Per-position log-probs of the NEXT token: ``[B, T-1]`` (fp32)."""
    return -_forward_token_nll(model, input_ids, attention_mask, **model_kwargs)


def _ensure_mm_token_type_ids(model: torch.nn.Module, input_ids: torch.Tensor, mm_kwargs: dict) -> None:
    """Rebuild Qwen3.5 M-RoPE modality ids when processors omit them.

    Placeholder tokens map to 0=text, 1=image and 2=video, matching
    ``input_ids`` shape. Existing values are discarded to avoid stale
    collation.
    """
    if "image_grid_thw" not in mm_kwargs and "video_grid_thw" not in mm_kwargs:
        return
    # Rebuild this per-token field instead of collating it with per-image fields.
    mm_kwargs.pop("mm_token_type_ids", None)
    image_token_id = None
    video_token_id = None
    for candidate in (model, getattr(model, "model", None)):  # unwrap FSDPTurbo
        config = getattr(candidate, "config", None)
        if config is None:
            continue
        image_token_id = getattr(config, "image_token_id", image_token_id)
        video_token_id = getattr(config, "video_token_id", video_token_id)
    if image_token_id is None and video_token_id is None:
        return  # not a Qwen-style VL config; assume the model needs no marker
    token_type_ids = torch.zeros_like(input_ids)
    if image_token_id is not None:
        token_type_ids[input_ids == int(image_token_id)] = 1
    if video_token_id is not None:
        token_type_ids[input_ids == int(video_token_id)] = 2
    mm_kwargs["mm_token_type_ids"] = token_type_ids


def _collate_multimodal_inputs(
    multimodal_train_inputs: Sequence[Mapping[str, Any] | None] | None,
    microbatch: Sequence[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Concatenate multimodal tensors along dim 0 in sample order.

    Qwen-VL grid metadata describes image boundaries, so no padding is needed.
    """
    if multimodal_train_inputs is None:
        return {}
    per_key: dict[str, list[torch.Tensor]] = {}
    for sample_index in microbatch:
        mm = multimodal_train_inputs[sample_index]
        if mm is None:
            continue
        for key, value in mm.items():
            if key == "mm_token_type_ids":
                continue  # per-token field: rebuilt in _ensure_mm_token_type_ids
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            per_key.setdefault(key, []).append(tensor)
    return {key: torch.cat(tensors, dim=0).to(device) for key, tensors in per_key.items()}


def _slice_response_log_probs(
    token_log_probs: torch.Tensor,
    row: int,
    total_length: int,
    response_length: int,
) -> torch.Tensor:
    """Response-token log-probs for one sample (predicted at positions t-1)."""
    start = total_length - response_length
    return token_log_probs[row, start - 1 : total_length - 1]


@torch.no_grad()
def compute_rollout_log_probs(
    model: torch.nn.Module,
    rollout_data: Mapping[str, Any],
    microbatches: Sequence[Sequence[int]],
    device: torch.device,
    temperature: float = 1.0,
) -> list[torch.Tensor]:
    """Recompute response log-probs when rollout log-probs are not reused."""
    model.eval()
    tokens = rollout_data["tokens"]
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    log_probs: list[torch.Tensor | None] = [None] * len(tokens)
    # Retained eval graphs can exhaust HBM and hang the NPU at the next HCCL collective.
    with torch.no_grad():
        for microbatch in microbatches:
            input_ids, attention_mask = _pad_batch([tokens[i] for i in microbatch], device)
            token_log_probs = _forward_token_log_probs(model, input_ids, attention_mask, temperature=temperature)
            for row, sample_index in enumerate(microbatch):
                log_probs[sample_index] = _slice_response_log_probs(
                    token_log_probs,
                    row,
                    int(total_lengths[sample_index]),
                    int(response_lengths[sample_index]),
                ).clone()
    model.train()
    return [lp for lp in log_probs if lp is not None]


def grpo_microbatch_loss(
    model: torch.nn.Module,
    args,
    rollout_data: Mapping[str, Any],
    microbatch: Sequence[int],
    device: torch.device,
    *,
    step_sample_count: float | None = None,
    step_token_count: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO clipped-surrogate loss for one microbatch.

    Accumulate per-sample masked token means over the complete step. The actor
    supplies data-group-averaged sample/token counts for unequal ranks; absent
    overrides, use the complete local batch rather than this microbatch.
    """
    tokens = rollout_data["tokens"]
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    loss_masks = rollout_data["loss_masks"]
    advantages = rollout_data["advantages"]
    old_log_probs = rollout_data["rollout_log_probs" if args.use_rollout_logprobs else "log_probs"]

    input_ids, attention_mask = _pad_batch([tokens[i] for i in microbatch], device)
    token_log_probs = _forward_token_log_probs(
        model, input_ids, attention_mask, temperature=float(getattr(args, "rollout_temperature", 1.0))
    )

    ratio_clipped_frac = 0.0
    pg_losses: list[torch.Tensor] = []
    sampled_nlls: list[torch.Tensor] = []
    for row, sample_index in enumerate(microbatch):
        total_length = int(total_lengths[sample_index])
        response_length = int(response_lengths[sample_index])
        new_logp = _slice_response_log_probs(token_log_probs, row, total_length, response_length)
        old_logp = old_log_probs[sample_index].to(device=new_logp.device, dtype=new_logp.dtype)
        adv = advantages[sample_index].to(device=new_logp.device, dtype=torch.float32)
        mask = loss_masks[sample_index].to(device=new_logp.device, dtype=torch.float32)

        clip_eps = float(getattr(args, "eps_clip", 0.2))
        high = getattr(args, "eps_clip_high", None)
        high = clip_eps if high is None else float(high)
        # The shared helper is torch.compile-decorated for CUDA. Its eager
        # function keeps the same math without invoking CUDA Inductor on NPU.
        loss_fn = compute_policy_loss
        if new_logp.device.type in ("npu", "privateuseone"):
            loss_fn = getattr(loss_fn, "_torchdynamo_orig_callable", loss_fn)
        loss_per_token, clipped = loss_fn(
            old_logp - new_logp,
            adv.detach(),
            clip_eps,
            high,
            getattr(args, "eps_clip_c", None),
        )
        with torch.no_grad():
            ratio_clipped_frac += ((clipped * mask).sum() / mask.sum().clamp(min=1.0)).item() / len(microbatch)
            sampled_nlls.append((-new_logp * mask).sum() / mask.sum().clamp(min=1.0))

        pg_losses.append((loss_per_token * mask).sum() / mask.sum().clamp(min=1.0))

    from relax.algorithms import get_algorithm

    is_token_global = get_algorithm(args.advantage_estimator).advantage_normalization == "token_global"
    if is_token_global or getattr(args, "calculate_per_token_loss", False):
        # Weight sample means by supervised tokens so microbatch gradients sum to the global token mean.
        local_total = step_token_count
        if local_total is None:
            local_total = sum(float(torch.as_tensor(mask).sum()) for mask in loss_masks)
        local_total = float(local_total) if local_total > 0 else 1.0
        weighted = torch.stack(
            [
                pg_loss * float(loss_masks[sample_index].to(torch.float32).sum())
                for pg_loss, sample_index in zip(pg_losses, microbatch)
            ]
        ).sum()
        loss = weighted / local_total
    else:
        sample_count = step_sample_count if step_sample_count is not None else len(tokens)
        loss = torch.stack(pg_losses).sum() / max(float(sample_count), 1.0)

    metrics = {
        "pg_loss": float(loss.detach().item()),
        "ratio_clipped_frac": ratio_clipped_frac,
        "sampled_nll": float(torch.stack(sampled_nlls).mean().item()),
    }
    return loss, metrics


def _supervised_mask(loss_mask: torch.Tensor, total_length: int, device: torch.device) -> torch.Tensor:
    """Return the next-token supervised mask for positions ``[0, T-1)``.

    Shift left once so position ``i`` supervises token ``i + 1``.
    """
    from relax.utils.sft_utils import align_loss_mask_for_sft

    aligned = align_loss_mask_for_sft(loss_mask.to(dtype=torch.float32))
    return aligned[: total_length - 1].to(device=device)


def _chunked_sft_targets(
    model: torch.nn.Module, input_ids: torch.Tensor, rollout_data: Mapping[str, Any], microbatch: Sequence[int]
) -> torch.Tensor | None:
    if not getattr(getattr(model, "model", model), "sft_chunk_rows", 0):
        return None

    targets = torch.full_like(input_ids[:, 1:], -100)
    for row, sample_index in enumerate(microbatch):
        length = int(rollout_data["total_lengths"][sample_index])
        mask = _supervised_mask(rollout_data["loss_masks"][sample_index], length, input_ids.device)
        targets[row, : length - 1] = input_ids[row, 1:length].masked_fill(mask == 0, -100)
    return targets


def sft_step_supervised_tokens(rollout_data: Mapping[str, Any]) -> int:
    """Count supervised next-token positions over all samples in the step.

    Complete, unsharded data masks make this a local computation. The caller
    averages these counts over the actual data group for the per-token
    normalizer. The group's size already excludes CP replicas.
    """
    from relax.utils.sft_utils import align_loss_mask_for_sft

    total = 0
    for total_length, mask in zip(rollout_data["total_lengths"], rollout_data["loss_masks"]):
        aligned = align_loss_mask_for_sft(mask.to(dtype=torch.float32))
        total += int(aligned[: int(total_length) - 1].sum().item())
    return total


def sft_microbatch_loss(
    model: torch.nn.Module,
    rollout_data: Mapping[str, Any],
    microbatch: Sequence[int],
    device: torch.device,
    num_local_samples: int,
    mtp_loss_scaling: float | None = None,
    per_token_loss: bool = False,
    step_token_count: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute masked next-token NLL for one SFT microbatch.

    By default, average per-sample token means and scale by
    ``len(microbatch) / num_local_samples`` for local-batch accumulation.
    With ``per_token_loss``, divide the raw token sum by ``step_token_count``:
    the whole-step count averaged over the actual data group. Together with
    CP's backward compensation, this offsets FSDP's gradient averaging and
    matches Megatron's global token mean. The caller must skip zero-token steps.

    Ulysses CP reduces NLL sums with forward SUM/backward x CP-size to offset
    FSDP averaging, and reduces grad-free token counts with a plain SUM.
    Positive ``mtp_loss_scaling`` adds the CP-aware MTP loss stashed by the
    patched forward.
    """
    tokens = rollout_data["tokens"]
    total_lengths = rollout_data["total_lengths"]
    mm_kwargs = _collate_multimodal_inputs(rollout_data.get("multimodal_train_inputs"), microbatch, device)

    eos_token_id = None
    for candidate in (model, getattr(model, "model", None)):  # unwrap FSDPTurbo
        config = getattr(candidate, "config", None)
        for token_config in (getattr(config, "text_config", None), config):
            eos_token_id = getattr(token_config, "eos_token_id", None)
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is not None:
                break
        if eos_token_id is not None:
            break
    input_ids, attention_mask = _pad_batch([tokens[i] for i in microbatch], device, pad_id_token=eos_token_id)
    mtp_labels = None
    mtp_loss_mask = None
    if mtp_loss_scaling:
        mtp_labels = input_ids.masked_fill(~attention_mask.bool(), -100)
        mtp_loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for row, sample_index in enumerate(microbatch):
            total_length = int(total_lengths[sample_index])
            supervised = rollout_data["loss_masks"][sample_index].to(device=device, dtype=torch.bool)
            # Position i supervises token i+1, matching Megatron's full_loss_masks.
            mtp_loss_mask[row, : total_length - 1] = supervised[1:total_length]
    targets = _chunked_sft_targets(model, input_ids, rollout_data, microbatch)
    token_nll = _forward_token_nll(
        model,
        input_ids,
        attention_mask,
        labels=mtp_labels,
        mtp_loss_mask=mtp_loss_mask,
        sft_targets=targets,
        **mm_kwargs,
    )

    ulysses_group, _ulysses_rank, ulysses_size = _ulysses_cp_context()
    nll_sums: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    for row, sample_index in enumerate(microbatch):
        total_length = int(total_lengths[sample_index])
        mask = _supervised_mask(rollout_data["loss_masks"][sample_index], total_length, device)
        if ulysses_size > 1:
            # token_nll is [B, L] (the local CP shard); align the complete
            # per-sample mask to the same shard.
            nll_row = token_nll[row]
            mask = _cp_local_mask(mask, ulysses_group, token_nll.size(1))
        else:
            nll_row = token_nll[row, : total_length - 1]
        nll_sums.append((nll_row * mask).sum())
        token_counts.append(mask.sum())
    nll_sums = torch.stack(nll_sums)
    token_counts = torch.stack(token_counts)
    if ulysses_size > 1:
        from fsdp_turbo.distributed.context_parallel.loss import cp_all_reduce_forward_backward

        nll_sums = cp_all_reduce_forward_backward(nll_sums, ulysses_group)
        token_counts = token_counts.detach()
        dist.all_reduce(token_counts, op=dist.ReduceOp.SUM, group=ulysses_group)

    if per_token_loss:
        # Normalize the raw token sum by the grad-free whole-step constant.
        if step_token_count is None or float(step_token_count) <= 0:
            raise ValueError(
                "per-token SFT loss requires a positive step_token_count; "
                "zero supervised-token steps must be skipped by the caller"
            )
        nll_sum = nll_sums.sum()
        mb_token_count = token_counts.sum()
        loss = nll_sum / float(step_token_count)
        metrics = {
            # Per-mb token-mean for display; the caller rebuilds the exact
            # step token-mean from the raw nll_sum / token_count metrics.
            "loss": float((nll_sum / mb_token_count.clamp(min=1.0)).detach().item()),
            "nll_sum": float(nll_sum.detach().item()),
            "token_count": float(mb_token_count.item()),
        }
    else:
        sample_losses = nll_sums / token_counts.clamp(min=1.0)
        mean_loss = sample_losses.mean()
        loss = mean_loss * (len(microbatch) / num_local_samples)
        metrics = {"loss": float(mean_loss.detach().item())}

    mtp_loss = _take_mtp_loss(model) if mtp_loss_scaling else None
    if mtp_loss is not None:
        if per_token_loss:
            # The patched forward returns an MTP mean; weight each microbatch
            # by its token share in per-token mode.
            micro_share = float(mb_token_count.item()) / float(step_token_count)
        else:
            micro_share = len(microbatch) / num_local_samples
        loss = loss + mtp_loss_scaling * mtp_loss * micro_share
        metrics["mtp_loss"] = float(mtp_loss.detach().item())
        metric_weight = float(mb_token_count.item()) if per_token_loss else float(len(microbatch))
        metrics["mtp_numerator"] = metrics["mtp_loss"] * metric_weight
        metrics["mtp_denominator"] = metric_weight
    return loss, metrics


@torch.no_grad()
def sft_microbatch_nll_stats(
    model: torch.nn.Module,
    rollout_data: Mapping[str, Any],
    microbatch: Sequence[int],
    device: torch.device,
) -> tuple[float, int]:
    """Return ``(sum_neg_log_prob, num_supervised_tokens)`` for evaluation.

    PPL is ``exp(sum / count)`` after the DP-wide sum. Ulysses CP statistics
    are already CP-reduced; the actor's later deduplication counts tokens once.
    """
    tokens = rollout_data["tokens"]
    total_lengths = rollout_data["total_lengths"]
    mm_kwargs = _collate_multimodal_inputs(rollout_data.get("multimodal_train_inputs"), microbatch, device)

    input_ids, attention_mask = _pad_batch([tokens[i] for i in microbatch], device)
    targets = _chunked_sft_targets(model, input_ids, rollout_data, microbatch)
    token_nll = _forward_token_nll(model, input_ids, attention_mask, sft_targets=targets, **mm_kwargs)

    ulysses_group, _ulysses_rank, ulysses_size = _ulysses_cp_context()
    nll_sums: list[torch.Tensor] = []
    token_counts: list[torch.Tensor] = []
    for row, sample_index in enumerate(microbatch):
        total_length = int(total_lengths[sample_index])
        mask = _supervised_mask(rollout_data["loss_masks"][sample_index], total_length, device)
        if ulysses_size > 1:
            nll_row = token_nll[row]
            mask = _cp_local_mask(mask, ulysses_group, token_nll.size(1))
        else:
            nll_row = token_nll[row, : total_length - 1]
        nll_sums.append((nll_row * mask).sum())
        token_counts.append(mask.sum())
    stats = torch.stack([torch.stack(nll_sums).sum(), torch.stack(token_counts).sum()])
    if ulysses_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=ulysses_group)
    return float(stats[0].item()), int(stats[1].item())


def _budget_from_config(model_config):
    """Structural budget: the model config declares num_hidden_layers /
    n_routed_experts (+ hash_moe count derived from mlp_layer_types, legacy
    num_hash_layers).

    Values are informational here; _check_budget cross-validates them against
    the target state dict.
    """
    out = {"N_LAYERS": None, "N_EXPERTS": None, "n_hash": 0}
    if model_config is None:
        return out

    def _attr(name, default=None):
        try:
            v = getattr(model_config, name)
            return v if v is not None else default
        except (AttributeError, TypeError):
            return default

    nl = _attr("num_hidden_layers")
    if nl is None:
        mlt = _attr("mlp_layer_types") or _attr("layer_types")
        nl = len(mlt) if mlt else None
    out["N_LAYERS"] = int(nl) if nl is not None else None
    ne = _attr("n_routed_experts")
    out["N_EXPERTS"] = int(ne) if ne is not None else None
    mlt = _attr("mlp_layer_types")
    if isinstance(mlt, (list, tuple)):
        out["n_hash"] = sum(1 for x in mlt if str(x) in ("hash_moe", "hash"))
    else:
        out["n_hash"] = int(_attr("num_hash_layers", 0) or 0)
    return out


def _check_budget(model_config, derived_layers, derived_experts, strict=False):
    """Cross-check model_config-declared layer/expert budget against values
    derived from the target state dict (ground truth).

    The derived values win on mismatch (a wrong config must never silently
    drive the crop). Returns True when consistent.
    """
    if model_config is None:
        return True

    def _attr(name, default=None):
        try:
            v = getattr(model_config, name)
            return int(v) if v is not None else default
        except (AttributeError, TypeError, ValueError):
            return default

    cfg_layers, cfg_experts = _attr("num_hidden_layers"), _attr("n_routed_experts")
    if cfg_layers is None or cfg_experts is None:
        logger.warning("budget cross-check skipped: model_config lacks num_hidden_layers / n_routed_experts")
        return True
    ok = cfg_layers == derived_layers and cfg_experts == derived_experts
    if ok:
        logger.info(
            f"budget config(layers={cfg_layers} experts={cfg_experts}) == "
            f"derived(layers={derived_layers} experts={derived_experts}) OK"
        )
        return True
    msg = (
        f"budget MISMATCH: config declares layers={cfg_layers} experts={cfg_experts} "
        f"but state dict derives layers={derived_layers} experts={derived_experts}; "
        f"using derived values"
    )
    if strict:
        raise ValueError(msg)
    logger.warning(msg)
    return False


def _pad_load_state(model, src_path, device="cpu", model_config=None, strict=False):
    """Load the 4-layer/16-expert novamlx slice into the adapter in memory.

    Translate keys, pack experts, crop compatible tensors and return load
    stats.
    """
    import glob as _glob
    import os as _os
    import re

    import torch
    from examples.deepseekv4 import load_weights as lw
    from safetensors import safe_open

    def dt(single):
        # top-level keys (head.weight -> lm_head.weight); always add one
        # `model.` layer so lm_head.weight hits model.lm_head.weight.
        return "model." + single

    state = model.state_dict()
    T = set(state.keys())
    stats = {"loaded": 0, "missing_final": [], "skipped_extra": [], "packed": 0, "skipped_shape_mismatch": []}

    # derive the per-config layer/expert budget from the target model state dict
    # Adapters expose the double-prefix layout (model.model.layers.N.*);
    # bare backbones the single-prefix one, so both are accepted.
    _layer_re = re.compile(r"model\.(?:model\.)?layers\.(\d+)\..+")
    _n_layers = [int(m.group(1)) for k in T for m in [_layer_re.fullmatch(k)] if m]
    assert _n_layers, "no model.layers.* keys in target state dict"
    N_LAYERS = 1 + max(_n_layers)
    _gu = [k for k in T if re.fullmatch(r"model\.(?:model\.)?layers\.0\.mlp\.experts\.gate_up_proj", k)]
    assert _gu, "model.layers.0.mlp.experts.gate_up_proj missing"
    N_EXPERTS = int(state[_gu[0]].shape[0])
    _cfg_budget = _budget_from_config(model_config)
    logger.info(
        f"budget from config: layers={_cfg_budget['N_LAYERS']} "
        f"experts={_cfg_budget['N_EXPERTS']} hash_moe={_cfg_budget['n_hash']}"
    )
    _check_budget(model_config, N_LAYERS, N_EXPERTS, strict=strict)

    shards = sorted(_glob.glob(_os.path.join(src_path, "*.safetensors")))
    direct, gate_crop, shared_crop, expert_pack = {}, {}, {}, {}
    for rk in sorted({k for sh in shards for k in __ks(sh)}):
        if rk.startswith("mtp."):
            continue
        m = lw._EXPERT_RE.match(rk)
        if m:
            layer, exp, which = int(m.group(1)), int(m.group(2)), m.group(3)
            if layer >= N_LAYERS or exp >= N_EXPERTS:
                stats["skipped_extra"].append(f"exp {layer}/{exp}")
                continue
            expert_pack.setdefault((layer, exp), {})[which] = rk
            continue
        m = lw._GATE_RE.match(rk)
        if m:
            layer = int(m.group(1))
            if layer >= N_LAYERS:
                stats["skipped_extra"].append(rk)
                continue
            which = m.group(2)
            tgt = dt(
                f"model.layers.{layer}.mlp.gate." + ("weight" if which == "weight" else "e_score_correction_bias")
            )
            if tgt not in T:
                stats["skipped_extra"].append(rk)
                continue
            gate_crop[rk] = tgt
            continue
        m = lw._SHARED_RE.match(rk)
        if m:
            layer = int(m.group(1))
            if layer >= N_LAYERS:
                stats["skipped_extra"].append(rk)
                continue
            proj = {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[m.group(2)]
            tgt = dt(f"model.layers.{layer}.mlp.shared_experts.{proj}.weight")
            if tgt not in T:
                stats["skipped_extra"].append(rk)
                continue
            shared_crop[rk] = tgt
            continue
        tgt = lw._translate_key(rk)
        if tgt is None:
            stats["skipped_extra"].append(rk)
            continue
        layer = int(rk.split(".")[1]) if rk.startswith("layers.") else -1
        if layer >= N_LAYERS:
            stats["skipped_extra"].append(rk)
            continue
        if ".attn.indexer." in rk and ".self_attn.indexer." in tgt:
            suffix = tgt.split(".self_attn.indexer.", 1)[1]
            tgt = f"model.layers.{layer}.self_attn.compressor.indexer.{suffix}"
        tgt = dt(tgt)
        if tgt not in T:
            stats["skipped_extra"].append(rk)
            continue
        direct[rk] = tgt

    for sh in shards:
        with safe_open(sh, framework="pt", device="cpu") as sf:
            for rk in sf.keys():
                tgt = direct.get(rk) or gate_crop.get(rk) or shared_crop.get(rk)
                if tgt is None:
                    continue
                tensor = sf.get_tensor(rk)
                tt = state[tgt]
                if tuple(tensor.shape) == tuple(tt.shape):
                    state[tgt] = tensor.to(device=device, dtype=tt.dtype)
                    stats["loaded"] += 1
                elif tensor.dim() == tt.dim() and all(a >= b for a, b in zip(tensor.shape, tt.shape)):
                    sl = tuple(slice(0, b) for b in tt.shape)
                    state[tgt] = tensor[sl].to(device=device, dtype=tt.dtype)
                    stats["loaded"] += 1
                else:
                    stats["skipped_shape_mismatch"].append((rk, tuple(tensor.shape), tuple(tt.shape)))

    # expert packing
    if expert_pack:
        need = {}
        for (_l, _e), wm in expert_pack.items():
            for w, k in wm.items():
                need[k] = w
        tensors = {}
        for sh in shards:
            with safe_open(sh, framework="pt", device="cpu") as sf:
                for rk in sf.keys():
                    if rk in need:
                        tensors[rk] = sf.get_tensor(rk)
        n_packed = 0
        for (layer, exp), wm in expert_pack.items():
            gu_key_l = dt(f"model.layers.{layer}.mlp.experts.gate_up_proj")
            dp_key_l = dt(f"model.layers.{layer}.mlp.experts.down_proj")
            if "w1" in wm and "w3" in wm and wm["w1"] in tensors and wm["w3"] in tensors:
                gu = torch.cat([tensors[wm["w1"]], tensors[wm["w3"]]], dim=0).to(dtype=state[gu_key_l].dtype)
                state[gu_key_l][exp] = gu
                n_packed += 1
            if "w2" in wm and wm["w2"] in tensors:
                state[dp_key_l][exp] = tensors[wm["w2"]].to(dtype=state[dp_key_l].dtype)
                n_packed += 1
        stats["packed"] = n_packed
        stats["loaded"] += n_packed

    stats["missing_final"] = sorted(T - set(state.keys()))
    return state, stats


def load_deepseek_v4_weights_sharded(model, src_path, model_config=None, strict=False):
    """Fill the local shards of a wrapped + materialized meta DeepSeek-V4 model
    from the native full-checkpoint safetensors, without any whole-model host
    build (the FSDPTurbo train actor calls this instead of the legacy full-CPU-
    build + _pad_load_state path).

    - non-expert params/buffers (attn / norms / compressors / embed / head /
      shared experts / router): read fully once per key (only a few GiB) and
      re-distribute onto each param's own (mesh, placements) via
      distribute_tensor, then copy into the local shard.
    - per-expert MoE weights (mlp.experts.*): native checkpoint stores one
      tensor per expert (layers.N.ffn.experts.K.w1/w2/w3), so only the local
      expert block (EP Shard(0): offset..offset+local_rows) is read and packed
      (gate||up) per expert; rank memory stays O(local shard).
    """
    import glob as _glob
    import os as _os
    import re as _re

    import torch
    from examples.deepseekv4 import load_weights as lw
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor, distribute_tensor

    state = model.state_dict()  # DTensor-valued local state dict
    T = set(state.keys())

    def _shape(k):
        v = state[k]
        return tuple(v.shape)  # DTensor.shape is the GLOBAL shape

    shp = {k: _shape(k) for k in T}
    _layers = [int(m.group(1)) for k in T for m in [_re.fullmatch(r"model\.(?:model\.)?layers\.(\d+)\..+", k)] if m]
    assert _layers, "no model.layers.* keys in target state dict"
    N_LAYERS = 1 + max(_layers)
    _gu = [k for k in T if _re.fullmatch(r"model\.(?:model\.)?layers\.0\.mlp\.experts\.gate_up_proj", k)]
    assert _gu, "model.layers.0.mlp.experts.gate_up_proj missing"
    N_EXPERTS = int(shp[_gu[0]][0])
    _cfg_budget = _budget_from_config(model_config)
    logger.info(
        f"budget from config: layers={_cfg_budget['N_LAYERS']} "
        f"experts={_cfg_budget['N_EXPERTS']} hash_moe={_cfg_budget['n_hash']}"
    )
    _check_budget(model_config, N_LAYERS, N_EXPERTS, strict=strict)

    def dt(single):
        return "model." + single

    shards = sorted(_glob.glob(_os.path.join(src_path, "*.safetensors")))
    rk_file = {k: sh for sh in shards for k in __ks(sh)}

    small_src = {}  # native rk -> adapter tgt (read fully, small payload)
    exp_src = {}  # (layer, exp, "w1"|"w2"|"w3") -> native rk
    skipped_extra, mismatches = [], []
    for rk in sorted(rk_file):
        if rk.startswith("mtp."):
            continue
        m = lw._EXPERT_RE.match(rk)
        if m:
            layer, exp, which = int(m.group(1)), int(m.group(2)), m.group(3)
            if layer >= N_LAYERS or exp >= N_EXPERTS:
                skipped_extra.append(rk)
                continue
            exp_src[(layer, exp, which)] = rk
            continue
        m = lw._GATE_RE.match(rk)
        if m:
            layer = int(m.group(1))
            if layer >= N_LAYERS:
                skipped_extra.append(rk)
                continue
            tgt = dt(
                f"model.layers.{layer}.mlp.gate." + ("weight" if m.group(2) == "weight" else "e_score_correction_bias")
            )
            if tgt in T:
                small_src[rk] = tgt
            else:
                skipped_extra.append(rk)
            continue
        m = lw._SHARED_RE.match(rk)
        if m:
            layer = int(m.group(1))
            if layer >= N_LAYERS:
                skipped_extra.append(rk)
                continue
            proj = {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[m.group(2)]
            tgt = dt(f"model.layers.{layer}.mlp.shared_experts.{proj}.weight")
            if tgt in T:
                small_src[rk] = tgt
            else:
                skipped_extra.append(rk)
            continue
        tgt = lw._translate_key(rk)
        if tgt is None:
            skipped_extra.append(rk)
            continue
        layer = int(rk.split(".")[1]) if rk.startswith("layers.") else -1
        if layer >= N_LAYERS:
            skipped_extra.append(rk)
            continue
        if ".attn.indexer." in rk and ".self_attn.indexer." in tgt:
            tgt = f"model.layers.{layer}.self_attn.compressor.indexer." + tgt.split(".self_attn.indexer.", 1)[1]
        tgt = dt(tgt)
        if tgt in T:
            small_src[rk] = tgt
        else:
            skipped_extra.append(rk)

    all_params = dict(model.named_parameters())
    all_params.update(dict(model.named_buffers()))
    required = {name for name, _ in model.named_parameters() if ".mlp.experts." not in name}
    missing = sorted(required.difference(small_src.values()))
    if missing:
        raise ValueError(f"DeepSeek-V4 missing checkpoint tensors: {missing}")
    loaded = 0

    def _scatter(target_key, full):
        """Distribute a full CPU tensor onto the target's own layout and copy
        it into the local shard.

        Returns True on success.
        """
        nonlocal loaded
        pv = all_params[target_key]
        pd = pv.data if hasattr(pv, "data") else pv
        if isinstance(pd, DTensor):
            d = distribute_tensor(full.contiguous(), pd.device_mesh, pd.placements)
            pd._local_tensor.copy_(d.to_local())
        else:
            pd.copy_(full)
        loaded += 1

    for rk in sorted(small_src):
        tgt = small_src[rk]
        pv = all_params.get(tgt)
        if pv is None:
            skipped_extra.append(rk)
            continue
        pd = pv.data if hasattr(pv, "data") else pv
        target_shape = tuple(pd.shape)
        with safe_open(rk_file[rk], framework="pt", device="cpu") as sf:
            full = sf.get_tensor(rk)
        if tuple(full.shape) == target_shape:
            _scatter(tgt, full)
        elif full.dim() == len(target_shape) and all(a >= b for a, b in zip(full.shape, target_shape)):
            _scatter(tgt, full[tuple(slice(0, b) for b in target_shape)])
        else:
            mismatches.append((rk, tuple(full.shape), target_shape))

    if mismatches:
        raise ValueError(f"DeepSeek-V4 incompatible checkpoint tensors: {mismatches}")

    # per-expert weights: local block only
    params = dict(model.named_parameters())
    packed = 0
    open_handles = {}

    def _open(sh):
        h = open_handles.get(sh)
        if h is None:
            h = safe_open(sh, framework="pt", device="cpu")
            open_handles[sh] = h
        return h

    for layer in range(N_LAYERS):
        for proj in ("gate_up_proj", "down_proj"):
            tgt = dt(f"model.layers.{layer}.mlp.experts.{proj}")
            p = params.get(tgt)
            if p is None:
                continue
            pd = p.data
            if not isinstance(pd, DTensor):
                continue  # experts not EP-distributed on this rank
            indices = _compose_dtensor_local_indices(pd.shape, pd.device_mesh, pd.placements)
            local_rows = len(indices[0])
            local_t = torch.empty(tuple(pd._local_tensor.shape), dtype=pd.dtype)
            for i, expert in enumerate(indices[0]):
                projections = ("w1", "w3") if proj == "gate_up_proj" else ("w2",)
                tensors = []
                for projection in projections:
                    key = exp_src.get((layer, expert, projection))
                    if key is None:
                        raise RuntimeError(
                            f"Missing expert weight: layer={layer}, expert={expert}, projection={projection}"
                        )
                    tensor = _open(rk_file[key]).get_tensor(key)
                    expected_shape = (pd.shape[1] // 2, pd.shape[2]) if proj == "gate_up_proj" else tuple(pd.shape[1:])
                    if tuple(tensor.shape) != expected_shape:
                        raise RuntimeError(
                            f"Invalid expert shape: layer={layer}, expert={expert}, projection={projection}, "
                            f"expected={expected_shape}, got={tuple(tensor.shape)}"
                        )
                    tensors.append(tensor)
                tensor = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
                for dim, selected in enumerate(indices[1:]):
                    tensor = tensor.index_select(dim, torch.tensor(list(selected), dtype=torch.long))
                local_t[i] = tensor.to(dtype=pd.dtype)
            pd._local_tensor.copy_(local_t)
            packed += local_rows
    loaded += packed
    logger.info(
        f"deepseek_v4 sharded weights loaded: loaded={loaded} packed={packed} "
        f"skipped_extra={len(skipped_extra)} mismatches={len(mismatches)}"
    )
    return {"loaded": loaded, "packed": packed, "skipped_extra": skipped_extra, "mismatches": mismatches}


def _compose_dtensor_local_indices(shape, device_mesh, placements) -> list:
    """Return the global indices of each dimension owned by this rank.

    ``distribute_tensor`` applies placements left to right, and a
    ``_StridedShard`` already encodes the composed right-to-left sharding
    internally.  Composing in reverse scrambles same-dimension stacks such as
    TP+FSDP or VP+VFSDP: values stay inside the checkpoint but land on the
    wrong rows, which is silent and grows with the layer count.
    """
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.placement_types import _StridedShard

    indices = [range(size) for size in shape]
    coordinate = device_mesh.get_coordinate()
    for mesh_dim in range(len(placements)):
        placement = placements[mesh_dim]
        if isinstance(placement, Replicate):
            continue
        if not isinstance(placement, (Shard, _StridedShard)):
            raise RuntimeError(f"Unsupported Qwen4Exp placement: {placement}")
        dim = placement.dim if placement.dim >= 0 else placement.dim + len(indices)
        current = indices[dim]
        rank = coordinate[mesh_dim]
        num_chunks = device_mesh.size(mesh_dim)
        if isinstance(placement, _StridedShard):
            # A strided shard is a regular rank shard inside each of the
            # split_factor outer chunks.  Compose two regular shard offset
            # calculations instead of relying on the newer ALL-offset API.
            selected = []
            for split_index in range(placement.split_factor):
                outer_size, outer_offset = _local_shard_size_and_offset(
                    Shard, len(current), placement.split_factor, split_index
                )
                inner_size, inner_offset = _local_shard_size_and_offset(Shard, outer_size, num_chunks, rank)
                start = outer_offset + inner_offset
                selected.extend(current[start : start + inner_size])
        else:
            local_size, offset = _local_shard_size_and_offset(placement, len(current), num_chunks, rank)
            selected = [] if local_size == 0 else current[offset : offset + local_size]
        indices[dim] = selected
    return indices


def load_qwen4_exp_weights_sharded(model: torch.nn.Module, src_path: str, strict: bool = True) -> dict:
    """Load Qwen4Exp checkpoint data directly into wrapped local shards.

    The model must already have been built on meta, wrapped by FSDPTurbo and
    materialized with ``to_empty``.  Safetensors' slice API keeps both host
    memory and IO proportional to the local shard instead of the global tensor.
    """
    import glob
    import json
    import os

    import torch
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor, distribute_tensor
    from torch.distributed.tensor.placement_types import _StridedShard

    if os.path.isdir(src_path):
        index_path = os.path.join(src_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                weight_map = {key: os.path.join(src_path, shard) for key, shard in json.load(f)["weight_map"].items()}
        else:
            shards = sorted(glob.glob(os.path.join(src_path, "*.safetensors")))
            weight_map = {}
            for shard in shards:
                with safe_open(shard, framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        weight_map[key] = shard
    else:
        with safe_open(src_path, framework="pt", device="cpu") as sf:
            weight_map = {key: src_path for key in sf.keys()}

    state = model.state_dict()
    skipped_ngram = []
    missing_mtp = []
    missing = []
    targets = {}
    for target_key, target in state.items():
        # Both VP and non-VP PLE tables are materialized by their dedicated
        # loading pass from ngram_embedding.shard_* keys. Their config buffers
        # are also restored separately after to_empty().
        if target_key.endswith("ngram_embedding.weight"):
            prefix = target_key.removesuffix(".weight") + ".shard_"
            exported_table = target_key in weight_map or os.path.isfile(
                os.path.join(src_path, "ngram_shards", "meta.json")
            )
            if exported_table and not (
                os.path.isfile(os.path.join(src_path, "model.safetensors.index.json"))
                and any(key.startswith(prefix) for key in weight_map)
            ):
                raise ValueError(
                    "PLE HF initialization requires indexed ngram_embedding.shard_* weights; "
                    "whole-table/VP checkpoint exports are unsupported. Use --load for checkpoint resume."
                )
            skipped_ngram.append(target_key)
            continue

        source_key = target_key
        if source_key not in weight_map and target_key == "lm_head.weight":
            source_key = "model.language_model.embed_tokens.weight"
        if source_key not in weight_map:
            if target_key.startswith("mtp."):
                missing_mtp.append(target_key)
            else:
                missing.append(target_key)
            continue
        targets[target_key] = (source_key, target)

    if strict and missing:
        raise KeyError(f"Qwen4Exp checkpoint is missing {len(missing)} target keys: {missing[:20]}")

    if missing_mtp:
        # from_pretrained used to initialize MTP weights absent from the base
        # checkpoint. Initialize only their owning modules, and teach HF's
        # init.copy_ to shard plain tensors before copying into DTensors.
        from transformers import initialization as hf_init

        original_copy = hf_init.copy_

        def _dtensor_aware_copy(tensor, other):
            if isinstance(tensor, DTensor) and not isinstance(other, DTensor):
                other = distribute_tensor(other, tensor.device_mesh, tensor.placements)
            return original_copy(tensor, other)

        hf_init.copy_ = _dtensor_aware_copy
        try:
            module_names = dict.fromkeys(target_key.rsplit(".", 1)[0] for target_key in missing_mtp)
            for module_name in module_names:
                model._init_weights(model.get_submodule(module_name))
        finally:
            hf_init.copy_ = original_copy

    by_shard = {}
    for target_key, (source_key, target) in targets.items():
        by_shard.setdefault(weight_map[source_key], []).append((target_key, source_key, target))

    def _copy_indexed_slice(source_slice, local_tensor, indices):
        """Copy Cartesian products of contiguous source runs into a non-
        contiguous local shard."""
        import itertools

        runs_by_dim = []
        for dim_indices in indices:
            runs = []
            run_start = 0
            for pos in range(1, len(dim_indices) + 1):
                if pos == len(dim_indices) or dim_indices[pos] != dim_indices[pos - 1] + 1:
                    runs.append((dim_indices[run_start], dim_indices[pos - 1] + 1, run_start, pos))
                    run_start = pos
            runs_by_dim.append(runs)
        for runs in itertools.product(*runs_by_dim):
            source_region = tuple(slice(source_start, source_stop) for source_start, source_stop, _, _ in runs)
            local_region = tuple(slice(local_start, local_stop) for _, _, local_start, local_stop in runs)
            local_tensor[local_region].copy_(source_slice[source_region])

    loaded = 0
    mismatches = []
    with torch.no_grad():
        for shard, items in by_shard.items():
            with safe_open(shard, framework="pt", device="cpu") as sf:
                for target_key, source_key, target in items:
                    source_slice = sf.get_slice(source_key)
                    source_shape = tuple(source_slice.get_shape())
                    target_shape = tuple(target.shape)
                    if len(source_shape) != len(target_shape) or any(
                        source_dim < target_dim for source_dim, target_dim in zip(source_shape, target_shape)
                    ):
                        mismatches.append((source_key, source_shape, target_shape))
                        continue

                    if isinstance(target, DTensor) and any(
                        isinstance(placement, _StridedShard) for placement in target.placements
                    ):
                        local_tensor = target._local_tensor
                        indices = _compose_dtensor_local_indices(target.shape, target.device_mesh, target.placements)
                        if tuple(len(dim_indices) for dim_indices in indices) != tuple(local_tensor.shape):
                            raise RuntimeError(f"Qwen4Exp local index shape mismatch for {target_key}")
                        _copy_indexed_slice(source_slice, local_tensor, indices)
                    else:
                        if isinstance(target, DTensor):
                            from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

                            local_shape, global_offset = compute_local_shape_and_global_offset(
                                target_shape, target.device_mesh, target.placements
                            )
                            local_tensor = target._local_tensor
                            region = tuple(
                                slice(offset, offset + size) for offset, size in zip(global_offset, local_shape)
                            )
                        else:
                            local_tensor = target
                            region = tuple(slice(0, size) for size in target_shape)
                        value = source_slice[region] if region else source_slice[()]
                        if tuple(value.shape) != tuple(local_tensor.shape):
                            raise RuntimeError(
                                f"Qwen4Exp local slice shape mismatch for {target_key}: "
                                f"source {tuple(value.shape)}, target {tuple(local_tensor.shape)}"
                            )
                        local_tensor.copy_(value)
                    loaded += 1
                    if loaded % 100 == 0:
                        logger.info(f"[qwen4_exp] loaded {loaded}/{len(targets)} local weight shards")

    if mismatches:
        raise ValueError(f"Qwen4Exp incompatible checkpoint tensors: {mismatches[:20]}")
    logger.info(
        f"[qwen4_exp] sharded weights loaded: loaded={loaded} missing={len(missing)} "
        f"initialized_mtp={len(missing_mtp)} skipped_ngram={len(skipped_ngram)} mismatches=0"
    )
    return {
        "loaded": loaded,
        "missing": missing,
        "missing_mtp": missing_mtp,
        "skipped_ngram": skipped_ngram,
        "mismatches": mismatches,
    }


def __ks(shard_path):
    from safetensors import safe_open

    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        return list(sf.keys())


def load_deepseek_v4_model(
    model_path: str, torch_dtype: torch.dtype, move_to_device: bool = True, build_mode: str = "auto"
):
    """Build the reduced 4-layer/16-expert DeepSeek-V4 model.

    Load novamlx weights when ``model_path`` contains safetensors; otherwise
    keep random initialization.
    """
    import os

    import torch
    from fsdp_turbo.models.deepseek.deepseek_v4 import (
        build_deepseek_v4_model,
        build_deepseek_v4_model_args,
        build_deepseek_v4_model_meta,
        prepare_deepseek_v4_model_for_fsdp,
    )
    from transformers import AutoConfig
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    # Model shape is driven by <model_path>/config.json when present (restores
    # num_hidden_layers / n_routed_experts / compress_ratios / mlp_layer_types
    # per the FSDPTurbo adapter notes — e.g. a 5L/256E truncated profile);
    # otherwise fall back to the adapter's stock 4L/16E reference profile.
    model_args = None
    if model_path and os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if isinstance(hf_config, DeepseekV4Config):
            model_args = hf_config
    if model_args is None:
        model_args = build_deepseek_v4_model_args()
    _mlt = getattr(model_args, "mlp_layer_types", None)
    if _mlt is not None and len(_mlt) != model_args.num_hidden_layers:
        raise ValueError(
            f"config inconsistency: mlp_layer_types length {len(_mlt)} != "
            f"num_hidden_layers {model_args.num_hidden_layers}"
        )
    logger.info(
        f"deepseek_v4 build profile: layers={model_args.num_hidden_layers} experts={model_args.n_routed_experts}"
    )
    has_weights = bool(
        model_path
        and os.path.isdir(model_path)
        and any(fname.endswith(".safetensors") for fname in os.listdir(model_path))
    )
    if build_mode != "full":
        # Default path: meta skeleton for every workload (random init or real
        # weights). Real weights are filled after the FSDP/EP wrap by
        # load_deepseek_v4_weights_sharded (per-rank local-shard reads);
        # "full" keeps the legacy whole-model CPU build + _pad_load_state.
        model = build_deepseek_v4_model_meta(model_args=model_args)
        model._built_on_meta = True
        if has_weights:
            model._weights_path = model_path
            model._weights_config = model_args
        logger.info(
            f"deepseek_v4 meta skeleton built: layers={model_args.num_hidden_layers} "
            f"experts={model_args.n_routed_experts} "
            f"{'real weights (sharded load after wrap)' if has_weights else 'random init'}"
        )
    else:
        model = build_deepseek_v4_model(model_args=model_args)  # BF16, stock official init
        if has_weights:
            state, stats = _pad_load_state(model, model_path, device="cpu", model_config=model_args)
            model.load_state_dict(state, strict=False)
            # DeepSeek-V4: hash-MoE token->expert map comes from the official
            # 256-expert world; clamp out-of-range expert ids to 1 (same behavior
            # as native train_with_weights.py's tid2eid clip) so the 16-expert
            # build stays valid when gathering experts.
            clamped = 0
            for bn, b in model.named_buffers():
                if bn.endswith("gate.tid2eid"):
                    over = b.data >= model_args.n_routed_experts
                    n_over = int(over.sum())
                    if n_over:
                        b.data[over] = 1
                        clamped += n_over
            logger.info(f"deepseek_v4 tid2eid clamped={clamped}")

            logger.info(
                f"deepseek_v4 real weights loaded: "
                f"loaded={stats['loaded']} packed={stats['packed']} "
                f"missing={len(stats['missing_final'])} extra_skipped={len(stats['skipped_extra'])}"
            )
            if stats.get("skipped_shape_mismatch"):
                logger.warning(f"deepseek_v4 shape mismatches: {stats['skipped_shape_mismatch']}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        import torch_npu  # noqa: F401

        device = torch.device("npu")
    return prepare_deepseek_v4_model_for_fsdp(
        model, device=device, parameter_dtype=torch_dtype, move_to_device=move_to_device
    )


def _is_qwen4_exp(hf_config) -> bool:
    """Qwen3.8-Flash-Next (qwen4_exp) detection — plain or composite config."""
    for candidate in (hf_config, getattr(hf_config, "text_config", None)):
        if getattr(candidate, "model_type", "") == "qwen4_exp":
            return True
    return False


def _is_deepseek_v4(hf_config) -> bool:
    """DeepSeek-V4 (deepseek_v4) detection — plain or composite config."""
    for candidate in (hf_config, getattr(hf_config, "text_config", None)):
        if getattr(candidate, "model_type", "") == "deepseek_v4":
            return True
    return False


def _apply_param_freeze(model: torch.nn.Module, args: Namespace) -> None:
    """Apply parameter freeze regexes to HF names before FSDP2 sharding.

    FSDP2 shards parameters individually, but ``requires_grad`` must be settled
    before gradient machinery is built.
    """
    only_train = getattr(args, "only_train_params_name_list", None)
    freeze = getattr(args, "freeze_params_name_list", None)
    if not only_train and not freeze:
        return

    if only_train:
        for name, param in model.named_parameters():
            param.requires_grad = False
            for pattern in only_train:
                if re.search(pattern, name):
                    param.requires_grad = True
                    break
    if freeze:
        for name, param in model.named_parameters():
            for pattern in freeze:
                if re.search(pattern, name):
                    param.requires_grad = False
                    break

    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen = [name for name, param in model.named_parameters() if not param.requires_grad]
    logger.info(
        f"[fsdp_turbo] param freeze applied (only-train patterns={only_train or []}, "
        f"freeze patterns={freeze or []}): {len(frozen)} frozen / {len(trainable)} trainable "
        f"params; frozen sample: {frozen[:5]}"
    )


def _exclude_frozen_visual_from_fsdp(model: torch.nn.Module, fsdp_plan: Any) -> bool:
    """Keep optional, fully frozen vision forwards free of FSDP collectives."""
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None or any(param.requires_grad for param in visual.parameters()):
        return False
    ignored_modules = list(fsdp_plan.ignored_modules or [])
    for pattern in ("model.visual", "model.visual.*"):
        if pattern not in ignored_modules:
            ignored_modules.append(pattern)
    fsdp_plan.ignored_modules = ignored_modules
    return True


def _materialize_frozen_visual(model: torch.nn.Module, fsdp_plan: Any, master_dtype: torch.dtype) -> None:
    """Place ignored vision weights on the compute device after checkpoint
    load."""
    from fsdp_turbo.utils.dtype import get_dtype

    visual = model.get_submodule("model.visual")
    if any(param.requires_grad for param in visual.parameters()):
        raise RuntimeError("Replicated vision weights must be fully frozen.")
    compute_dtype = get_dtype(fsdp_plan.param_dtype) if fsdp_plan.param_dtype else master_dtype
    visual.to(device=make_current_torch_device(), dtype=compute_dtype)
    logger.info("[qwen4_exp] fully frozen visual replicated on compute device, excluded from FSDP collectives")


def _pin_fp_state_to_compute_dtype(model: torch.nn.Module, turbo_config, master_dtype: torch.dtype) -> None:
    """Pin floating-point buffers to the FSDP2 compute dtype.

    FSDP2 mixed precision re-casts managed parameters only; buffers retain the
    master dtype. Managed parameters must keep a uniform original dtype, so
    they are not pinned here. Qwen4 text rotary frequencies stay in FP32 by
    default; visual frequencies follow Megatron's BF16 path.
    """
    from fsdp_turbo.utils.dtype import get_dtype

    fsdp_plan = getattr(getattr(turbo_config, "distributed", None), "fsdp_plan", None)
    param_dtype_str = getattr(fsdp_plan, "param_dtype", None) if fsdp_plan is not None else None
    compute_dtype = get_dtype(param_dtype_str) if param_dtype_str else None
    if compute_dtype is None or compute_dtype == master_dtype:
        return

    rotary_buffer_ids = set()
    if Envs.RELAX_SFT_ROPE_FP32:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

        for module in model.modules():
            if isinstance(module, Qwen4ExpTextRotaryEmbedding):
                rotary_buffer_ids.update(id(buffer) for buffer in module.buffers(recurse=False))

    pinned_buffers = []
    preserved_rotary_buffers = []
    for name, buffer in model.named_buffers():
        if id(buffer) in rotary_buffer_ids:
            buffer.data = buffer.data.float()
            preserved_rotary_buffers.append(name)
            continue
        if buffer is not None and buffer.is_floating_point() and buffer.dtype != compute_dtype:
            buffer.data = buffer.data.to(compute_dtype)
            pinned_buffers.append(name)

    logger.info(
        f"[fsdp_turbo] master dtype {master_dtype} != compute dtype {compute_dtype}: pinned "
        f"{len(pinned_buffers)} floating-point buffers (sample: {pinned_buffers[:5]}) to {compute_dtype}"
    )
    if preserved_rotary_buffers:
        logger.info(f"[SFT RoPE FP32] preserved frequency buffers: {preserved_rotary_buffers}")


def _pin_ngram_tables_to_bf16(model: torch.nn.Module) -> None:
    """Re-pin CPU-offloaded PLE n-gram tables to bf16 after materialization.

    Shared-memory loading can give non-source ranks the fp32 placeholder dtype;
    these frozen FSDP-ignored tables must not promote activations.
    """
    from fsdp_turbo.models.qwen.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    fixed = []
    for name, module in model.named_modules():
        if not isinstance(module, Qwen4ExpTextNGramEmbedding):
            continue
        weight = module.ngram_embedding.weight
        if weight is not None and weight.is_floating_point() and weight.dtype != torch.bfloat16:
            module.ngram_embedding.weight.data = weight.data.to(torch.bfloat16)
            fixed.append(f"{name} ({weight.dtype} -> torch.bfloat16)")
    if fixed:
        logger.warning(f"[fsdp_turbo] n-gram table dtype re-pinned after materialize: {fixed}")


def _megatron_qwen4_exp_vision_rope_forward(
    module: torch.nn.Module,
    position_ids: torch.Tensor,
    new_position_ids: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Use BF16 angle products and FP32 trig inputs, matching Megatron."""
    if new_position_ids is not None:
        position_ids = new_position_ids
    angles = (position_ids.unsqueeze(-1) * module.inv_freq).flatten(1).float()
    with torch.autocast(device_type=angles.device.type, enabled=False):
        emb = torch.cat((angles, angles), dim=-1)
        module._megatron_position_embeddings = (emb.cos(), emb.sin())
    if new_position_ids is not None:
        return module._megatron_position_embeddings
    return angles


def _make_megatron_qwen4_exp_vision_block_hook(rotary_pos_emb: torch.nn.Module) -> Callable:
    def _hook(_module, args, kwargs):
        kwargs["position_embeddings"] = rotary_pos_emb._megatron_position_embeddings
        return args, kwargs

    return _hook


class _MegatronQwen4ExpRowParallelLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        group,
    ) -> torch.Tensor:
        ctx.save_for_backward(input_, weight, bias)
        tp_rank = dist.get_rank(group=group)
        shard_size = weight.shape[1] // dist.get_world_size(group=group)
        input_shard = input_.narrow(-1, tp_rank * shard_size, shard_size)
        weight_shard = weight.narrow(1, tp_rank * shard_size, shard_size)
        output = torch.matmul(input_shard, weight_shard.t()).contiguous()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        output = output + bias if bias is not None else output
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_, weight, bias = ctx.saved_tensors
        grad_input = torch.matmul(grad_output, weight) if ctx.needs_input_grad[0] else None
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(
                grad_output.reshape(-1, grad_output.shape[-1]).t(), input_.reshape(-1, input_.shape[-1])
            )
        grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None


def _megatron_qwen4_exp_vision_row_parallel_forward(module: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
    return _MegatronQwen4ExpRowParallelLinear.apply(
        input_,
        module.weight,
        module.bias,
        module._megatron_tp_group,
    )


def _megatron_qwen4_exp_vision_column_parallel_forward(module: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
    """Match Megatron's per-TP-rank output-channel GEMM geometry."""
    tp_size = dist.get_world_size(group=module._megatron_tp_group)
    shard_size = module.weight.shape[0] // tp_size
    outputs = []
    for tp_rank in range(tp_size):
        start = tp_rank * shard_size
        weight_shard = module.weight.narrow(0, start, shard_size)
        output = torch.matmul(input_, weight_shard.t())
        if module.bias is not None:
            output = output + module.bias.narrow(0, start, shard_size)
        outputs.append(output)
    return torch.cat(outputs, dim=-1)


def _restore_qwen4_exp_nonpersistent_buffers(model: torch.nn.Module, device: str) -> None:
    """Recompute rotary buffers discarded by meta construction/to_empty()."""
    from fsdp_turbo.distributed.parallel_state import get_parallel_state
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRotaryEmbedding,
        Qwen4ExpVisionModel,
        Qwen4ExpVisionRotaryEmbedding,
    )

    restored = []
    for name, module in model.named_modules():
        if isinstance(module, Qwen4ExpTextRotaryEmbedding):
            reference = Qwen4ExpTextRotaryEmbedding(module.config, device="cpu")
        elif isinstance(module, Qwen4ExpVisionRotaryEmbedding):
            if hasattr(module, "config"):
                reference = Qwen4ExpVisionRotaryEmbedding(module.config)
            else:
                reference = Qwen4ExpVisionRotaryEmbedding(module.dim, module.theta)
        else:
            continue
        for buffer_name, buffer in module.named_buffers(recurse=False):
            source = getattr(reference, buffer_name, None)
            if source is not None:
                if isinstance(module, Qwen4ExpVisionRotaryEmbedding) and buffer_name == "inv_freq":
                    buffer.data = source.to(device=buffer.device, dtype=torch.bfloat16)
                elif (
                    Envs.RELAX_SFT_ROPE_FP32
                    and isinstance(module, Qwen4ExpTextRotaryEmbedding)
                    and buffer_name in ("inv_freq", "original_inv_freq")
                ):
                    # Widening an already rounded BF16 buffer cannot recover
                    # its frequencies. Rebuild from the FP32 reference instead.
                    buffer.data = source.to(device=buffer.device, dtype=torch.float32)
                else:
                    buffer.data.copy_(source.to(dtype=buffer.dtype))
        if isinstance(module, Qwen4ExpVisionRotaryEmbedding):
            module.forward = types.MethodType(_megatron_qwen4_exp_vision_rope_forward, module)
        # CPU offload keeps FSDP master params on CPU, but non-persistent
        # compute buffers must follow position_ids onto the accelerator.
        module.to(torch.device(device))
        restored.append(name)
    parallel_state = get_parallel_state()
    from .data_parallel import vision_tensor_parallel_group

    # FSDP ranks now consume different images. Reducing vision outputs over
    # FSDP/arbitrary contiguous groups would mix unrelated samples together.
    tp_group = vision_tensor_parallel_group(parallel_state)
    for module in model.modules():
        if isinstance(module, Qwen4ExpVisionModel):
            for block in module.blocks:
                block.register_forward_pre_hook(
                    _make_megatron_qwen4_exp_vision_block_hook(module.rotary_pos_emb), with_kwargs=True
                )
                if tp_group is not None:
                    block.attn.proj._megatron_tp_group = tp_group
                    block.attn.proj.forward = types.MethodType(
                        _megatron_qwen4_exp_vision_row_parallel_forward, block.attn.proj
                    )
                    block.mlp.linear_fc1._megatron_tp_group = tp_group
                    block.mlp.linear_fc1.forward = types.MethodType(
                        _megatron_qwen4_exp_vision_column_parallel_forward, block.mlp.linear_fc1
                    )
                    block.mlp.linear_fc2._megatron_tp_group = tp_group
                    block.mlp.linear_fc2.forward = types.MethodType(
                        _megatron_qwen4_exp_vision_row_parallel_forward, block.mlp.linear_fc2
                    )
            if tp_group is not None:
                module.merger.linear_fc2._megatron_tp_group = tp_group
                module.merger.linear_fc2.forward = types.MethodType(
                    _megatron_qwen4_exp_vision_row_parallel_forward, module.merger.linear_fc2
                )
    logger.info(f"[qwen4_exp] restored non-persistent rotary buffers on {len(restored)} modules")


def _build_qwen4_exp_turbo(args, turbo_config, torch_dtype):
    """Build and wrap Qwen3.8-Flash-Next with fsdp_turbo's consolidated
    modeling.

    Nested parent/child ``apply_modules`` require post-order sharding so
    children are sharded before parents.
    """
    from fsdp_turbo.fsdp_turbo import FSDPTurbo
    from fsdp_turbo.models.qwen.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration
    from fsdp_turbo.utils.model import convert_model_dtype, set_num_hidden_layers
    from transformers import AutoConfig

    model_path = args.hf_checkpoint
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    for cfg in (
        model_config,
        getattr(model_config, "text_config", None),
        getattr(model_config, "vision_config", None),
    ):
        if cfg is not None:
            cfg._attn_implementation = "sdpa"

    # Keep stock SDPA numerics in the vision tower under a dedicated registry name.
    if model_config.vision_config is not None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward as _orig_sdpa_attention
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _ATTN_REGISTRY

        def _vision_attention_local(*args, **kwargs):
            return _orig_sdpa_attention(*args, **kwargs)

        _ATTN_REGISTRY.register("sdpa_vision", _vision_attention_local)
        model_config.vision_config._attn_implementation = "sdpa_vision"

    set_num_hidden_layers(model_config, getattr(turbo_config.model, "num_hidden_layers", None))

    text_cfg = getattr(model_config, "text_config", model_config)
    vision_cfg = getattr(model_config, "vision_config", None)
    ple_layer_ids = getattr(turbo_config.model, "ple_layer_ids", None)
    if ple_layer_ids is not None:
        text_cfg.ple_layer_ids = list(ple_layer_ids)
        if not text_cfg.ple_layer_ids:
            text_cfg.number_of_conv_states = 1
        logger.info(f"[qwen4_exp] PLE layer override: {text_cfg.ple_layer_ids}")
    text_cfg.gdn_implementation = "triton"
    text_cfg.causal_conv1d_implementation = "triton"
    text_cfg.qsa_implementation = "triton"
    text_cfg.skip_gdn_recompute = True
    text_cfg.skip_flash_attn_recompute = True
    if vision_cfg is not None:
        vision_cfg.skip_flash_attn_recompute = True
    text_cfg.use_grouped_expert_matmul = True
    text_cfg._experts_implementation = "eager"
    model_config._experts_implementation = "eager"
    # Start with an empty PLE table. VP materializes and shards it before wrapping;
    # otherwise materialize frozen CPU weights after wrapping.
    vp_enabled = getattr(turbo_config.distributed, "vp_plan", None) is not None
    text_cfg.ple_embedding_on_cpu = True
    text_cfg._ngram_hf_dir = model_path
    # Allow functional and smoke tests to shrink the PLE table deterministically.
    ngram_vocab_size_base = os.environ.get("NGRAM_VOCAB_SIZE_BASE", "").strip()
    if ngram_vocab_size_base:
        text_cfg.ngram_vocab_size_base = int(ngram_vocab_size_base)
        if text_cfg.ngram_vocab_size_base <= 0:
            raise ValueError(f"Invalid NGRAM_VOCAB_SIZE_BASE='{ngram_vocab_size_base}'.")
    text_cfg.enable_chunk_loss = getattr(turbo_config.memory, "chunk_loss", False)
    text_cfg.sft_chunk_rows = turbo_config.memory.chunk_loss_size if text_cfg.enable_chunk_loss else 0
    text_cfg.enable_dynamic_chunk_loss = False
    text_cfg.enable_ep_balance = False
    text_cfg.max_dup_experts_num = 2
    text_cfg.mtp_num_layers = int(getattr(turbo_config.model, "mtp_num_layers", 0) or 0)

    from .qwen4_indexer import configure as configure_indexer
    from .qwen4_indexer import install as install_indexer

    configure_indexer(turbo_config, text_cfg)
    # Build only the meta skeleton here.  Materializing a complete model in
    # every rank before FSDP wrapping causes a ranks x full-model host-memory
    # peak, especially with fp32 master weights.
    with torch.device("meta"):
        model = Qwen4ExpForConditionalGeneration(model_config)
    if is_sft_mode(args):
        router_fp32 = Envs.RELAX_SFT_ROUTER_FP32
        patched_routers = enable_qwen4_router_fp32(model) if router_fp32 else 0
        logger.info(f"[qwen4_exp] router precision: fp32={router_fp32} patched_modules={patched_routers}")
    install_indexer(model, turbo_config)
    convert_model_dtype(model, turbo_config.model.torch_dtype)

    from fsdp_turbo.models.qwen.modeling_qwen4_exp import restore_ngram_integer_buffers

    restore_ngram_integer_buffers(model)

    _pin_fp_state_to_compute_dtype(model, turbo_config, torch_dtype)
    _apply_param_freeze(model, args)
    frozen_visual_replica = _exclude_frozen_visual_from_fsdp(model, turbo_config.distributed.fsdp_plan)

    # Clear the default ParallelState created by from_pretrained so FSDPTurbo uses the real plan.
    from fsdp_turbo.distributed.parallel_state import ParallelState, Singleton

    Singleton._instances.pop(ParallelState, None)

    from fsdp_turbo.distributed.fully_shard_parallel import fully_shard_parallel as _fsp

    _orig_get_fsdp_modules = _fsp.get_fsdp_modules

    def _get_fsdp_modules_post_order(model, fsdp_plan, ignored_modules):
        fsdp_modules = {}
        for name, module in _fsp._post_order_traverse(model):
            for pattern, plan in fsdp_plan.apply_modules.items():
                if _fsp.module_name_match(pattern, name) and name not in ignored_modules:
                    fsdp_modules.setdefault(module, {}).update(plan)
        if not fsdp_modules:
            raise RuntimeError(f"[FSDP2] No module named {list(fsdp_plan.apply_modules.keys())}.")
        return fsdp_modules

    _fsp.get_fsdp_modules = _get_fsdp_modules_post_order
    try:
        if vp_enabled:
            # Materialize the trainable table before VP shards its local slice onto the accelerator.
            from fsdp_turbo.models.qwen.modeling_qwen4_exp import materialize_ngram_embedding_on_cpu

            materialize_ngram_embedding_on_cpu(model, dtype=torch.bfloat16, hf_dir=model_path, freeze=False)
        turbo = FSDPTurbo(config=turbo_config, model=model)
    finally:
        _fsp.get_fsdp_modules = _orig_get_fsdp_modules

    # Materialize meta shards without replacing the already loaded VP table.
    # Restore non-persistent buffers separately because checkpoints omit them.
    turbo.model._apply(lambda tensor: torch.empty_like(tensor, device="cpu") if tensor.is_meta else tensor)
    _restore_qwen4_exp_nonpersistent_buffers(turbo.model, device="npu")
    load_qwen4_exp_weights_sharded(turbo.model, model_path)
    restore_ngram_integer_buffers(turbo.model)
    # The large ngram table stays on its configured CPU/VP path; only the small
    # hashing buffers participate in NPU integer arithmetic.  Restore them
    # after checkpoint loading so config-derived values cannot be overwritten.
    ngram_hash_buffer_names = {"layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"}
    for name, buffer in turbo.model.named_buffers():
        if name.rsplit(".", 1)[-1] in ngram_hash_buffer_names:
            buffer.data = buffer.data.to(torch.device("npu"))
    _pin_fp_state_to_compute_dtype(turbo.model, turbo_config, torch_dtype)
    # FSDP CPU offload requires master params to stay on CPU through lazy init;
    # the param group moves them to the compute device during forward.  Without
    # offload, mirror the dsv4 meta-loading path and materialize on NPU now.
    if not getattr(turbo_config.distributed.fsdp_plan, "cpu_offload", False):
        turbo.model.to(torch.device("npu"))

    if frozen_visual_replica:
        _materialize_frozen_visual(turbo.model, turbo_config.distributed.fsdp_plan, torch_dtype)

    # Without VP, materialize frozen tables after model weights leave CPU memory.
    if not vp_enabled:
        from fsdp_turbo.models.qwen.modeling_qwen4_exp import materialize_ngram_embedding_on_cpu

        materialize_ngram_embedding_on_cpu(turbo, dtype=torch.bfloat16, hf_dir=model_path, freeze=True)
        _pin_ngram_tables_to_bf16(turbo)

    logger.info("[qwen4_exp] meta model materialized with local shards and wrapped by FSDPTurbo")
    return turbo


def build_turbo_model(args: Namespace, turbo_config, hf_config) -> torch.nn.Module:
    """Build the configured model and wrap it with FSDPTurbo."""
    from .performance_config import validate_model_options

    validate_model_options(turbo_config.relax_performance, is_qwen4_exp=_is_qwen4_exp(hf_config))
    # MoonEP's NPU backend still uses this legacy accelerator API.
    if not hasattr(torch.accelerator, "current_device"):
        torch.accelerator.current_device = torch.accelerator.current_device_index
    from fsdp_turbo.fsdp_turbo import FSDPTurbo

    # Preserve a torch.dtype from the YAML loader; string-mapping it would incorrectly hit the bf16 fallback.
    raw_dtype = turbo_config.model.torch_dtype
    if isinstance(raw_dtype, torch.dtype):
        torch_dtype = raw_dtype
    else:
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(raw_dtype).lower(), torch.bfloat16)
    logger.info(f"[fsdp_turbo] model.torch_dtype={raw_dtype!r} -> master dtype {torch_dtype}")
    if _is_qwen4_exp(hf_config):
        turbo = _build_qwen4_exp_turbo(args, turbo_config, torch_dtype)
    elif _is_deepseek_v4(hf_config):
        # 5L/256E ≈ 67 GiB BF16 — larger than one 61 GiB NPU — so the full
        # model cannot be moved to the accelerator before FSDP sharding.
        # Keep it on CPU through the FSDP wrap; only the local shards
        # (≈4.2 GiB/rank) are moved to the NPU right after the wrap.
        model = load_deepseek_v4_model(args.hf_checkpoint, torch_dtype, move_to_device=False)
        _apply_param_freeze(model, args)
        turbo = FSDPTurbo(config=turbo_config, model=model)
        # Meta-deferred random-init build (5L/256E): the skeleton was built
        # on meta device and FSDP/EP-wrapped without materializing, so here
        # only the local shards (a few GiB/rank) are allocated and filled
        # with the stock official init rules — no full-model host build.
        if getattr(model, "_built_on_meta", False):
            from fsdp_turbo.models.deepseek.deepseek_v4 import (
                materialize_deepseek_v4_model,
            )

            turbo.model.to_empty(device="cpu")
            materialize_deepseek_v4_model(turbo.model)
            if getattr(model, "_weights_path", None):
                # real weights: fill the local shards directly (no full
                # 67 GiB CPU build + whole-model pad_load). Per-rank IO is
                # the local expert block + a few GiB of shared params.
                load_deepseek_v4_weights_sharded(
                    turbo.model,
                    model._weights_path,
                    model_config=getattr(model, "_weights_config", None),
                )
        turbo.model.to(torch.device("cuda" if torch.cuda.is_available() else "npu"))
    else:
        model = load_hf_model(args.hf_checkpoint, torch_dtype)
        _apply_param_freeze(model, args)
        turbo = FSDPTurbo(config=turbo_config, model=model)

    return turbo
