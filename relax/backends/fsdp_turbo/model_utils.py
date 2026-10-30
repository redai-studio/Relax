# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Model and loss helpers for HF models wrapped by FSDPTurbo.

GRPO shares ``flow_grpo`` clipping; SFT mirrors Megatron's masked NLL. With
Ulysses CP, callers feed complete batches while model forwards return local
sequence shards, and SFT statistics are reduced over the CP group.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional

from relax.models import flow_grpo
from relax.utils.logging_utils import get_logger


# FSDPTurbo clip-grads needs this API, which the NPU stack exposes only as
# ``current_device_index``.
if not hasattr(torch.accelerator, "current_device"):
    torch.accelerator.current_device = torch.accelerator.current_device_index  # type: ignore[attr-defined]


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


def build_microbatches(
    total_lengths: Sequence[int],
    max_tokens_per_gpu: int | None,
    micro_batch_size: int | None = None,
) -> list[list[int]]:
    """Greedy-pack sample indices under padded-token and sample-count budgets.

    Padded positions are ``rows * longest_sample``, not the sum of lengths.
    An over-budget sample still forms a one-sample microbatch.
    """
    if not total_lengths:
        return []
    budgets = [int(t) for t in total_lengths]
    microbatches: list[list[int]] = []
    current: list[int] = []
    current_max_len = 0
    for index, length in enumerate(budgets):
        over_budget = False
        if max_tokens_per_gpu and current:
            padded_positions = (len(current) + 1) * max(current_max_len, length)
            if padded_positions > max_tokens_per_gpu:
                over_budget = True
        if micro_batch_size and len(current) >= micro_batch_size:
            over_budget = True
        if over_budget:
            microbatches.append(current)
            current = []
            current_max_len = 0
        current.append(index)
        current_max_len = max(current_max_len, length)
    if current:
        microbatches.append(current)
    return microbatches


def _pad_batch(tokens: Sequence[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad per-sample 1-D token tensors into ``(input_ids,
    attn_mask)``."""
    max_len = max(int(t.numel()) for t in tokens)
    input_ids = torch.zeros(len(tokens), max_len, dtype=torch.long, device=device)
    attention_mask = torch.zeros(len(tokens), max_len, dtype=torch.long, device=device)
    for row, tensor in enumerate(tokens):
        length = int(tensor.numel())
        input_ids[row, :length] = tensor.to(device=device, dtype=torch.long)
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


_NLL_CHUNK_ROWS = 8192  # ~2.4 GiB transient at ~152k vocab (bf16) per chunk


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


def _cp_shard_targets(targets: torch.Tensor, group) -> torch.Tensor:
    """Right-pad the complete ``[B, T]`` targets to the CP-group size and
    split."""
    from fsdp_turbo.distributed.context_parallel.batch import (
        pad_tensor_to_group,
        split_tensor_to_group,
    )

    return split_tensor_to_group(pad_tensor_to_group(targets, group, dim=1, pad_value=-100), group, dim=1)


def _forward_token_nll(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
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
    forward_kwargs = {"use_cache": False, **model_kwargs}
    # MTP-patched models (Qwen3.8/qwen4_exp) consume the final hidden states
    # in their forward hook; without this the MTP loss silently never runs.
    inner = getattr(model, "model", model)
    if getattr(inner, "mtp", None) is not None:
        forward_kwargs["output_hidden_states"] = True
        if labels is not None:
            forward_kwargs["labels"] = labels
            forward_kwargs["mtp_only_labels"] = True
    logits = model(input_ids=input_ids, attention_mask=attention_mask, **forward_kwargs).logits

    ulysses_group, _ulysses_rank, ulysses_size = _ulysses_cp_context()
    if ulysses_size > 1:
        batch_size = input_ids.size(0)
        # Build global next-token targets before sharding; -100 ignores padding and the last position.
        targets = torch.full(input_ids.shape, -100, dtype=torch.long, device=input_ids.device)
        targets[:, :-1] = input_ids[:, 1:]
        targets = _cp_shard_targets(targets, ulysses_group)
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
            nll[start:end] = torch.nn.functional.cross_entropy(
                flat_logits[start:end], flat_targets[start:end], reduction="none", ignore_index=-100
            ).to(torch.float32)
        return nll.reshape(batch_size, -1)

    shifted_logits = logits[:, :-1]  # view [B, T-1, V]
    batch_size, num_positions, vocab_size = shifted_logits.shape
    flat_logits = shifted_logits.reshape(batch_size * num_positions, vocab_size)
    flat_targets = input_ids[:, 1:].reshape(-1)
    nll = torch.empty(batch_size * num_positions, dtype=torch.float32, device=input_ids.device)
    chunk = _NLL_CHUNK_ROWS
    for start in range(0, batch_size * num_positions, chunk):
        end = min(start + chunk, batch_size * num_positions)
        nll[start:end] = torch.nn.functional.cross_entropy(
            flat_logits[start:end], flat_targets[start:end], reduction="none"
        ).to(torch.float32)
    return nll.reshape(batch_size, num_positions)


def _take_mtp_loss(model: torch.nn.Module) -> torch.Tensor | None:
    """Consume the MTP loss stashed on the module by the patched forward.

    FSDP2 rebuilds the output dataclass and drops dynamic attributes. Return
    None when no loss was produced, including ``no_grad`` evaluation.
    """
    inner = getattr(model, "model", model)
    mtp_loss = getattr(inner, "_last_mtp_loss", None)
    if mtp_loss is None:
        return None
    inner._last_mtp_loss = None  # consume for the next microbatch
    if isinstance(mtp_loss, list):
        mtp_loss = [loss for loss in mtp_loss if loss is not None]
        if not mtp_loss:
            return None
        mtp_loss = torch.mean(torch.stack(mtp_loss))
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
            token_log_probs = _forward_token_log_probs(model, input_ids, attention_mask)
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute GRPO clipped-surrogate loss for one microbatch.

    Use per-sample masked token means, or whole-local-batch token normalization
    when requested by the algorithm.
    """
    tokens = rollout_data["tokens"]
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    loss_masks = rollout_data["loss_masks"]
    advantages = rollout_data["advantages"]
    old_log_probs = rollout_data["rollout_log_probs" if args.use_rollout_logprobs else "log_probs"]

    input_ids, attention_mask = _pad_batch([tokens[i] for i in microbatch], device)
    token_log_probs = _forward_token_log_probs(model, input_ids, attention_mask)

    ratio_clipped_frac = 0.0
    pg_losses: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for row, sample_index in enumerate(microbatch):
        total_length = int(total_lengths[sample_index])
        response_length = int(response_lengths[sample_index])
        new_logp = _slice_response_log_probs(token_log_probs, row, total_length, response_length)
        old_logp = old_log_probs[sample_index].to(device=new_logp.device, dtype=new_logp.dtype)
        adv = advantages[sample_index].to(device=new_logp.device, dtype=torch.float32)
        mask = loss_masks[sample_index].to(device=new_logp.device, dtype=torch.float32)

        loss_per_token, _ = flow_grpo.grpo_clip_loss(
            new_logp,
            old_logp,
            adv,
            float(getattr(args, "eps_clip", 0.2)),
            getattr(args, "eps_clip_high", None),
        )
        with torch.no_grad():
            ratio = torch.exp(new_logp - old_logp)
            clip_eps = float(getattr(args, "eps_clip", 0.2))
            ratio_clipped_frac += ((ratio - 1.0).abs() > clip_eps).to(torch.float32).mean().item() / len(microbatch)
            entropies.append((-new_logp).mean())

        pg_losses.append((loss_per_token * mask).sum() / mask.sum().clamp(min=1.0))

    from relax.algorithms import get_algorithm

    is_token_global = get_algorithm(args.advantage_estimator).advantage_normalization == "token_global"
    if is_token_global:
        # Weight sample means by supervised tokens so microbatch gradients sum to the global token mean.
        local_total = max(sum(float(m.sum()) for m in loss_masks if isinstance(m, torch.Tensor)), 1.0)
        weighted = torch.stack(
            [
                pg_loss * float(loss_masks[sample_index].to(torch.float32).sum())
                for pg_loss, sample_index in zip(pg_losses, microbatch)
            ]
        ).sum()
        loss = weighted / local_total
    else:
        loss = torch.stack(pg_losses).mean()

    metrics = {
        "pg_loss": float(loss.detach().item()),
        "ratio_clipped_frac": ratio_clipped_frac,
        "entropy": float(torch.stack(entropies).mean().item()),
    }
    return loss, metrics


def _supervised_mask(loss_mask: torch.Tensor, total_length: int, device: torch.device) -> torch.Tensor:
    """Return the next-token supervised mask for positions ``[0, T-1)``.

    Shift left once so position ``i`` supervises token ``i + 1``.
    """
    from relax.utils.sft_utils import align_loss_mask_for_sft

    aligned = align_loss_mask_for_sft(loss_mask.to(dtype=torch.float32))
    return aligned[: total_length - 1].to(device=device)


def sft_step_supervised_tokens(rollout_data: Mapping[str, Any]) -> int:
    """Count supervised next-token positions over all samples in the step.

    Complete, unsharded data masks make this a local computation. The caller
    all-reduces over the data mesh and divides by DP size for the per-token
    normalizer, matching Megatron's global token count.
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
    the global supervised count divided by DP size. This offsets FSDP's DP
    gradient average and matches Megatron's global token mean. The caller must
    skip zero-token steps.

    Ulysses CP reduces NLL sums with forward SUM/backward x CP-size to offset
    FSDP averaging, and reduces grad-free token counts with a plain SUM.
    Positive ``mtp_loss_scaling`` adds the CP-aware MTP loss stashed by the
    patched forward.
    """
    tokens = rollout_data["tokens"]
    total_lengths = rollout_data["total_lengths"]
    mm_kwargs = _collate_multimodal_inputs(rollout_data.get("multimodal_train_inputs"), microbatch, device)

    input_ids, attention_mask = _pad_batch([tokens[i] for i in microbatch], device)
    mtp_labels = None
    if mtp_loss_scaling:
        # MTP shifts these HF labels itself; -100 ignores padding and unsupervised tokens.
        mtp_labels = torch.full_like(input_ids, -100)
        for row, sample_index in enumerate(microbatch):
            total_length = int(total_lengths[sample_index])
            supervised = rollout_data["loss_masks"][sample_index].to(device=device, dtype=torch.bool)
            mtp_labels[row, :total_length] = input_ids[row, :total_length].masked_fill(
                ~supervised[:total_length], -100
            )
    token_nll = _forward_token_nll(model, input_ids, attention_mask, labels=mtp_labels, **mm_kwargs)

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
    token_nll = _forward_token_nll(model, input_ids, attention_mask, **mm_kwargs)

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


def _pad_load_state(model, src_path, device="cpu"):
    """Load the 4-layer/16-expert novamlx slice into the adapter in memory.

    Translate keys, pack experts, crop compatible tensors and return load
    stats.
    """
    import glob as _glob
    import os as _os

    import torch
    from examples.deepseekv4 import load_weights as lw
    from safetensors import safe_open

    N_LAYERS, N_EXPERTS = 4, 16

    def dt(single):
        # single-prefix target (model.layers.*) -> adapter double-prefix
        return single if not single.startswith("model.") else "model." + single

    state = model.state_dict()
    T = set(state.keys())
    stats = {"loaded": 0, "missing_final": [], "skipped_extra": [], "packed": 0, "skipped_shape_mismatch": []}

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
            suffix = tgt.split(".self_attn.", 1)[1]
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


def __ks(shard_path):
    from safetensors import safe_open

    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        return list(sf.keys())


def load_deepseek_v4_model(model_path: str, torch_dtype: torch.dtype):
    """Build the reduced 4-layer/16-expert DeepSeek-V4 model.

    Load novamlx weights when ``model_path`` contains safetensors; otherwise
    keep random initialization.
    """
    import os

    import torch
    from fsdp_turbo.models.deepseek.deepseek_v4 import (
        build_deepseek_v4_model,
        prepare_deepseek_v4_model_for_fsdp,
    )

    model = build_deepseek_v4_model()  # BF16, 4L/16E, stock official init
    if (
        model_path
        and os.path.isdir(model_path)
        and any(fname.endswith(".safetensors") for fname in os.listdir(model_path))
    ):
        state, stats = _pad_load_state(model, model_path, device="cpu")
        model.load_state_dict(state, strict=False)
        # Official hash-MoE ids target 256 experts; clamp them for this 16-expert build.
        clamped = 0
        for bn, b in model.named_buffers():
            if bn.endswith("gate.tid2eid"):
                over = b.data >= 16
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
    return prepare_deepseek_v4_model_for_fsdp(model, device=device, parameter_dtype=torch_dtype)
