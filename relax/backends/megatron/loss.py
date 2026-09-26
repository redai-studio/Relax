# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from relax.algorithms import get_algorithm
from relax.algorithms.advantages import compute_advantages_and_returns as compute_advantages_and_returns_impl
from relax.algorithms.policy import compute_policy_loss_for
from relax.algorithms.spec import ALGORITHM_SPECS
from relax.utils.distributed_utils import distributed_masked_normalize, distributed_masked_whiten
from relax.utils.misc import load_function
from relax.utils.opd.opd_utils import (
    apply_opd_to_advantages,
    compute_opd_topk_log_probs,
    compute_policy_opd_loss,
    resolve_opd_gather_topk_token_ids,
    validate_opd_topk_gather,
)
from relax.utils.replay import capture_hooks
from relax.utils.training.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
)
from relax.utils.training.preference_utils import (
    build_preference_pair_indices,
    dpo_pair_loss,
    require_tensor_condition,
)
from relax.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_cp_local_num_tokens,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    maybe_padded_total_lengths,
    slice_log_prob_with_cp,
)


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    padded_total_lengths: list[int] | None = None,
    apply_temperature: bool = True,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    # SFT chunked path (--sft-chunked-logits) feeds hidden_states (bf16) into
    # this slicer — slicing itself is dtype-agnostic; that caller casts to fp32
    # per sub-chunk. All other paths (RL, SFT legacy logits, SFT eval PPL)
    # produce real fp32 logits and must stay strict.
    if args.loss_type == "sft" and getattr(args, "sft_chunked_logits", False):
        assert logits.dtype in (torch.float32, torch.bfloat16, torch.float16), f"{logits.dtype}"
    else:
        assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    cp_size = dynamic_cp_size if dynamic_cp_size is not None else mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        padded_total_length = padded_total_lengths[i] if padded_total_lengths is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            if response_length == total_length:
                # SFT branch; see relax.utils.sft_utils.compute_sft_response_chunk.
                from relax.utils.sft_utils import compute_sft_response_chunk

                logits_chunk, tokens_chunk = compute_sft_response_chunk(logits, tokens, start, end)
            else:
                logits_chunk = logits[start - 1 : end - 1]
                tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length,
                response_length,
                qkv_format,
                max_seq_len,
                padded_total_length,
                dynamic_cp_size=dynamic_cp_size,
                dynamic_cp_rank=dynamic_cp_rank,
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        # Apply temperature per-chunk instead of on the full [T, V] logits to avoid
        # a single ~16GiB allocation that OOMs under fragmentation.
        # Skipped when apply_temperature is False (value head: scalar predictions
        # must not be temperature-scaled), and on the SFT chunked path where
        # `logits_chunk` is hidden_states [R, H] not logits (that caller applies
        # temperature on the real per-sub-chunk logits after lm_head instead).
        if (
            apply_temperature
            and args.rollout_temperature != 1.0
            and not (args.loss_type == "sft" and getattr(args, "sft_chunked_logits", False))
        ):
            logits_chunk = logits_chunk / args.rollout_temperature

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits: torch.Tensor,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    padded_total_lengths: list[int] | None = None,
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-
    attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits: Model output used only to determine the local sequence length
            (``logits.size(1)``).
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()

    logits_local_len = logits.size(1)  # logits shape: [1, T_local, ...]
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if e <= s:
                # This rank has no response logprobs for this sample. Match the
                # placeholder's requires_grad to the real data (`value`): forcing
                # requires_grad=True makes the concatenated all_reduce input require
                # grad on ranks with an empty chunk while non-empty ranks (F.pad of
                # a no-grad `value`) do not, so the differentiable dist.nn.all_reduce
                # below builds a backward graph on some CP ranks but not others and
                # the collective deadlocks.
                full_resp = torch.zeros(
                    response_length,
                    dtype=value.dtype,
                    device=value.device,
                    requires_grad=value.requires_grad,
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (resp_start, response_length - resp_end))

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            padded_total_length = padded_total_lengths[idx] if padded_total_lengths is not None else None
            new_values.append(
                slice_log_prob_with_cp(
                    full_resp,
                    total_length,
                    response_length,
                    args.qkv_format,
                    max_seq_len,
                    padded_total_length,
                )
            )

        res[key] = new_values


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    with_topk: bool = False,
    topk_k: int | None = None,
    gather_topk_token_ids: list[torch.Tensor] | None = None,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    padded_total_lengths: list[int] | None = None,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
    lm_head_forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] | None = None,
    **_,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Compute per-token log-probabilities (and optionally entropy) on
    responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. When entropy is requested only as a
    metric (`entropy_coef == 0`), its backward activations are not retained.

    Args:
        logits: Policy logits with shape `[1, T, V]`. When ``lm_head_forward``
            is provided this is instead per-sample hidden_states
            `[B, S, H]` — see ``lm_head_forward`` below.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        with_topk: If True, include per-sample student-side top-K *indices*
            (selected from current logits via ``torch.topk``) in result key
            ``"topk_token_ids"``. Used for diagnostic / membership metrics.
        topk_k: Override for the K used by ``with_topk``; defaults to
            ``args.opd_log_prob_top_k``.
        gather_topk_token_ids: Optional list of per-sample ``[R, K]`` long
            tensors (already CP-sliced by ``post_process_rollout_data``).
            When provided, compute student's *current-step* log-probabilities
            at exactly these K token ids per position via
            :func:`compute_log_probs_on_topk_token_ids` (vocab-parallel
            gather + cross-rank logsumexp, differentiable). The returned dict
            gains key ``"topk_log_probs"`` mapping to list of ``[R, K]``
            tensors with gradient flowing back to the policy.
            Used by the OPD-as-loss top-K path; the K ids come from the
            student's rollout-time top-K stored in
            ``batch["student_topk_token_ids"]``.
        non_loss_data: Unused; kept for API compatibility.
        lm_head_forward: If set, ``logits`` is actually hidden_states; we defer
            the lm_head into this loss path and chunk the matmul at
            ``args.sft_logits_chunk_size`` so the full ``[B, S, V/TP]`` logits
            never materialize (SFT chunked path). Requires ``with_entropy=False``
            and ``with_topk=False`` — chunked path has no full-vocab tensor to
            derive entropy/top-k from.

    Returns:
        Tuple of:
        - empty tensor (placeholder for loss-compatible API)
        - Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors. If `with_topk` is True, also includes
        "topk_token_ids" key. If `gather_topk_token_ids` is provided, also
        includes "topk_log_probs" key with list of `[R, K]` tensors.
    """
    assert non_loss_data

    validate_opd_topk_gather(args, gather_topk_token_ids)
    if lm_head_forward is not None:
        assert not with_entropy and not with_topk, (
            "lm_head_forward chunked path doesn't materialize full vocab — entropy/topk unavailable."
        )
        sft_chunk_size = getattr(args, "sft_logits_chunk_size", 1024)
        if sft_chunk_size <= 0:
            sft_chunk_size = 1024
    resolved_topk_k = topk_k if topk_k is not None else getattr(args, "opd_log_prob_top_k", 0)
    tp_group = mpu.get_tensor_model_parallel_group()
    # Keep entropy metrics, but skip saving entropy-backward activations when
    # the entropy term cannot affect the loss.
    with_entropy_grad = with_entropy and getattr(args, "entropy_coef", 0.0) != 0
    log_probs_list = []
    entropy_list = []
    topk_token_ids_list = []
    topk_log_probs_list: list[torch.Tensor] = []
    for sample_idx, (logits_chunk, tokens_chunk) in enumerate(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            padded_total_lengths=padded_total_lengths,
            dynamic_cp_size=dynamic_cp_size,
            dynamic_cp_rank=dynamic_cp_rank,
        )
    ):
        if lm_head_forward is not None:
            # SFT chunked: logits_chunk is per-sample hidden_states [R, H].
            # Run lm_head per sub-chunk so the full [R, V/TP] logits tensor
            # never materializes; concat per-sub-chunk log_probs back into the
            # per-sample [R] tensor that downstream expects.
            chunk_lps: list[torch.Tensor] = []
            for s in range(0, logits_chunk.size(0), sft_chunk_size):
                e = min(s + sft_chunk_size, logits_chunk.size(0))
                h_sub = logits_chunk[s:e].unsqueeze(1)  # [sub, 1, H]
                logits_sub, _ = lm_head_forward(h_sub)
                logits_sub = logits_sub.squeeze(1).float()
                if args.rollout_temperature != 1.0:
                    logits_sub = logits_sub / args.rollout_temperature
                log_prob_sub, _ = calculate_log_probs_and_entropy(
                    logits_sub,
                    tokens_chunk[s:e],
                    tp_group,
                    with_entropy=False,
                )
                chunk_lps.append(log_prob_sub.squeeze(-1))
            log_prob = (
                torch.cat(chunk_lps, dim=0)
                if chunk_lps
                # fp32 to match calculate_log_probs_and_entropy's return dtype;
                # logits are upcast above. A mismatch would break the downstream
                # torch.cat over per-sample log-probabilities.
                else logits_chunk.new_zeros((0,), dtype=torch.float32)
            )
            entropy = None
        else:
            log_prob, entropy = calculate_log_probs_and_entropy(
                logits_chunk,
                tokens_chunk,
                tp_group,
                with_entropy=with_entropy,
                with_entropy_grad=with_entropy_grad,
                chunk_size=args.log_probs_chunk_size,
            )
            log_prob = log_prob.squeeze(-1)

        log_probs_list.append(log_prob)
        entropy_list.append(entropy)

        if with_topk:
            k = min(max(int(resolved_topk_k), 1), int(logits_chunk.size(-1)))
            topk_token_ids_list.append(torch.topk(logits_chunk, k=k, dim=-1).indices)

        if gather_topk_token_ids is not None:
            topk_log_probs_list.append(compute_opd_topk_log_probs(logits_chunk, gather_topk_token_ids, sample_idx))

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    if with_topk:
        res["topk_token_ids"] = topk_token_ids_list
    if gather_topk_token_ids is not None:
        res["topk_log_probs"] = topk_log_probs_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            padded_total_lengths=padded_total_lengths,
        )

    return torch.empty((0,), device=logits.device), res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    padded_total_lengths: list[int] | None = None,
    **_,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration passed to `get_responses`; temperature scaling is
            disabled for value outputs.
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Tuple of:
        - empty tensor (placeholder for loss-compatible API)
        - Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        padded_total_lengths=padded_total_lengths,
        apply_temperature=False,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            padded_total_lengths=padded_total_lengths,
        )

    return torch.empty((0,), device=logits.device), res


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on
    `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then dispatches to the estimator
    named by `relax.algorithms.spec.ALGORITHM_SPECS[...].advantage_fn`. The
    supported methods are whatever that registry holds -- deliberately not
    listed here, because keeping algorithm names in prose is the duplication
    the registry exists to remove. When `args.normalize_advantages` is True,
    advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] | list[torch.Tensor] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)
    padded_total_lengths: list[int] | None = maybe_padded_total_lengths(
        total_lengths,
        args.qkv_format,
        getattr(args, "is_vl_model", False)
        or rollout_data.get("multimodal_train_inputs") is not None
        or getattr(args, "uses_unsplit_forward", False),
    )

    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    advantages, returns = compute_advantages_and_returns_impl(
        args,
        rewards=rewards,
        kl=kl,
        loss_masks=loss_masks,
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        values=values,
        # Only this path can compute it: `maybe_padded_total_lengths` reads
        # args.qkv_format plus the VL / unsplit-forward flags, which the
        # Advantages deployment does not have. GAE needs it to slice CP shards
        # at the padded offsets; omitting it does not raise, it reads the wrong
        # token positions.
        padded_total_lengths=padded_total_lengths,
    )

    # Optional pure OPD mode: remove all non-OPD reward contribution.
    # This keeps only the OPD KL term injected below.
    if args.use_opd and getattr(args, "opd_only_reward", False):
        assert isinstance(advantages, list), f"Expected list advantages, got {type(advantages)}"
        assert isinstance(returns, list), f"Expected list returns, got {type(returns)}"
        advantages = [torch.zeros_like(a) for a in advantages]
        returns = [torch.zeros_like(r) for r in returns]

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd:
        apply_opd_to_advantages(args, rollout_data, advantages)

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        # Under dynamic CP, rollout_data was merged back to full-length responses
        # (dynamic_cp_merge_output), so advantages/masks are already full — use cp=1
        # (no zig-zag chunking). The whitening stats are unchanged by the CP-group
        # replication (uniform duplication doesn't move mean/std).
        cp_size = 1 if getattr(args, "dynamic_context_parallel", False) else mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
                padded_total_length = padded_total_lengths[i] if padded_total_lengths is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len, padded_total_length
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        assert all_advs.size() == all_masks.size(), (
            f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
        )
        # Which normalisation to apply is declared by the algorithm registry.
        # This used to be a set of estimator names maintained here and a second
        # identical set in `policy_loss_function` below, which is exactly the
        # kind of pair that drifts: the two have to stay in step because
        # token-global normalisation is only correct together with the
        # mask-safe reducer.
        is_token_global = get_algorithm(args.advantage_estimator).advantage_normalization == "token_global"
        if is_token_global or all_masks.numel() > 0:
            dp_group = mpu.get_data_parallel_group()

            if is_token_global:
                whitened_advs_flat, raw_mean, raw_variance, valid_count = distributed_masked_normalize(
                    all_advs,
                    all_masks,
                    process_group=dp_group,
                )
                variance_floor = torch.tensor(1e-8, device=raw_variance.device, dtype=raw_variance.dtype)
                normalized_variance = raw_variance / torch.maximum(raw_variance, variance_floor)
                num_samples = len(advantages)
                raw_mean = raw_mean.detach()
                raw_std = raw_variance.sqrt().detach()
                normalized_mean = torch.zeros_like(raw_mean)
                normalized_std = normalized_variance.sqrt().detach()
                valid_count = valid_count.detach()
                zero_variance = (raw_variance == 0).to(dtype=raw_variance.dtype).detach()
                rollout_data["reinforce_pp_advantage_raw_mean"] = [raw_mean] * num_samples
                rollout_data["reinforce_pp_advantage_raw_std"] = [raw_std] * num_samples
                rollout_data["reinforce_pp_advantage_normalized_mean"] = [normalized_mean] * num_samples
                rollout_data["reinforce_pp_advantage_normalized_std"] = [normalized_std] * num_samples
                rollout_data["reinforce_pp_valid_token_count"] = [valid_count] * num_samples
                rollout_data["reinforce_pp_zero_variance"] = [zero_variance] * num_samples
            else:
                whitened_advs_flat = distributed_masked_whiten(
                    all_advs,
                    all_masks,
                    process_group=dp_group,
                    shift_mean=True,
                )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)

    log_ratio = old_log_probs - rollout_log_probs
    tis = torch.exp(log_ratio)
    tis_abs = (tis - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    # K3 KL ≈ E[exp(log_ratio) - log_ratio - 1]; direct KL = E[log π_rollout - log π_train].
    mismatch_k3_kl = tis - log_ratio - 1
    mismatch_kl = -log_ratio
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
        "mismatch_kl": mismatch_kl.clone().detach(),
        "mismatch_k3_kl": mismatch_k3_kl.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)

    log_ratio = old_log_probs - rollout_log_probs
    ice_ratio = torch.exp(log_ratio)
    ice_abs = (ice_ratio - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    mismatch_k3_kl = ice_ratio - log_ratio - 1
    mismatch_kl = -log_ratio
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
        "mismatch_kl": mismatch_kl.clone().detach(),
        "mismatch_k3_kl": mismatch_k3_kl.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def _get_reinforce_plus_plus_mask_safe_reducer(
    reducer: Callable[[torch.Tensor], torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Exclude masked non-finite values before a REINFORCE++ reduction."""
    flat_valid_mask = torch.cat([mask.reshape(-1) != 0 for mask in loss_masks], dim=0)

    def reduce_valid_tokens(values: torch.Tensor) -> torch.Tensor:
        valid_mask = flat_valid_mask.to(device=values.device)
        if values.shape[0] != valid_mask.numel():
            raise ValueError(
                "REINFORCE++ reducer expected one value per response token, "
                f"got {values.shape[0]} values for {valid_mask.numel()} mask elements."
            )
        safe_values = torch.where(valid_mask, values, torch.zeros_like(values))
        return reducer(safe_values)

    return reduce_valid_tokens


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO/SAPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates policy gradient loss. For GSPO, gathers full sequences via
    context-parallel all-gather before computing per-sample KL. For SAPO,
    uses soft gating instead of hard clipping. Optionally applies TIS
    (Truncated Importance Sampling) correction and adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    if isinstance(batch["advantages"], list):
        advantages = torch.cat(batch["advantages"], dim=0)
    else:
        advantages = batch["advantages"]

    # Same registry field as `compute_advantages_and_returns` reads: the
    # mask-safe reducer is the other half of token-global normalisation, not an
    # independent choice.
    is_token_global = get_algorithm(args.advantage_estimator).advantage_normalization == "token_global"

    if is_token_global:
        sum_of_sample_mean = _get_reinforce_plus_plus_mask_safe_reducer(sum_of_sample_mean, batch["loss_masks"])

    true_on_policy = getattr(args, "true_on_policy_mode", False)
    # In true on-policy mode, actor_fwd is absent so batch["log_probs"] is missing;
    # old_log_probs is derived inline from this forward (assigned after the pass below).
    if not true_on_policy:
        old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    padded_total_lengths = batch.get("padded_total_lengths", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        gather_topk_token_ids=resolve_opd_gather_topk_token_ids(args, batch),
        max_seq_lens=max_seq_lens,
        padded_total_lengths=padded_total_lengths,
        dynamic_cp_size=batch.get("dynamic_cp_size", None),
        dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if true_on_policy:
        # Same weights + deterministic Megatron forward ⇒ old_log_probs == log_probs.
        # Detaching makes ratio = exp(log_probs - log_probs.detach()) ≡ 1 with the
        # gradient of log_probs flowing through pg_loss, recovering vanilla PG.
        # TIS (which compares train-engine vs rollout-engine log_probs) still works
        # because train_log_probs is numerically identical to what actor_fwd produced.
        old_log_probs = [lp.detach() for lp in log_probs]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    algorithm = get_algorithm(args.advantage_estimator)
    need_full_log_probs = args.use_opsm or algorithm.needs_full_log_probs

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        if padded_total_lengths is None:
            padded_iter = [None] * len(log_probs)
        else:
            padded_iter = padded_total_lengths
        # OPSM supports CP: reconstruct each sample's full response from its CP-local
        # zig-zag shards. Under dynamic CP, gather over this mb's dynamic CP sub-group
        # (size/rank/group), not the static CP group. (GSPO does not support CP.)
        dynamic_cp_size = batch.get("dynamic_cp_size", None)
        dynamic_cp_rank = batch.get("dynamic_cp_rank", None)
        dynamic_cp_group = (
            mpu.get_dynamic_data_context_parallel_groups(group_size=dynamic_cp_size)
            if dynamic_cp_size is not None
            else None
        )
        full_log_probs = [
            all_gather_with_cp(
                log_prob,
                total_length,
                response_length,
                padded_total_length,
                dynamic_cp_size=dynamic_cp_size,
                dynamic_cp_rank=dynamic_cp_rank,
                dynamic_cp_group=dynamic_cp_group,
            )
            for log_prob, total_length, response_length, padded_total_length in zip(
                log_probs, total_lengths, response_lengths, padded_iter, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(
                old_log_prob,
                total_length,
                response_length,
                padded_total_length,
                dynamic_cp_size=dynamic_cp_size,
                dynamic_cp_rank=dynamic_cp_rank,
                dynamic_cp_group=dynamic_cp_group,
            )
            for old_log_prob, total_length, response_length, padded_total_length in zip(
                old_log_probs, total_lengths, response_lengths, padded_iter, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if algorithm.kl_level == "sequence":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)

    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac, policy_scalar_metrics = compute_policy_loss_for(
        args,
        log_probs=log_probs,
        ppo_kl=ppo_kl,
        advantages=advantages,
    )

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        # In true on-policy mode batch["log_probs"] is absent; old_log_probs (detached
        # list of this-step forward) is numerically identical to what actor_fwd produced,
        # so TIS measures the same train-engine vs rollout-engine mismatch.
        tis_train_log_probs = (
            [lp.detach() for lp in log_probs_and_entropy["log_probs"]] if true_on_policy else batch["log_probs"]
        )
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": tis_train_log_probs,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # Rebuild with the modified numerator mask while preserving the original
        # logical-sample denominator. The masks are sliced with CP in the reducer.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
            padded_total_lengths,
            dynamic_cp_size=batch.get("dynamic_cp_size", None),
            dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
            sample_denoms=batch.get("sample_index_mask_sums", None),
        )
        if is_token_global:
            sum_of_sample_mean = _get_reinforce_plus_plus_mask_safe_reducer(
                sum_of_sample_mean, modified_response_masks
            )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
        if is_token_global:
            pg_loss_reducer = _get_reinforce_plus_plus_mask_safe_reducer(pg_loss_reducer, pg_loss_masks)
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)

    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    opd_loss, opd_reported_loss = compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        log_probs_and_entropy=log_probs_and_entropy,
    )
    if opd_loss is not None:
        loss = loss + opd_loss

    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"] is not None:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        # Train/inference mismatch = |train-engine logprob - rollout-engine logprob| for
        # the SAME tokens+weights. old_log_probs is the wrong reference under
        # --use-rollout-logprobs: there old_log_probs IS rollout_log_probs (see the
        # assignment above), so the diff would collapse to 0. Use the actor-fwd
        # train-side recompute (batch["log_probs"], populated when --get-mismatch-metrics
        # forces the extra forward) and fall back to this step's fresh forward log_probs
        # otherwise (colocate on-policy: rollout weights == current weights).
        if not args.use_rollout_logprobs:
            train_side_log_probs = old_log_probs
        elif batch.get("log_probs"):
            train_side_log_probs = torch.cat(batch["log_probs"], dim=0)
        else:
            train_side_log_probs = log_probs.detach()
        train_rollout_logprob_abs_diff = sum_of_sample_mean((train_side_log_probs - rollout_log_probs).abs())
        train_rollout_prob_abs_diff = sum_of_sample_mean(
            (torch.exp(train_side_log_probs) - torch.exp(rollout_log_probs)).abs()
        )

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    # Trajectory-replay capture: record the loss.policy stage. No-op unless
    # capture is enabled and this step is selected (single global read).
    capture_hooks.capture_policy_loss(
        old_log_probs=old_log_probs,
        log_probs=log_probs,
        entropy=entropy,
        advantages=advantages,
        loss_masks=batch["loss_masks"],
        response_lengths=response_lengths,
        total_lengths=total_lengths,
        reported_loss=reported_loss,
        micro_batch_index=batch.get("replay_micro_batch_index"),
    )

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
        reported_loss["train_rollout_prob_abs_diff"] = train_rollout_prob_abs_diff.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    reported_loss.update(opd_reported_loss)

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    duplicate_policy_metrics = reported_loss.keys() & policy_scalar_metrics.keys()
    if duplicate_policy_metrics:
        raise ValueError(f"Policy scalar metrics would overwrite existing metrics: {sorted(duplicate_policy_metrics)}")
    reported_loss.update(policy_scalar_metrics)

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
        padded_total_lengths=batch.get("padded_total_lengths", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
        padded_total_lengths=batch.get("padded_total_lengths", None),
        dynamic_cp_size=batch.get("dynamic_cp_size", None),
        dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def dpo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],  # noqa: ARG001
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a pair-summed DPO objective using explicit pair identity."""
    if len(batch["response_lengths"]) % 2 != 0:
        raise ValueError("DPO micro-batch must contain an even number of chosen/rejected branches")
    _, values = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
        padded_total_lengths=batch.get("padded_total_lengths", None),
        dynamic_cp_size=batch.get("dynamic_cp_size", None),
        dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
    )
    policy_token_log_probs = values["log_probs"]

    def sequence_sums(token_values) -> torch.Tensor:
        if len(token_values) != len(batch["loss_masks"]):
            raise ValueError("DPO token log-probabilities are not branch aligned")
        sums = []
        for branch_values, mask in zip(token_values, batch["loss_masks"], strict=True):
            branch_values = torch.as_tensor(branch_values, device=logits.device)
            branch_mask = mask.to(device=logits.device, dtype=branch_values.dtype)
            if branch_values.shape != branch_mask.shape:
                raise ValueError(
                    "DPO branch log-probability/mask shape mismatch: "
                    f"{tuple(branch_values.shape)} vs {tuple(branch_mask.shape)}"
                )
            require_tensor_condition(
                branch_mask.to(dtype=torch.bool).any(),
                "DPO branch completion mask must contain at least one supervised token",
            )
            sums.append((branch_values * branch_mask).sum())
        return torch.stack(sums)

    pair_ids = batch.get("preference_branch_pair_ids")
    branch_is_chosen = batch.get("preference_is_chosen")
    if pair_ids is None or branch_is_chosen is None:
        raise ValueError("DPO batch is missing preference pair identity fields")
    chosen_indices, rejected_indices = build_preference_pair_indices(pair_ids, branch_is_chosen)
    chosen_index = torch.as_tensor(chosen_indices, dtype=torch.long, device=logits.device)
    rejected_index = torch.as_tensor(rejected_indices, dtype=torch.long, device=logits.device)

    policy_sums = sequence_sums(policy_token_log_probs)
    policy_chosen = policy_sums.index_select(0, chosen_index)
    policy_rejected = policy_sums.index_select(0, rejected_index)
    reference_free = bool(args.dpo_reference_free)
    if reference_free:
        reference_chosen = reference_rejected = None
        ref_chosen_for_metrics = torch.zeros_like(policy_chosen)
        ref_rejected_for_metrics = torch.zeros_like(policy_rejected)
    else:
        reference_values = batch.get("ref_log_probs")
        if reference_values is None:
            raise ValueError("standard DPO batch is missing frozen-reference log-probabilities")
        reference_sums = sequence_sums(reference_values)
        reference_chosen = reference_sums.index_select(0, chosen_index)
        reference_rejected = reference_sums.index_select(0, rejected_index)
        ref_chosen_for_metrics = reference_chosen
        ref_rejected_for_metrics = reference_rejected
    pair_losses = dpo_pair_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen=reference_chosen,
        reference_rejected=reference_rejected,
        beta=args.dpo_beta,
        reference_free=reference_free,
    )
    chosen_rewards = args.dpo_beta * (policy_chosen - ref_chosen_for_metrics)
    rejected_rewards = args.dpo_beta * (policy_rejected - ref_rejected_for_metrics)
    # pair_losses is never empty: build_preference_pair_indices raises on an
    # empty micro-batch, so no gradient-safety fallback is needed here.
    loss = pair_losses.sum()
    reward_margin = chosen_rewards - rejected_rewards
    tie = reward_margin.abs() <= 1e-6
    strict = reward_margin > 0
    correct = reward_margin > 1e-6
    metrics = {
        "dpo/loss": pair_losses.detach().sum(),
        "dpo/logps_chosen": policy_chosen.detach().sum(),
        "dpo/logps_rejected": policy_rejected.detach().sum(),
        "dpo/reward_chosen": chosen_rewards.detach().sum(),
        "dpo/reward_rejected": rejected_rewards.detach().sum(),
        "dpo/reward_margin": reward_margin.detach().sum(),
        "dpo/strict_accuracy": strict.to(torch.float32).detach().sum(),
        "dpo/tie_rate": tie.to(torch.float32).detach().sum(),
        "dpo/tie_aware_accuracy": (correct.to(torch.float32) + 0.5 * tie.to(torch.float32)).detach().sum(),
    }
    metrics["dpo/pair_accuracy"] = metrics["dpo/strict_accuracy"]
    if not reference_free:
        metrics["dpo/ref_logps_chosen"] = reference_chosen.detach().sum()
        metrics["dpo/ref_logps_rejected"] = reference_rejected.detach().sum()
    return loss, metrics


def get_sequence_classification_outputs(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Pool one classification logit vector per sample on the owning CP
    rank."""
    local_logits: list[torch.Tensor] = []
    local_indices: list[int] = []
    for sample_idx, (logits_chunk, _) in enumerate(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            max_seq_lens=batch.get("max_seq_lens", None),
            padded_total_lengths=batch.get("padded_total_lengths", None),
            apply_temperature=False,
            dynamic_cp_size=batch.get("dynamic_cp_size", None),
            dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
        )
    ):
        if logits_chunk.shape[0] == 0:
            continue
        if logits_chunk.shape != (1, int(args.num_labels)):
            raise ValueError(
                "sequence classification pooling expected one logit vector with shape "
                f"(1, {args.num_labels}), got {tuple(logits_chunk.shape)} for sample {sample_idx}"
            )
        local_logits.append(logits_chunk.squeeze(0))
        local_indices.append(sample_idx)

    if not local_logits:
        return logits.new_empty((0, int(args.num_labels))), local_indices
    return torch.stack(local_logits, dim=0), local_indices


def sequence_classification_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute single-label CE or multi-label BCE on last-valid-token
    logits."""
    labels = batch.get("classification_labels")
    if labels is None:
        raise ValueError("sequence classification batch is missing classification_labels")

    metric_name = "accuracy" if args.problem_type == "single_label_classification" else "subset_accuracy"
    local_logits, local_indices = get_sequence_classification_outputs(args, batch, logits)
    if local_indices:
        local_labels = [labels[i] for i in local_indices]
        if args.problem_type == "single_label_classification":
            label_tensor = torch.stack([torch.as_tensor(label).reshape(()) for label in local_labels]).to(
                device=local_logits.device, dtype=torch.long
            )
            per_sample_loss = F.cross_entropy(local_logits, label_tensor, reduction="none")
            per_sample_correct = (local_logits.argmax(dim=-1) == label_tensor).float()
        else:
            label_tensor = torch.stack([torch.as_tensor(label).reshape(-1) for label in local_labels]).to(
                device=local_logits.device, dtype=torch.float32
            )
            if label_tensor.shape != local_logits.shape:
                raise ValueError(
                    f"multi-label targets must have shape {tuple(local_logits.shape)}, got {tuple(label_tensor.shape)}"
                )
            per_sample_loss = F.binary_cross_entropy_with_logits(
                local_logits,
                label_tensor,
                reduction="none",
            ).mean(dim=-1)
            predictions = torch.sigmoid(local_logits) >= args.classification_threshold
            per_sample_correct = predictions.eq(label_tensor.bool()).all(dim=-1).float()

        loss = sum_of_sample_mean(per_sample_loss)
        accuracy = sum_of_sample_mean(per_sample_correct)
    else:
        loss = logits.sum() * 0.0
        accuracy = logits.detach().sum() * 0.0

    return loss, {"loss": loss.detach(), metric_name: accuracy.detach()}


def sft_loss_function_chunked(
    args: Namespace,
    batch: RolloutBatch,
    hidden_states: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    *,
    lm_head_forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SFT loss that defers lm_head into the loss to avoid materializing the
    full [B, S, V/TP] logits tensor. ``hidden_states`` is [B, S, H] (the bypass
    in forward_step gathers any SP-sharded hidden into full S);
    ``lm_head_forward`` is the captured original output_layer.forward.

    Inner-func signature matches ``sft_loss_function``; dispatched from
    ``loss_function`` via ``partial(this, lm_head_forward=lm_head_forward)`` so
    the recompute wrap, CP grad-flow guard, Megatron scaling tail, and return-
    tuple shape all reuse ``loss_function``'s outer body. Body itself mirrors
    ``sft_loss_function`` one-for-one — only difference is passing
    ``lm_head_forward`` into ``get_log_probs_and_entropy`` (which then chunks
    the lm_head + CE matmul internally — see that function's docstring).
    """
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        hidden_states,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
        padded_total_lengths=batch.get("padded_total_lengths", None),
        dynamic_cp_size=batch.get("dynamic_cp_size", None),
        dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
        lm_head_forward=lm_head_forward,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    if log_probs.numel() == 0:
        loss += 0 * hidden_states.sum()

    return loss, {"loss": loss.clone().detach()}


def mtp_only_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    hidden_states: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Trigger the attached MTP auxiliary loss without a main language loss."""
    del args, batch, sum_of_sample_mean
    loss = 0.0 * hidden_states.sum()
    return loss, {"loss": loss.detach()}


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
    *,
    lm_head_forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] | None = None,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head). For SFT chunked path,
            this is actually hidden_states `[B, S, H]` (model() bypassed the
            output_layer); ``lm_head_forward`` must be supplied so the loss
            function can run lm_head per sub-chunk.
        lm_head_forward: Captured original output_layer.forward callable from
            ``_bypass_output_layer``; only meaningful when SFT chunked path is
            active. Forward_step always partials this in (None when not chunked).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [denominator, metric1, metric2, ...]). The
          denominator is the token count for per-token loss and a zero
          placeholder for sample-mean loss.
    """
    # CP-local token count (tokens whose loss this rank actually contributes).
    # Summed across the CP group in finalize_model_grads / the metric all-reduce,
    # it counts every token exactly once regardless of CP degree, so the per-token
    # normalizer is correct even when CP differs across micro-batches (dynamic CP).
    # Under static CP it equals the old full-sample count distributed across ranks,
    # so the final loss/grad/metric are unchanged after all-reduce.
    num_tokens = get_cp_local_num_tokens(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.qkv_format,
        batch.get("max_seq_lens", None),
        batch.get("padded_total_lengths", None),
        dynamic_cp_size=batch.get("dynamic_cp_size", None),
        dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
    )

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
        batch.get("padded_total_lengths", None),
        dynamic_cp_size=batch.get("dynamic_cp_size", None),
        dynamic_cp_rank=batch.get("dynamic_cp_rank", None),
        sample_denoms=batch.get("sample_index_mask_sums", None),
    )

    if getattr(args, "mtp_only_training", False):
        func = mtp_only_loss_function
    else:
        match args.loss_type:
            case "policy_loss":
                func = policy_loss_function
            case "value_loss":
                func = value_loss_function
            case "sft":
                if getattr(args, "sft_objective", "causal_lm") == "dpo":
                    func = dpo_loss_function
                elif getattr(args, "task_type", "causal_lm") == "seq_cls":
                    func = sequence_classification_loss_function
                elif getattr(args, "sft_chunked_logits", False) and lm_head_forward is not None:
                    # Bind lm_head_forward so chunked path matches the standard
                    # inner-func signature; outer body (recompute, CP guard,
                    # Megatron scaling, return-tuple) is then shared with legacy.
                    func = partial(sft_loss_function_chunked, lm_head_forward=lm_head_forward)
                else:
                    func = sft_loss_function
            case "custom_loss":
                func = load_function(args.custom_loss_function_path)
            case _:
                raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(
            func,
            args,
            batch,
            logits,
            sum_of_sample_mean,
            use_reentrant=getattr(args, "recompute_loss_function_use_reentrant", True),
        )
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Preserve the pre-registry scalar logging convention: only per-token
    # aggregation compensates for the framework's token denominator.
    if args.calculate_per_token_loss:
        algorithm = ALGORITHM_SPECS.get(getattr(args, "advantage_estimator", None))
        if algorithm is not None:
            for key in algorithm.policy_scalar_metric_names:
                if key in log:
                    log[key] = log[key] * num_tokens

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding or all-masked). Without this, gradient doesn't flow through their attention
    # path, so the CP gather's backward (reduce-scatter) is not called, deadlocking other
    # CP ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    # Fully-async + dynamic-batch path injects per-rollout loss_scale and tags
    # dummy micro-batches (used to align num_microbatches across DPs). Dummy
    # mbs must contribute zero gradient AND zero metric values.
    is_dummy = batch.get("__is_dummy__", False)
    explicit_loss_scale = batch.get("__loss_scale__", None)

    # Rescale the loss for Megatron's gradient accumulation. The non-per-token
    # branch folds in the DP(+CP) world size (cancelled by DDP's 1/dp_cp grad
    # scaling); the per-token branch does NO CP scaling (normalization is the
    # all-reduced CP-local token count in finalize_model_grads).
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        if is_dummy:
            # Zero-out gradient contribution but keep the autograd graph
            # connected so PP/CP backward collectives still complete.
            loss = 0.0 * loss
        elif explicit_loss_scale is not None:
            loss = loss * explicit_loss_scale
        else:
            loss = (
                loss
                * num_microbatches
                / global_batch_size
                * mpu.get_data_parallel_world_size(with_context_parallel=True)
            )
    else:
        if is_dummy:
            loss = 0.0 * loss
        # Non-dummy per-token path: do NOT scale by cp_size. `loss` is the
        # CP-local token-sum; finalize_model_grads normalizes the summed gradient
        # by the all-reduced CP-local `num_tokens`. A `* cp_size` here would weight
        # each sample by its CP degree — wrong when CP differs across micro-batches
        # (dynamic CP). Under static CP the removed factor exactly cancels the old
        # full-count denominator, leaving the final loss/grad unchanged.

    effective_num_tokens = torch.zeros_like(num_tokens) if is_dummy else num_tokens
    is_sequence_classification = getattr(args, "task_type", "causal_lm") == "seq_cls"
    denominator = (
        effective_num_tokens
        if args.calculate_per_token_loss or is_sequence_classification
        else torch.zeros_like(num_tokens)
    )
    metric_values = [
        value.to(logits.device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=logits.device)
        for value in log.values()
    ]
    log_values = torch.stack([denominator] + metric_values)
    if is_dummy:
        # Drop this mb's contribution from logged metric averages.
        log_values = torch.zeros_like(log_values)

    return (
        loss,
        (effective_num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            "values": log_values,
        },
    )
