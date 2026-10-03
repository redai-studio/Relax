# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from argparse import Namespace
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training.global_vars import get_args
from torch.nn.utils.rnn import pad_sequence

from relax.engine.sft.runtime import is_preference_mode
from relax.utils import device as device_utils
from relax.utils import tracking_utils
from relax.utils.data.data import get_minimum_num_micro_batch_size
from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions
from relax.utils.logging_utils import get_logger
from relax.utils.metrics.metric_utils import compute_rollout_step
from relax.utils.opd.opd_utils import OPD_ROLLOUT_LOG_SKIP_FIELDS
from relax.utils.timer import Timer
from relax.utils.training import train_metric_utils
from relax.utils.training.flops_counter import FlopsCounter
from relax.utils.training.preference_utils import pack_preference_pair_indices
from relax.utils.types import RolloutBatch

from .cp_utils import (
    dynamic_cp_split_data,
    get_sum_of_sample_mean,
    maybe_padded_total_lengths,
    slice_log_prob_with_cp,
    slice_with_cp,
)


logger = get_logger(__name__)

ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY = "rollout_mini_local_sample_counts"
ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY = "rollout_mini_global_sample_counts"
ROLLOUT_MINI_PROMPT_GROUP_COUNTS_KEY = "rollout_mini_prompt_group_counts"


class PrepackedBatch(dict):
    """Marker dict for SFT micro-batches that have already gone through
    ``get_batch`` on the prefetch worker.

    ``get_batch`` short-circuits on ``isinstance(batch, PrepackedBatch)`` so
    the second pass on the training thread only moves data to device without
    re-doing CP splitting / packed-seq bookkeeping.
    """

    __slots__ = ()


@dataclass(frozen=True)
class RolloutMiniBatchPlan:
    num_rollout_minis: int
    mini_rollout_batch_size: int
    fixed_n_samples_per_prompt: int | None
    mini_global_samples: int | None
    mini_local_sample_request: int | None


def build_rollout_minibatch_plan(args: Namespace, dp_size: int) -> RolloutMiniBatchPlan:
    """Build a prompt-group based mini plan for one rollout partition."""
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")

    rollout_batch_size = int(args.rollout_batch_size)
    n_samples_per_prompt = int(args.n_samples_per_prompt)
    if rollout_batch_size <= 0:
        raise ValueError(f"rollout_batch_size must be positive, got {rollout_batch_size}")
    if n_samples_per_prompt <= 0:
        raise ValueError(f"n_samples_per_prompt must be positive, got {n_samples_per_prompt}")

    num_rollout_minis = getattr(args, "num_steps_per_rollout", None)
    if num_rollout_minis is None:
        global_batch_size = getattr(args, "global_batch_size", None)
        if global_batch_size is None:
            raise ValueError("global_batch_size is required when num_steps_per_rollout is not set")
        total_samples = rollout_batch_size * n_samples_per_prompt
        if total_samples % global_batch_size != 0:
            raise ValueError(
                "rollout_batch_size * n_samples_per_prompt must be divisible by global_batch_size "
                f"when deriving rollout minis, got {total_samples=} and {global_batch_size=}"
            )
        num_rollout_minis = total_samples // global_batch_size

    num_rollout_minis = int(num_rollout_minis)
    if num_rollout_minis <= 0:
        raise ValueError(f"num_rollout_minis must be positive, got {num_rollout_minis}")
    if rollout_batch_size % num_rollout_minis != 0:
        raise ValueError(
            "rollout_batch_size must be divisible by num_rollout_minis, "
            f"got rollout_batch_size={rollout_batch_size}, num_rollout_minis={num_rollout_minis}"
        )

    mini_rollout_batch_size = rollout_batch_size // num_rollout_minis
    if mini_rollout_batch_size <= 0:
        raise ValueError(f"mini_rollout_batch_size must be positive, got {mini_rollout_batch_size}")
    if mini_rollout_batch_size % dp_size != 0:
        raise ValueError(
            "mini_rollout_batch_size must be divisible by data parallel size when using current "
            "sample-count TransferQueue API, "
            f"got mini_rollout_batch_size={mini_rollout_batch_size}, dp_size={dp_size}"
        )

    mini_global_samples = mini_rollout_batch_size * n_samples_per_prompt
    if mini_global_samples % dp_size != 0:
        raise ValueError(
            "mini_rollout_batch_size * n_samples_per_prompt must be divisible by data parallel size, "
            f"got mini_global_samples={mini_global_samples}, dp_size={dp_size}"
        )

    global_batch_size = getattr(args, "global_batch_size", None)
    if global_batch_size is not None and int(global_batch_size) != mini_global_samples:
        raise ValueError(
            "global_batch_size must match the fixed-n_samples rollout mini size in phase 1, "
            f"got global_batch_size={global_batch_size}, expected={mini_global_samples}"
        )

    return RolloutMiniBatchPlan(
        num_rollout_minis=num_rollout_minis,
        mini_rollout_batch_size=mini_rollout_batch_size,
        fixed_n_samples_per_prompt=n_samples_per_prompt,
        mini_global_samples=mini_global_samples,
        mini_local_sample_request=mini_global_samples // dp_size,
    )


def _same_scalar_value(lhs: Any, rhs: Any) -> bool:
    if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
        return torch.equal(lhs, rhs)
    return lhs == rhs


def concat_rollout_batches(rollout_batches: Sequence[RolloutBatch]) -> RolloutBatch:
    """Concatenate rollout mini windows while preserving scalar metadata."""
    if not rollout_batches:
        raise ValueError("rollout_batches must not be empty")

    merged: RolloutBatch = {}
    tensor_batches: dict[str, list[torch.Tensor]] = {}
    scalar_values: dict[str, Any] = {}

    for batch in rollout_batches:
        if batch is None:
            raise ValueError("rollout_batches must not contain None")
        batch_size = len(batch.get("total_lengths", []))
        if batch_size <= 0:
            raise ValueError("rollout mini batch must contain at least one sample")

        for key, value in batch.items():
            if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
                if key not in merged:
                    merged[key] = []
                merged[key].extend(value)
            elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == batch_size:
                tensor_batches.setdefault(key, []).append(value)
            else:
                if key in scalar_values:
                    if not _same_scalar_value(scalar_values[key], value):
                        raise ValueError(f"Scalar rollout field {key!r} differs across mini batches")
                else:
                    scalar_values[key] = value

    for key, tensors in tensor_batches.items():
        if key in merged:
            raise ValueError(f"Rollout field {key!r} appears as both list-like and tensor-like")
        merged[key] = torch.cat(tensors, dim=0)

    for key, value in scalar_values.items():
        if key in merged:
            raise ValueError(f"Rollout field {key!r} appears as both scalar and batched data")
        merged[key] = value

    return merged


PAD_RULES = {
    # shape like [1, 128, 1036]
    "input_features": dict(
        transpose=(0, 2),
        padding_value=0.0,
    ),
    # shape like [1, 1036]
    "feature_attention_mask": dict(
        transpose=(0, 1),
        padding_value=0,
    ),
}


def _round_up_to_microbatch_group(num_microbatches: torch.Tensor, microbatch_group_size: int) -> torch.Tensor:
    return torch.clamp(
        (num_microbatches + microbatch_group_size - 1) // microbatch_group_size * microbatch_group_size,
        min=1,
    )


def _get_seqlen_partitions_with_dummy_padding(
    seqlens: list[int],
    num_partitions: int,
) -> tuple[list[list[int]], set[int]]:
    real_partition_count = min(len(seqlens), num_partitions)
    partitions = get_seqlen_balanced_partitions(seqlens, real_partition_count, equal_size=False)
    dummy_offsets: set[int] = set()
    if real_partition_count < num_partitions:
        shortest_partition = min(partitions, key=lambda partition: sum(seqlens[index] for index in partition))
        for offset in range(real_partition_count, num_partitions):
            partitions.append(list(shortest_partition))
            dummy_offsets.add(offset)
    return partitions, dummy_offsets


def pad_and_flatten(
    tensor_list,
    transpose=None,
    padding_value=0,
):
    """Pad a list of variable-length tensors to the same length and then
    flatten them.

    This function:
    1. Padding is applied along dim=0 by default, pass 'transpose' to swap dimensions
       before and after padding;
    2. Pad all tensors in the tensor list to match the longest tensor in the list.;
    3. Concatenates the padded tensors along the first dimension;
    4. Returns the concatenated tensor and a list of lengths (size of padding dim) for
       each padded tensor.

    Args:
        tensor_list (List[Tensor]):
            A list of tensors. The first dimension is treated as the variable-length
            dimension.
        transpose (Tuple[int, int] or None, optional):
            If not None, each tensor is transposed with `t.transpose(*transpose)`
            before padding, and each padded tensor is transposed back with the same
            arguments afterward.
        padding_value (float or int, optional):
            Value used for padding shorter tensors up to the maximum length.

    Returns:
        Tuple[Tensor, List[int]]:
            - A single concatenated tensor of all padded tensors along dim 0.
            - A list of integers indicating the size of padding dim for each
              padded tensor.

    Note:
        - When `len(tensor_list) == 1`, no padding is performed. The single tensor
          is returned as-is along with its length in a list.
        - This function assumes that all tensors are compatible for padding and
          concatenation after any optional transpose.
    """
    if len(tensor_list) == 1:
        t = tensor_list[0]
        return t, [t.size(0)]

    if transpose is not None:
        tensor_list = [t.transpose(*transpose) for t in tensor_list]

    padded = pad_sequence(
        tensor_list,
        batch_first=True,
        padding_value=padding_value,
    )

    if transpose is not None:
        padded_list = [t.transpose(*transpose) for t in padded]
    else:
        padded_list = [t for t in padded]  # noqa: C416

    num_items = [t.size(0) for t in padded_list]
    return torch.cat(padded_list, dim=0), num_items


def get_batch(
    data_iterator: "DataIterator",
    keys: Sequence[str],
    pad_multiplier: int = 128,
    qkv_format: str = "thd",
    allgather_cp: bool = False,
    is_vl_model: bool = False,
    pack_device: torch.device | None = None,
) -> dict[str, torch.Tensor | PackedSeqParams | list[torch.Tensor] | None]:
    """Generate a CP-ready micro-batch with packed sequence parameters.

    Steps:
    - Fetch raw fields via iterator.
    - Save original token tensors under "unconcat_tokens".
    - Slice tokens into two chunks for Context Parallelism (CP), concatenate, and pad to a configurable multiple.
    - Build cu_seqlens and `PackedSeqParams` with T-H-D layout (T: sequence length, H: attention heads, D: head dimension).

    Args:
        data_iterator: Iterator providing micro-batch data.
        keys: List of keys to fetch from the iterator.
        pad_multiplier: Multiplier for padding size calculation (default: 128).

    Returns a dict including:
    - "tokens": torch.LongTensor of shape [1, T_padded] on the current CUDA device
    - "unconcat_tokens": list[torch.LongTensor] for the micro-batch before CP slicing/concat
    - "packed_seq_params": PackedSeqParams with T-H-D settings (cu_seqlens on CUDA, dtype=int)
    Plus any other requested keys forwarded from the iterator.
    """

    assert "tokens" in keys
    if isinstance(data_iterator, DataIterator):
        batch = data_iterator.get_next(keys)

        if getattr(get_args(), "partial_rollout", False) and getattr(
            get_args(), "use_dynamic_global_batch_size", False
        ):
            step_global_sample_counts = data_iterator.rollout_data.get(ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY)
            if step_global_sample_counts is not None:
                batch["dynamic_global_batch_size"] = int(step_global_sample_counts[0])
            else:
                dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
                batch["dynamic_global_batch_size"] = len(data_iterator.rollout_data["total_lengths"]) * dp_size
        elif "dynamic_global_batch_size" in data_iterator.rollout_data:
            batch["dynamic_global_batch_size"] = data_iterator.rollout_data["dynamic_global_batch_size"]
    else:
        batch, _ = next(data_iterator)

    if isinstance(batch, PrepackedBatch):
        return batch

    if pack_device is not None:
        for key, dtype in (("tokens", torch.long), ("loss_masks", torch.int)):
            values = batch.get(key)
            if values is not None:
                batch[key] = [torch.as_tensor(value, dtype=dtype, device=pack_device) for value in values]

    use_dynamic_context_parallel = getattr(get_args(), "dynamic_context_parallel", False)
    if use_dynamic_context_parallel and pack_device is not None and pack_device.type == "cpu":
        raise ValueError("CPU prepacking does not support dynamic context parallelism.")
    if use_dynamic_context_parallel:
        # Pick this mb's CP size with the SAME per-GPU token budget the iterator was packed
        # with: forward-only iterators carry log_probs_max_tokens_per_gpu, the training
        # iterator carries max_tokens_per_gpu. Using the wrong (smaller) budget would over-
        # estimate CP (fewer DP subdivisions) and lose forward-only throughput. Falls back to
        # args.max_tokens_per_gpu when the iterator didn't record one.
        mt = getattr(data_iterator, "max_tokens_per_gpu", None) or get_args().max_tokens_per_gpu
        cp_size = dynamic_cp_split_data(batch, mt)
        batch["dynamic_cp_size"] = cp_size
        cp_group = mpu.get_dynamic_data_context_parallel_groups(group_size=cp_size)
        cp_rank = dist.get_rank(cp_group)
        batch["dynamic_cp_rank"] = cp_rank
        logger.info(
            f"[dynamic_cp] micro-step dynamic_cp_size={cp_size} (num_samples={len(batch['total_lengths'])}, max_token_per_gpu={mt})"
        )
    else:
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()

    tokens = batch["tokens"]
    batch_device = pack_device or tokens[0].device
    # use 0 as the pad token id should be fine?
    pad_token_id = 0
    pad_size = mpu.get_tensor_model_parallel_world_size() * pad_multiplier
    # for cp, we need all tokens to calculate logprob
    batch["unconcat_tokens"] = tokens

    if qkv_format == "bshd":
        max_seqlen = batch["max_seq_lens"][0]
        assert max([t.size(0) for t in tokens]) <= max_seqlen

        # For VL models with CP > 1, Bridge expects UNSPLIT tokens (it handles CP
        # splitting internally after vision embedding).  Save padded-but-unsplit
        # tokens so model.py can pass them to Bridge instead of the CP-split ones.
        if cp_size > 1:
            chunk_size = (max_seqlen + 2 * cp_size - 1) // (2 * cp_size)
            padded_len = 2 * cp_size * chunk_size
            unsplit = [F.pad(t, (0, padded_len - t.size(0)), value=pad_token_id) for t in tokens]
            batch["unsplit_tokens"] = torch.stack(unsplit)

        tokens = [slice_with_cp(t, pad_token_id, qkv_format, max_seqlen) for t in tokens]
        tokens = torch.stack(tokens)
        packed_seq_params = None

    elif qkv_format == "thd":
        # bridge Qwen3VLModel.forward (used for Qwen3-VL and text-only
        # Qwen3.5 / Qwen3.6 sharing the same architecture) expects per-sample
        # BSHD-padded input_ids + attention_mask, and re-derives the THD
        # packing internally with align_size = tp*cp*2.  Provide unsplit
        # inputs and a matching packed_seq_params so the caller-side cu_seqlens
        # agrees with what the bridge derives from attention_mask.
        # Mirrors verl's build_vlm_attn_mask_thd + preprocess_thd_engine.
        # Routed by `is_vl_model` (set from hf_config) rather than presence of
        # multimodal_train_inputs, so VL models with text-only batches still
        # land here — bridge skips vision embedding when image_grid_thw is None.
        has_mm_inputs = batch.get("multimodal_train_inputs") is not None
        needs_unsplit_input = is_vl_model or has_mm_inputs or getattr(get_args(), "uses_unsplit_forward", False)
        if needs_unsplit_input and cp_size > 1:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            align_size = tp_size * cp_size * 2
            device = batch_device

            seqlens = [t.size(0) for t in tokens]
            seqlens_padded = [(s + align_size - 1) // align_size * align_size for s in seqlens]
            cu_seqlens_padded = torch.zeros(len(tokens) + 1, dtype=torch.int32, device=device)
            cu_seqlens_padded[1:] = torch.cumsum(torch.tensor(seqlens_padded, dtype=torch.int32, device=device), dim=0)
            max_seqlen_padded = max(seqlens_padded)

            unsplit_tokens = pad_sequence(tokens, batch_first=True, padding_value=pad_token_id)
            # Bridge treats a single-row input as an already packed THD stream.
            # Its physical length must match cu_seqlens, including TP/CP padding.
            unsplit_tokens = F.pad(unsplit_tokens, (0, max_seqlen_padded - unsplit_tokens.size(1)), value=pad_token_id)
            unsplit_attention_mask = torch.zeros_like(unsplit_tokens, dtype=torch.bool)
            for i, s in enumerate(seqlens):
                unsplit_attention_mask[i, :s] = True

            batch["unsplit_tokens"] = unsplit_tokens
            batch["unsplit_attention_mask"] = unsplit_attention_mask
            vlm_packed_seq_params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens_padded,
                cu_seqlens_kv=cu_seqlens_padded,
                max_seqlen_q=max_seqlen_padded,
                max_seqlen_kv=max_seqlen_padded,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv_padded=cu_seqlens_padded,
            )
            if use_dynamic_context_parallel:
                vlm_packed_seq_params.local_cp_size = cp_size
                vlm_packed_seq_params.cp_group = cp_group
            batch["vlm_packed_seq_params"] = vlm_packed_seq_params
            # Per-sample tp*cp*2-aligned lengths consumed by loss helpers so
            # their per-sample chunking matches bridge's preprocess_packed_seqs.
            batch["padded_total_lengths"] = seqlens_padded

        if allgather_cp:
            # DSA mode: concatenate all sequences first, then slice once with CP.
            # We also pad the *global* concatenated stream to make per-rank chunks equal.
            cu_seqlens_list: list[int] = [0]
            for t in tokens:
                cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))

            tokens = torch.cat(tokens, dim=0)

            # Pad global stream so (1) divisible by cp_size (equal chunks),
            # (2) divisible by pad_size (reduce fragmentation).
            global_pad_size = cp_size * pad_size
            pad = (global_pad_size - tokens.size(0) % global_pad_size) % global_pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens_list.append(cu_seqlens_list[-1] + pad)

            cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=batch_device)
            cu_seqlens_cpu = cu_seqlens_list
            tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
        else:
            tokens = [
                slice_with_cp(t, pad_token_id, qkv_format, dynamic_cp_size=cp_size, dynamic_cp_rank=cp_rank)
                for t in tokens
            ]

            cu_seqlens = [0]
            for t in tokens:
                cu_seqlens.append(cu_seqlens[-1] + t.size(0))

            tokens = torch.cat(tokens)

            # Always pad to reduce memory fragmentation and maybe make the computation faster
            pad = (pad_size - tokens.size(0) % pad_size) % pad_size
            if pad != 0:
                tokens = F.pad(tokens, (0, pad), value=pad_token_id)
                cu_seqlens.append(cu_seqlens[-1] + pad)

            # thd requires the cu_seqlens to be of the origin length
            cu_seqlens_cpu = [offset * cp_size for offset in cu_seqlens]
            cu_seqlens = torch.tensor(cu_seqlens_cpu, dtype=torch.int, device=batch_device)

        max_seqlen = max(end - start for start, end in zip(cu_seqlens_cpu, cu_seqlens_cpu[1:]))
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )
        # Python boundaries let attention implementations iterate packed
        # subsequences without synchronizing individual accelerator scalars.
        packed_seq_params.cu_seqlens_q_cpu = cu_seqlens_cpu
        if use_dynamic_context_parallel:
            packed_seq_params.local_cp_size = cp_size
            packed_seq_params.cp_group = cp_group

        tokens = tokens.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported qkv_format: {qkv_format}")

    batch["tokens"] = tokens
    batch["packed_seq_params"] = packed_seq_params

    from relax.utils.sft_utils import align_loss_mask_for_sft

    batch["raw_loss_masks"] = list(batch["loss_masks"])
    loss_masks: list[torch.Tensor] = []
    per_sample_loss_masks: list[torch.Tensor] = []
    full_per_sample_loss_masks: list[torch.Tensor] = []
    for loss_mask, total_length, response_length in zip(
        batch["loss_masks"], batch["total_lengths"], batch["response_lengths"], strict=True
    ):
        if response_length == total_length:
            loss_mask = align_loss_mask_for_sft(loss_mask)
            per_sample_loss_masks.append(loss_mask)
        else:
            per_sample_loss_masks.append(loss_mask)
            prompt_length = total_length - response_length
            loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
        # Pre-CP, per-sample full-length mask used for the bridge-aligned MTP
        # labels/mask below (built after this loop closes).
        full_per_sample_loss_masks.append(loss_mask)
        if not allgather_cp:
            loss_mask = slice_with_cp(
                loss_mask, 0, qkv_format, max_seqlen, dynamic_cp_size=cp_size, dynamic_cp_rank=cp_rank
            )
        loss_masks.append(loss_mask)
    batch["loss_masks"] = per_sample_loss_masks

    if qkv_format == "bshd":
        loss_masks = torch.stack(loss_masks)
    elif qkv_format == "thd" and allgather_cp:
        # DSA: concatenate first (same as tokens), pad globally (same pad as above), then slice once.
        loss_masks = torch.cat(loss_masks, dim=0)
        if pad != 0:
            loss_masks = F.pad(loss_masks, (0, pad), value=0)
        loss_masks = loss_masks.chunk(cp_size, dim=0)[cp_rank].unsqueeze(0)
    elif qkv_format == "thd":
        loss_masks = torch.cat(loss_masks)
        loss_masks = F.pad(loss_masks, (0, pad), value=0).unsqueeze(0)

    assert loss_masks.shape == tokens.shape, f"loss_masks.shape: {loss_masks.shape}, tokens.shape: {tokens.shape}"
    batch["full_loss_masks"] = loss_masks

    # Bridge-aligned MTP labels/loss_mask for the VL+THD+CP unsplit path.
    # Legacy `batch["tokens"]` / `batch["full_loss_masks"]` use per-sample
    # align=2*cp_size + global pad, but the bridge's preprocess_packed_seqs
    # repacks hidden_states with per-sample align=tp*cp*2 (matching
    # vlm_packed_seq_params). The two per-rank lengths diverge, so MTP labels
    # must mirror the bridge layout: per-sample pad to seqlens_padded[i],
    # then CP-slice with the standard 2-chunk pattern, then concat.
    if qkv_format == "thd" and "vlm_packed_seq_params" in batch and getattr(get_args(), "enable_mtp_training", False):
        orig_tokens = batch["unconcat_tokens"]
        seqlens_padded_list = batch["padded_total_lengths"]
        mtp_label_chunks: list[torch.Tensor] = []
        mtp_loss_chunks: list[torch.Tensor] = []
        for sample_tokens, sample_mask, pad_to in zip(
            orig_tokens, full_per_sample_loss_masks, seqlens_padded_list, strict=True
        ):
            pad_to = int(pad_to)
            t_padded = F.pad(sample_tokens, (0, pad_to - sample_tokens.size(0)), value=pad_token_id)
            m_padded = F.pad(sample_mask, (0, pad_to - sample_mask.size(0)), value=0)
            # This VL/THD path bypasses MCore's default label construction, so
            # labels must be pre-shifted by one here (label[t] = x[t+1]); MCore's
            # per-depth roll advances label and mask together afterwards.
            #
            # The mask must NOT be shifted: ``m_padded`` (from
            # ``full_per_sample_loss_masks`` via ``align_loss_mask_for_sft``) is
            # already in the same next-token-aligned frame as the shifted label.
            # Shifting it again would leave it leading the label by one token, so
            # each MTP depth would supervise the position one past what it predicts.
            mtp_labels_padded = F.pad(t_padded[1:], (0, 1), value=pad_token_id)
            mtp_loss_mask_padded = m_padded
            chunk = pad_to // (2 * cp_size)
            s1, e1 = chunk * cp_rank, chunk * (cp_rank + 1)
            s2, e2 = chunk * (2 * cp_size - cp_rank - 1), chunk * (2 * cp_size - cp_rank)
            mtp_label_chunks.append(torch.cat([mtp_labels_padded[s1:e1], mtp_labels_padded[s2:e2]]))
            mtp_loss_chunks.append(torch.cat([mtp_loss_mask_padded[s1:e1], mtp_loss_mask_padded[s2:e2]]))
        batch["unsplit_mtp_labels"] = torch.cat(mtp_label_chunks).unsqueeze(0)
        batch["unsplit_mtp_loss_mask"] = torch.cat(mtp_loss_chunks).unsqueeze(0)

    # Process multimodal training tensors if present
    multimodal_train_inputs = batch.get("multimodal_train_inputs", None)
    if multimodal_train_inputs is not None:
        multimodal_data = {}  # key -> concatenated tensor
        multimodal_num_items = {}  # key -> list of item counts per sequence
        tensor_dict_list = {}

        for mm_input_dict in multimodal_train_inputs:
            if mm_input_dict is None:
                continue
            for key, mm_tensor in mm_input_dict.items():
                if isinstance(mm_tensor, list):
                    mm_tensor = torch.tensor(mm_tensor)
                tensor_dict_list.setdefault(key, []).append(mm_tensor)

        for key, tensor_list in tensor_dict_list.items():
            if key in PAD_RULES:
                multimodal_data[key], multimodal_num_items[key] = pad_and_flatten(
                    tensor_list,
                    **PAD_RULES[key],
                )
            else:
                if len(tensor_list) == 1:
                    multimodal_data[key] = tensor_list[0]
                else:
                    multimodal_data[key] = torch.cat(tensor_list, dim=0)
                multimodal_num_items[key] = [t.size(0) for t in tensor_list]

        batch["multimodal_train_inputs"] = multimodal_data
        batch["multimodal_num_items"] = multimodal_num_items

    # Dynamic CP: forward_only merged its per-sample outputs back to full-length
    # responses (dynamic_cp_merge_output), so rollout_data stores full fields.
    # Re-slice them into this mb's CP-local zig-zag layout (with the dynamic cp_size/
    # cp_rank) so they line up with the CP-local log_probs the training forward emits.
    # Reuses batch["padded_total_lengths"] (dynamic-cp-aligned above) — the same value
    # the forward pass consumes — so chunk boundaries match. forward_only never fetches
    # these fields, so this is a no-op there; cp_size == 1 (pure DP) needs no slicing.
    if use_dynamic_context_parallel and cp_size > 1:
        padded_total_lengths = batch.get("padded_total_lengths")
        ptls = padded_total_lengths if padded_total_lengths is not None else [None] * len(batch["total_lengths"])
        _dcp_response_fields = (
            "log_probs",
            "ref_log_probs",
            "rollout_log_probs",
            "advantages",
            "teacher_log_probs",
            "opd_reverse_kl",
        )
        for key in _dcp_response_fields:
            vals = batch.get(key)
            # slice_log_prob_with_cp accepts list[float] | Tensor, so (unlike merge's
            # all_gather) no tensor check: mirror split's type-agnostic per-sample check.
            if isinstance(vals, (list, tuple)) and len(vals) == len(batch["total_lengths"]):
                batch[key] = [
                    slice_log_prob_with_cp(
                        v,
                        tl,
                        rl,
                        qkv_format,
                        padded_total_length=ptl,
                        dynamic_cp_size=cp_size,
                        dynamic_cp_rank=cp_rank,
                    )
                    for v, tl, rl, ptl in zip(
                        vals, batch["total_lengths"], batch["response_lengths"], ptls, strict=False
                    )
                ]

    batch = move_tensors_to_device(batch, batch["tokens"].device)
    if pack_device is not None:
        batch = PrepackedBatch(batch)
    return batch


def prepack_sft_micro_batch_cpu(args: Namespace, batch: RolloutBatch) -> PrepackedBatch:
    """Build one complete SFT micro-batch on CPU for background prefetch."""
    iterator = iter([(batch, None)])
    packed = get_batch(
        iterator,
        ["tokens"],
        args.data_pad_size_multiplier,
        args.qkv_format,
        args.allgather_cp,
        getattr(args, "is_vl_model", False),
        pack_device=torch.device("cpu"),
    )
    pinned = pin_tensors(packed)
    return pinned if isinstance(pinned, PrepackedBatch) else PrepackedBatch(pinned)


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, float],
    metric_weights: dict[str, int] | None = None,
) -> dict[str, float] | None:
    """Gather per-rank metrics, reduce on the DP source rank, and log.

    Expects `log_dict` to contain plain scalars. `metric_weights` provides
    local observation counts for metrics whose DP shards have different sizes.
    The DP source rank prints and optionally logs to WandB/TensorBoard with a
    step derived from `rollout_id` and batch sizes. Returns the reduced dict on
    the DP source rank; returns None on others.
    """

    if mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)

        gathered_metrics = [None] * dp_size
        # Not sure if this will be a performance bottleneck.
        dist.gather_object(
            (log_dict, metric_weights or {}),
            gathered_metrics,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )

        reduced_log_dict = {}
        for key in log_dict:
            weights = [weights_by_key.get(key, 1) for _, weights_by_key in gathered_metrics]
            reduced_log_dict[f"{metric_name}/{key}"] = sum(
                metrics[key] * weight for (metrics, _), weight in zip(gathered_metrics, weights, strict=True)
            ) / sum(weights)
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        tracking_utils.log(args, reduced_log_dict, step_key="rollout/step")

        return reduced_log_dict
    else:
        dist.gather_object(
            (log_dict, metric_weights or {}),
            None,
            dst=mpu.get_data_parallel_src_rank(with_context_parallel=True),
            group=mpu.get_data_parallel_group_gloo(with_context_parallel=True),
        )
        return None


class DataIterator:
    """Micro-batch iterator over rollout dicts.

    Supports either fixed contiguous micro-batches or an explicit per-step
    index schedule (for dynamic batch sizing / sequence-length balancing).
    """

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_size: int | None = None,
        micro_batch_indices: list[list[int]] | None = None,
        max_tokens_per_gpu: int | None = None,
        dummy_micro_batch_offsets: set[int] | None = None,
    ) -> None:
        """Initialize an iterator over `rollout_data`.

        Args:
            rollout_data: Dict of per-sample fields for the local step.
            micro_batch_size: Fixed contiguous slice size when not using dynamic scheduling.
            micro_batch_indices: Explicit indices per micro-batch when using dynamic balancing.
                Must be mutually exclusive with `micro_batch_size`.
            max_tokens_per_gpu: Per-GPU token budget this iterator was built with. Dynamic CP
                reads it in get_batch to pick each mb's CP size consistently with how the
                micro-batches were packed (forward-only uses log_probs_max_tokens_per_gpu,
                training uses max_tokens_per_gpu). None falls back to args.max_tokens_per_gpu.
            dummy_micro_batch_offsets: Microbatch offsets that repeat real rows only to align
                the number of microbatches across data-parallel ranks.
        """
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.dummy_micro_batch_offsets = dummy_micro_batch_offsets or set()
        assert micro_batch_size is None or micro_batch_indices is None
        self.offset = 0

    def get_next(self, keys: Sequence[str]) -> dict[str, Any]:
        """Return the next micro-batch for the requested keys.

        - If `micro_batch_indices` is provided, selects rows according to the current
          index list for each requested key.
        - Otherwise, slices a contiguous window of size `micro_batch_size` starting
          at the current offset.

        Returns a dict mapping each key to a list subset (or None if absent).
        """
        if self.micro_batch_indices is not None:
            micro_batch_index = self.offset
        else:
            micro_batch_index = self.offset // self.micro_batch_size

        batch = {}
        for key in keys:
            vals = self.rollout_data.get(key, None)
            if vals is None:
                batch[key] = None
            else:
                if self.micro_batch_indices is not None:
                    indices = self.micro_batch_indices[self.offset]
                    batch[key] = [vals[i] for i in indices]
                else:
                    assert self.offset + self.micro_batch_size <= len(vals), (
                        f"offset: {self.offset}, micro_batch_size: {self.micro_batch_size}, len(vals): {len(vals)}"
                    )
                    batch[key] = vals[self.offset : self.offset + self.micro_batch_size]

        if self.micro_batch_indices is not None:
            if self.offset in self.dummy_micro_batch_offsets:
                batch["__is_dummy__"] = True
            self.offset += 1
        else:
            self.offset += self.micro_batch_size
        # DataIterator micro-batch ordinal (0-based). Trajectory replay stamps
        # this onto each sample so --batch selects a real training micro-batch.
        batch["replay_micro_batch_index"] = micro_batch_index
        return batch

    def reset(self) -> "DataIterator":
        """Reset internal offset to the start and return self."""
        self.offset = 0
        return self


def expand_preference_rollout_data(rollout_data: RolloutBatch) -> RolloutBatch:
    """Expand pair rows into adjacent chosen/rejected model sequences."""
    if "preference_pair_costs" in rollout_data:
        return rollout_data
    required = (
        "pair_ids",
        "chosen_tokens",
        "rejected_tokens",
        "chosen_loss_masks",
        "rejected_loss_masks",
        "chosen_total_lengths",
        "rejected_total_lengths",
        "chosen_score_positions",
        "rejected_score_positions",
    )
    missing = [key for key in required if key not in rollout_data]
    if missing:
        raise ValueError(f"preference rollout data is missing fields: {missing}")
    pair_count = len(rollout_data["pair_ids"])
    if pair_count <= 0:
        raise ValueError("preference rollout batch must contain at least one pair")
    for key in required:
        if len(rollout_data[key]) != pair_count:
            raise ValueError(
                f"preference field {key!r} is not pair-row aligned: expected {pair_count}, got {len(rollout_data[key])}"
            )

    flat: RolloutBatch = {
        "tokens": [],
        "loss_masks": [],
        "total_lengths": [],
        "response_lengths": [],
        "score_positions": [],
        "preference_branch_pair_ids": [],
        "preference_is_chosen": [],
    }
    pair_costs: list[int] = []
    pair_ids: list[int] = []
    for index in range(pair_count):
        chosen_length = int(rollout_data["chosen_total_lengths"][index])
        rejected_length = int(rollout_data["rejected_total_lengths"][index])
        if chosen_length <= 0 or rejected_length <= 0:
            raise ValueError(f"preference pair row {index} has non-positive branch length")
        pair_id = int(rollout_data["pair_ids"][index])
        pair_ids.append(pair_id)
        pair_costs.append(chosen_length + rejected_length)
        for prefix, is_chosen in (("chosen", True), ("rejected", False)):
            tokens = rollout_data[f"{prefix}_tokens"][index]
            loss_mask = rollout_data[f"{prefix}_loss_masks"][index]
            total_length = int(rollout_data[f"{prefix}_total_lengths"][index])
            score_position = int(rollout_data[f"{prefix}_score_positions"][index])
            if len(tokens) != total_length or len(loss_mask) != total_length:
                raise ValueError(
                    f"preference pair row {index} {prefix} tensor length does not match declared total_length"
                )
            if not 0 <= score_position < total_length:
                raise ValueError(f"preference pair row {index} {prefix} score position is out of range")
            flat["tokens"].append(tokens)
            flat["loss_masks"].append(loss_mask)
            flat["total_lengths"].append(total_length)
            flat["response_lengths"].append(total_length)
            flat["score_positions"].append(score_position)
            flat["preference_branch_pair_ids"].append(pair_id)
            flat["preference_is_chosen"].append(is_chosen)
    flat["preference_pair_costs"] = pair_costs
    flat["preference_pair_ids"] = pair_ids
    if "dynamic_global_batch_size" in rollout_data:
        flat["dynamic_global_batch_size"] = rollout_data["dynamic_global_batch_size"]
    return flat


def _split_preference_bins_to_count(bins: list[list[int]], target_count: int) -> list[list[int]]:
    bins = [list(group) for group in bins]
    while len(bins) < target_count:
        candidates = [(len(group), -index, index) for index, group in enumerate(bins) if len(group) > 1]
        if not candidates:
            raise RuntimeError(
                f"cannot split {len(bins)} preference micro-batches to DP-synchronized count {target_count}"
            )
        _, _, index = max(candidates)
        group = bins[index]
        bins[index] = group[:-1]
        bins.insert(index + 1, [group[-1]])
    return bins


def _get_preference_data_iterator(
    args: Namespace,
    rollout_data: RolloutBatch,
    max_tokens_per_gpu: int | None,
) -> tuple[list[DataIterator], list[int]]:
    pair_costs = [int(cost) for cost in rollout_data["preference_pair_costs"]]
    pair_ids = [str(pair_id) for pair_id in rollout_data["preference_pair_ids"]]
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dynamic_count = rollout_data.get("dynamic_global_batch_size")
    if isinstance(dynamic_count, (list, tuple)):
        normalized = {int(value) for value in dynamic_count}
        if len(normalized) != 1:
            raise ValueError(f"dynamic_global_batch_size values must be identical, got {dynamic_count}")
        dynamic_count = normalized.pop()
    expected_global_pairs = int(args.global_batch_size) if dynamic_count is None else int(dynamic_count)
    capacity = int(max_tokens_per_gpu or args.max_tokens_per_gpu)
    pair_bins = pack_preference_pair_indices(pair_costs, pair_ids, capacity=capacity)
    control_values = [None] * dp_size
    dist.all_gather_object(
        control_values,
        (len(pair_costs), expected_global_pairs, len(pair_bins)),
        group=mpu.get_data_parallel_group_gloo(with_context_parallel=False),
    )
    local_pair_counts = {int(values[0]) for values in control_values}
    if len(local_pair_counts) != 1:
        raise ValueError(f"preference objectives require equal local pair rows on every DP rank: {local_pair_counts}")
    declared_global_pairs = {int(values[1]) for values in control_values}
    if len(declared_global_pairs) != 1:
        raise ValueError(f"dynamic_global_batch_size must be identical on every DP rank: {declared_global_pairs}")
    step_global_pair_count = local_pair_counts.pop() * dp_size
    declared_global_pair_count = declared_global_pairs.pop()
    if declared_global_pair_count != step_global_pair_count:
        raise ValueError(
            "dynamic_global_batch_size must equal the step-global preference pair count: "
            f"declared={declared_global_pair_count}, actual={step_global_pair_count}, "
            f"local={len(pair_costs)}, dp_size={dp_size}"
        )
    rollout_data["dynamic_global_batch_size"] = step_global_pair_count
    rollout_data[ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY] = [step_global_pair_count]
    pair_bins = _split_preference_bins_to_count(pair_bins, max(int(values[2]) for values in control_values))
    if any(sum(pair_costs[index] for index in group) > capacity for group in pair_bins):
        raise RuntimeError("preference DP bin synchronization produced an over-capacity micro-batch")
    branch_bins = [
        [branch for pair_index in group for branch in (2 * pair_index, 2 * pair_index + 1)] for group in pair_bins
    ]
    covered = [index for group in pair_bins for index in group]
    if sorted(covered) != list(range(len(pair_costs))):
        raise RuntimeError("preference dynamic batching lost or duplicated a pair")
    for group in branch_bins:
        if len(group) % 2 != 0 or any(group[index + 1] != group[index] + 1 for index in range(0, len(group), 2)):
            raise RuntimeError("preference dynamic batching split a chosen/rejected pair")
    iterator = DataIterator(rollout_data, micro_batch_indices=branch_bins, max_tokens_per_gpu=capacity)
    return [iterator], [len(branch_bins)]


def get_data_iterator(
    args: Namespace,
    model: torch.nn.Module | Sequence[torch.nn.Module],
    rollout_data: RolloutBatch,
    max_tokens_per_gpu: int | None = None,
) -> tuple[list[DataIterator], list[int]]:
    """Create iterators and a micro-batch schedule for a rollout step.

    - If `use_dynamic_batch_size` is False, splits into fixed-size contiguous
      micro-batches of `micro_batch_size`.
    - If True, computes the number of micro-batches per local step based on
      `max_tokens_per_gpu` (or the override passed via the parameter) and per-sample lengths, all-reduces to a DP-wide
      maximum, optionally enforces divisibility for Virtual Pipeline Parallelism (VPP), and builds a balanced
      index schedule to equalize token counts across micro-batches.

    Returns `(data_iterators, num_microbatches)` where:
    - `data_iterators`: list of `DataIterator`, one per VPP stage (size 1 if VPP disabled)
    - `num_microbatches`: list[int], one per local step in the rollout (length = steps)
    """
    if is_preference_mode(args):
        return _get_preference_data_iterator(args, rollout_data, max_tokens_per_gpu)

    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dp_group = mpu.get_data_parallel_group()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1
    if vpp_size > 1:
        from megatron.core.utils import get_model_config

        config = get_model_config(model[0])
        microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
    cp_size = mpu.get_context_parallel_world_size()

    num_local_samples = len(rollout_data["total_lengths"])
    step_global_sample_counts = rollout_data.get(ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY)
    if getattr(args, "partial_rollout", False) and getattr(args, "use_dynamic_global_batch_size", False):
        if step_global_sample_counts is not None:
            if not isinstance(step_global_sample_counts, list) or len(step_global_sample_counts) != 1:
                raise ValueError(
                    f"{ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY} must contain exactly one logical sample count "
                    f"for partial dynamic-global training, got {step_global_sample_counts}"
                )
            global_batch_size = int(step_global_sample_counts[0])
        else:
            global_batch_size = num_local_samples * dp_size
    else:
        global_batch_size = rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
    num_local_gbs = global_batch_size // dp_size
    step_local_sample_counts = rollout_data.get(ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY)

    if step_local_sample_counts is not None:
        if not isinstance(step_local_sample_counts, list) or not step_local_sample_counts:
            raise ValueError(f"{ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY} must be a non-empty list")
        if any((not isinstance(count, int)) or count <= 0 for count in step_local_sample_counts):
            raise ValueError(
                f"{ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY} must contain positive integers, "
                f"got {step_local_sample_counts}"
            )
        if sum(step_local_sample_counts) != num_local_samples:
            raise ValueError(
                f"sum({ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY}) must equal num_local_samples, "
                f"got counts={step_local_sample_counts}, num_local_samples={num_local_samples}"
            )
        num_steps_per_rollout = len(step_local_sample_counts)
    elif num_local_samples <= num_local_gbs:
        num_local_gbs = num_local_samples
        num_steps_per_rollout = 1
    else:
        if num_local_samples % num_local_gbs != 0:
            raise ValueError(
                "num_local_samples must be divisible by local global-batch size when explicit rollout mini "
                f"boundaries are absent, got num_local_samples={num_local_samples}, num_local_gbs={num_local_gbs}"
            )
        num_steps_per_rollout = num_local_samples // num_local_gbs

    # With balance_data, synchronise num_steps_per_rollout across DP ranks so
    # collectives stay aligned. Explicit rollout-mini boundaries or divisible
    # fixed-size local steps must already produce the same count on every rank;
    # otherwise fail instead of generating empty batches.
    if getattr(args, "balance_data", False):
        local_num_steps_per_rollout = num_steps_per_rollout
        steps_tensor = torch.tensor(
            [num_steps_per_rollout], dtype=torch.int, device=device_utils.make_current_torch_device()
        )
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.MAX, group=dp_group)
        num_steps_per_rollout = steps_tensor.item()
        if num_steps_per_rollout != local_num_steps_per_rollout:
            raise RuntimeError(
                "training step count differs across DP ranks under balance_data: "
                f"local={local_num_steps_per_rollout}, global_max={num_steps_per_rollout}"
            )

    if step_local_sample_counts is None:
        step_local_sample_counts = [num_local_gbs for _ in range(num_steps_per_rollout)]

    if step_global_sample_counts is None and global_batch_size != args.global_batch_size:
        step_global_sample_counts = [global_batch_size for _ in range(num_steps_per_rollout)]
    if step_global_sample_counts is not None:
        rollout_data[ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY] = [int(count) for count in step_global_sample_counts]

    if global_batch_size != args.global_batch_size:
        logger.info(
            f"Using dynamic global_batch_size={global_batch_size} (original={args.global_batch_size}), "
            f"num_local_samples={num_local_samples}, num_steps_per_rollout={num_steps_per_rollout}"
        )

    def _generate_data_iterator(
        rollout_data,
        micro_batch_size,
        micro_batch_indices=None,
        max_tokens_per_gpu=None,
        dummy_micro_batch_offsets=None,
    ):
        data_iterator = []
        for _ in range(vpp_size):
            data_iterator.append(
                DataIterator(
                    rollout_data,
                    micro_batch_size,
                    micro_batch_indices,
                    max_tokens_per_gpu=max_tokens_per_gpu,
                    dummy_micro_batch_offsets=dummy_micro_batch_offsets,
                )
            )
        return data_iterator

    if not args.use_dynamic_batch_size:
        invalid_counts = [count for count in step_local_sample_counts if count % args.micro_batch_size != 0]
        if invalid_counts:
            raise ValueError(
                "Each rollout mini local sample count must be divisible by micro_batch_size, "
                f"got invalid_counts={invalid_counts}, micro_batch_size={args.micro_batch_size}"
            )
        num_microbatches = [count // args.micro_batch_size for count in step_local_sample_counts]
        data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    else:
        _max_tokens = max_tokens_per_gpu if max_tokens_per_gpu is not None else args.max_tokens_per_gpu
        assert _max_tokens is not None
        # calculate the number of mirobatches for each step
        samples = rollout_data["total_lengths"]
        assert len(samples) == num_local_samples
        num_microbatches = []
        step_offsets = np.cumsum([0, *step_local_sample_counts]).tolist()
        for i in range(num_steps_per_rollout):
            start = step_offsets[i]
            end = step_offsets[i + 1]
            num_microbatches.append(get_minimum_num_micro_batch_size(samples[start:end], _max_tokens * cp_size))

        required_num_microbatches = torch.tensor(
            num_microbatches, dtype=torch.int, device=device_utils.make_current_torch_device()
        )
        dist.all_reduce(required_num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)
        num_microbatches = required_num_microbatches

        if vpp_size > 1:
            # vpp requires the number of microbatches to be divisible by vpp_size
            num_microbatches = _round_up_to_microbatch_group(num_microbatches, microbatch_group_size_per_vp_stage)

        num_microbatches = num_microbatches.tolist()

        # balance the each micro batch
        samples = rollout_data["total_lengths"]
        # balance the number of mirobatches across steps
        micro_batch_indices = []
        dummy_micro_batch_offsets: set[int] = set()
        for i, num_mbs in enumerate(num_microbatches):
            start = step_offsets[i]
            end = step_offsets[i + 1]
            step_samples = samples[start:end]
            partitions, local_dummy_offsets = _get_seqlen_partitions_with_dummy_padding(step_samples, num_mbs)
            base_offset = len(micro_batch_indices)
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] = start + partitions[j][k]
            micro_batch_indices.extend(partitions)
            dummy_micro_batch_offsets.update(base_offset + offset for offset in local_dummy_offsets)

        if getattr(args, "dynamic_context_parallel", False):
            # Dynamic CP: within each step, order micro-batches by their longest
            # packed sub-sequence, ascending (shortest first).
            total_lengths = rollout_data["total_lengths"]
            step_mb_offsets = np.cumsum([0, *num_microbatches]).tolist()
            ordered: list[list[int]] = []
            ordered_dummy_offsets: set[int] = set()
            for s in range(len(num_microbatches)):
                block = [
                    (micro_batch_indices[offset], offset in dummy_micro_batch_offsets)
                    for offset in range(step_mb_offsets[s], step_mb_offsets[s + 1])
                ]
                block.sort(key=lambda item: max((total_lengths[i] for i in item[0]), default=0))
                for micro_batch, is_dummy in block:
                    if is_dummy:
                        ordered_dummy_offsets.add(len(ordered))
                    ordered.append(micro_batch)
            micro_batch_indices = ordered
            dummy_micro_batch_offsets = ordered_dummy_offsets

        assert len(set(sum(micro_batch_indices, []))) == num_local_samples
        logger.info(
            f"After dynamic batching, num_microbatches: {num_microbatches}, micro_batch_indices: {micro_batch_indices}, "
            f"dummy_microbatches={len(dummy_micro_batch_offsets)}"
        )
        data_iterator = _generate_data_iterator(
            rollout_data,
            None,
            micro_batch_indices,
            max_tokens_per_gpu=_max_tokens,
            dummy_micro_batch_offsets=dummy_micro_batch_offsets,
        )

    return (
        data_iterator,
        num_microbatches,
    )


def log_rollout_data(
    rollout_id: int,
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    """Summarize rollout fields and log reduced metrics on PP last stage, TP
    rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        # Under dynamic CP, rollout_data was merged back to full-length responses
        # (dynamic_cp_merge_output), so metrics here run on full data — use cp=1 (no
        # zig-zag chunking / no * cp_size scaling), matching the merged layout.
        cp_size = 1 if getattr(args, "dynamic_context_parallel", False) else mpu.get_context_parallel_world_size()
        log_dict = {}
        metric_weights = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        sample_index_mask_sums = rollout_data.get("sample_index_mask_sums", None)
        step_global_sample_counts = rollout_data.get(ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY)
        if step_global_sample_counts is not None:
            global_identity_count = sum(int(count) for count in step_global_sample_counts)
        elif "dynamic_global_batch_size" in rollout_data:
            global_identity_count = int(rollout_data["dynamic_global_batch_size"])
        else:
            global_identity_count = int(args.rollout_batch_size) * int(args.n_samples_per_prompt)
        max_seq_lens = rollout_data.get("max_seq_lens", None)
        padded_total_lengths = maybe_padded_total_lengths(
            total_lengths,
            args.qkv_format,
            getattr(args, "is_vl_model", False)
            or rollout_data.get("multimodal_train_inputs") is not None
            or getattr(args, "uses_unsplit_forward", False),
        )

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "sample_index_mask_sums",
                "rollout_routed_experts",
                "max_seq_lens",
                "dynamic_global_batch_size",
                "packed_seq_params",
                "vlm_packed_seq_params",
                "__loss_scale__",
                ROLLOUT_MINI_LOCAL_SAMPLE_COUNTS_KEY,
                ROLLOUT_MINI_GLOBAL_SAMPLE_COUNTS_KEY,
                ROLLOUT_MINI_PROMPT_GROUP_COUNTS_KEY,
                "preference_pair_costs",
                "preference_pair_ids",
                "preference_branch_pair_ids",
                "preference_is_chosen",
                "score_positions",
            ]:
                continue
            if args.use_opd and key in OPD_ROLLOUT_LOG_SKIP_FIELDS:
                continue
            # Upload per-sample means. Plain numeric row metrics carry their
            # local row counts into the DP reduction because identity-aware
            # placement can assign different physical row counts to each DP.
            if isinstance(val, (list, tuple)):
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    use_sample_mean = key in [
                        "rewards",
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "teacher_log_probs",
                        "opd_kl_term",
                    ]
                    if use_sample_mean:
                        val = torch.cat(val).clone().detach()
                        val = val.to(loss_masks[0].device)
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                            padded_total_lengths=padded_total_lengths,
                            dynamic_cp_size=cp_size,
                            sample_denoms=sample_index_mask_sums,
                        )
                        dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
                        val = cp_size * sum_of_sample_mean(val) * dp_size / global_identity_count
                    else:
                        try:
                            val = torch.cat(val).clone().detach().float()
                        except RuntimeError:
                            # Tensors have mismatched shapes (e.g. variable-length mbs in
                            # streaming mode) — fall back to per-mb mean then average.
                            val = torch.stack([v.float().mean() for v in val])
                        val = val.mean() * cp_size
                else:
                    if not isinstance(val[0], (int, float)):
                        continue
                    metric_weights[key] = len(val)
                    val = sum(val) / len(val)
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            else:
                continue
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        if total_lengths:
            dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
            stats = torch.tensor(
                [max(total_lengths), -min(total_lengths)],
                dtype=torch.int64,
                device=device_utils.make_current_torch_device(),
            )
            dist.all_reduce(stats, op=dist.ReduceOp.MAX, group=dp_group)
            log_dict["total_lengths/max"] = int(stats[0].item())
            log_dict["total_lengths/min"] = -int(stats[1].item())

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict, metric_weights)
        if args.ci_test and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # TODO: figure out why there is a small numerical difference in log_probs and ref_log_probs in CI test, and whether it's expected or not.
                # assert reduced_log_dict["rollout/log_probs"] == reduced_log_dict["rollout/ref_log_probs"]
                assert abs(reduced_log_dict["rollout/log_probs"] - reduced_log_dict["rollout/ref_log_probs"]) < 1e-8
            if "rollout/log_probs" in reduced_log_dict:
                assert -0.5 < reduced_log_dict["rollout/log_probs"] < 0
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.5

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """Log multi-turn auxiliary metrics such as raw/observed response lengths
    and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when
    available to compute statistics without host transfers.
    """
    if mpu.get_tensor_model_parallel_rank() == 0 and mpu.is_pipeline_last_stage():
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_perf_data_fwd(args, rollout_id):
    from megatron.core import mpu

    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()
    is_primary_rank = (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}
    step = compute_rollout_step(args, rollout_id)
    log_dict["actor_fwd/step"] = step
    tracking_utils.log(args, log_dict, step_key="actor_fwd/step")


def log_perf_data(rollout_id: int, args: Namespace, flops_counter: FlopsCounter | None = None) -> None:
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.is_pipeline_last_stage()
            and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        ),
        flops_counter=flops_counter,
        world_size=dist.get_world_size(),
    )


def sync_actor_critic_data(
    args: Namespace,
    rollout_data: RolloutBatch | None = None,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Broadcast `values` (from critic) and optionally
    `log_probs`/`ref_log_probs` (from actor) across PP ranks to align data
    dependencies.

    - Values are broadcast from src=1.
    - Log-probs and ref-log-probs are broadcast from src=0 when KL is used.
    Updates `rollout_data` in place with the synchronized tensors.
    """
    log_probs_key = "log_probs" if not args.use_rollout_logprobs else "rollout_log_probs"
    values, log_probs, ref_log_probs = map(rollout_data.get, ("values", log_probs_key, "ref_log_probs"))

    # return when not the pp last stage
    if not values and not log_probs:
        return

    handles = []

    if not values:
        values = [torch.empty_like(log_prob) for log_prob in log_probs]
    for value in values:
        handles.append(dist.broadcast(value, src=1, group=group, async_op=True))

    if args.kl_coef != 0 or args.use_kl_loss:
        if not log_probs:
            log_probs = [torch.empty_like(value) for value in values]
        if not ref_log_probs:
            ref_log_probs = [torch.empty_like(value) for value in values]
        for ref_log_prob, log_prob in zip(ref_log_probs, log_probs, strict=False):
            handles.append(dist.broadcast(log_prob, src=0, group=group, async_op=True))
            handles.append(dist.broadcast(ref_log_prob, src=0, group=group, async_op=True))

    for handle in handles:
        handle.wait()

    rollout_data.update(
        {
            k: v
            for k, v in {
                "values": values,
                log_probs_key: log_probs,
                "ref_log_probs": ref_log_probs,
            }.items()
            if v is not None
        }
    )


def _map_packed_seq_params(data: PackedSeqParams, map_value: Callable[[Any], Any]) -> PackedSeqParams:
    mapped = copy(data)
    for name, value in vars(data).items():
        setattr(mapped, name, map_value(value))
    return mapped


def move_tensors_to_device(data: Any, device: torch.device, *, non_blocking: bool = False) -> Any:
    """Recursively move tensors in a (nested) dict/list to the specified
    device.

    Non-tensor values are left unchanged.
    """
    if isinstance(data, dict):
        return {k: move_tensors_to_device(v, device, non_blocking=non_blocking) for k, v in data.items()}
    if isinstance(data, list):
        return [move_tensors_to_device(v, device, non_blocking=non_blocking) for v in data]
    if isinstance(data, tuple):
        return tuple(move_tensors_to_device(v, device, non_blocking=non_blocking) for v in data)
    if isinstance(data, PackedSeqParams):
        return _map_packed_seq_params(
            data,
            lambda value: move_tensors_to_device(value, device, non_blocking=non_blocking),
        )
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    return data  # e.g., int, str, None, etc.


def pin_tensors(data: Any) -> Any:
    """Recursively pin CPU tensors in a pre-packed micro-batch."""
    if isinstance(data, dict):
        return {k: pin_tensors(v) for k, v in data.items()}
    if isinstance(data, list):
        return [pin_tensors(v) for v in data]
    if isinstance(data, tuple):
        return tuple(pin_tensors(v) for v in data)
    if isinstance(data, PackedSeqParams):
        return _map_packed_seq_params(data, pin_tensors)
    if isinstance(data, torch.Tensor) and data.device.type == "cpu":
        try:
            return data.contiguous().pin_memory()
        except RuntimeError as exc:
            from relax.utils.data.micro_batch_ring import _raise_pin_memory_failure

            _raise_pin_memory_failure("pin_tensors", exc)
    return data


def record_tensors_on_stream(data: Any, stream: Any) -> None:
    """Keep asynchronously copied tensors alive on the consuming stream."""
    if isinstance(data, dict):
        for value in data.values():
            record_tensors_on_stream(value, stream)
    elif isinstance(data, (list, tuple)):
        for value in data:
            record_tensors_on_stream(value, stream)
    elif isinstance(data, PackedSeqParams):
        for value in vars(data).values():
            record_tensors_on_stream(value, stream)
    elif isinstance(data, torch.Tensor):
        data.record_stream(stream)
