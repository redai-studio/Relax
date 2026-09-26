# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F


try:
    from megatron.core import mpu
except ModuleNotFoundError as exc:
    if exc.name not in {"megatron", "megatron.core"}:
        raise
    mpu = None


def maybe_padded_total_lengths(
    total_lengths: list[int],
    qkv_format: str,
    is_vl_model: bool,
) -> list[int] | None:
    """Per-sample tp*cp*2 padded lengths for the bridge VL+CP+thd path.

    Bridge's `preprocess_packed_seqs` (Qwen3-VL et al.) pads each sample to a
    multiple of `tp*cp*2` before zigzag-splitting along CP, so the local logits
    returned by the bridge index per-sample chunks at `padded_len // (2*cp)`.
    Relax helpers that re-derive those chunks must agree.

    Returns None for non-VL/non-CP/non-thd paths so callers fall back to the
    standard `ceil(total_length / (2*cp))` formula.
    """
    cp_size = mpu.get_context_parallel_world_size()
    if not (is_vl_model and cp_size > 1 and qkv_format == "thd"):
        return None
    tp_size = mpu.get_tensor_model_parallel_world_size()
    align = tp_size * cp_size * 2
    return [(t + align - 1) // align * align for t in total_lengths]


def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    padded_total_length: int | None = None,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
):
    """All offsets start from the begining of the prompt."""
    cp_rank = dynamic_cp_rank if dynamic_cp_rank is not None else mpu.get_context_parallel_rank()
    cp_size = dynamic_cp_size if dynamic_cp_size is not None else mpu.get_context_parallel_world_size()
    assert cp_size > 1

    prompt_length = total_length - response_length
    if padded_total_length is not None:
        # Bridge VL+CP+thd: per-sample padded length is already aligned to tp*cp*2.
        assert padded_total_length % (2 * cp_size) == 0, (
            f"padded_total_length={padded_total_length} not divisible by 2*cp={2 * cp_size}"
        )
        chunk_size = padded_total_length // (2 * cp_size)
    elif qkv_format == "thd":
        chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    else:
        assert max_seq_len is not None, "max_seq_len must be provided for qkv_format=bshd"
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # the offset of 2 chunks
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)

    # the offset of 2 logits, note that the logits need a "-1".
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))

    # when the sequence is empty, make an empty slice to continue the gradient flow.
    if logits_0[0] < logits_0[1]:
        token_0 = (logits_0[0] + 1, logits_0[1] + 1)
    else:
        logits_0 = (0, 0)
        token_0 = (0, 0)

    if logits_1[0] < logits_1[1]:
        token_1 = (logits_1[0] + 1, logits_1[1] + 1)
    else:
        logits_1 = (0, 0)
        token_1 = (0, 0)

    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)


def _slice_loss_mask_with_cp(
    loss_mask: torch.Tensor,
    total_length: int,
    response_length: int,
    qkv_format: str,
    max_seq_len: int | None,
    padded_total_length: int | None,
    dynamic_cp_size: int | None,
    dynamic_cp_rank: int | None,
) -> torch.Tensor:
    """Slice a response mask in the coordinate system used by CP log-probs.

    RL masks are target-token aligned, while SFT masks have already been
    shifted to predictor coordinates by ``align_loss_mask_for_sft``. CP log-
    probs are ordered by target token in both the zigzag and allgather-
    redistributed paths, so SFT must slice its mask with the matching
    predictor/logit offsets.
    """
    prompt_length = total_length - response_length
    _, _, logits_offsets, token_offsets = get_logits_and_tokens_offset_with_cp(
        total_length,
        response_length,
        qkv_format,
        max_seq_len,
        padded_total_length,
        dynamic_cp_size=dynamic_cp_size,
        dynamic_cp_rank=dynamic_cp_rank,
    )
    if response_length == total_length:
        mask_offsets = logits_offsets
    else:
        mask_offsets = tuple((start - prompt_length, end - prompt_length) for start, end in token_offsets)
    return torch.cat([loss_mask[start:end] for start, end in mask_offsets], dim=0)


def get_sum_of_sample_mean(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
    padded_total_lengths: list[int] | None = None,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
    sample_denoms: list[int] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Calculate correct sample mean for CP."""
    if not calculate_per_token_loss:
        if sample_denoms is None:
            sample_denoms = torch.stack([loss_mask.sum() for loss_mask in loss_masks])
        else:
            sample_denoms = torch.as_tensor(sample_denoms, device=loss_masks[0].device)
    cp_size = dynamic_cp_size if dynamic_cp_size is not None else mpu.get_context_parallel_world_size()
    if cp_size == 1:

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum() / torch.clamp_min(denom, 1)
                    for x_i, loss_mask_i, denom in zip(
                        x.split(response_lengths, dim=0), loss_masks, sample_denoms, strict=False
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * loss_mask_i).sum()
                    for x_i, loss_mask_i in zip(x.split(response_lengths, dim=0), loss_masks, strict=False)
                ]
            )

    else:
        cp_chunk_lengths: list[int] = []
        chunked_loss_masks: list[torch.Tensor] = []

        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            padded_total_length = padded_total_lengths[i] if padded_total_lengths is not None else None
            chunked_loss_mask = _slice_loss_mask_with_cp(
                loss_mask,
                total_length,
                response_length,
                qkv_format,
                max_seq_len,
                padded_total_length,
                dynamic_cp_size,
                dynamic_cp_rank,
            )
            chunked_loss_masks.append(chunked_loss_mask)
            cp_chunk_lengths.append(chunked_loss_masks[i].size(0))

        def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum() / torch.clamp_min(denom, 1)
                    for x_i, chunked_loss_mask, denom in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, sample_denoms, strict=False
                    )
                ]
            )

        def sum_of_token(x: torch.Tensor) -> torch.Tensor:
            return sum(
                [
                    (x_i * chunked_loss_mask).sum()
                    for x_i, chunked_loss_mask in zip(
                        x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, strict=False
                    )
                ]
            )

    return sum_of_sample_mean if not calculate_per_token_loss else sum_of_token


def get_cp_local_mask_sums(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
    padded_total_lengths: list[int] | None = None,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
) -> torch.Tensor:
    """Return each row's loss-contributing token count on this CP rank."""
    cp_size = dynamic_cp_size if dynamic_cp_size is not None else mpu.get_context_parallel_world_size()
    if cp_size == 1:
        return torch.stack([loss_mask.sum() for loss_mask in loss_masks])

    local_mask_sums: list[torch.Tensor] = []
    for i, (total_length, response_length, loss_mask) in enumerate(
        zip(total_lengths, response_lengths, loss_masks, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        padded_total_length = padded_total_lengths[i] if padded_total_lengths is not None else None
        chunked_loss_mask = _slice_loss_mask_with_cp(
            loss_mask,
            total_length,
            response_length,
            qkv_format,
            max_seq_len,
            padded_total_length,
            dynamic_cp_size,
            dynamic_cp_rank,
        )
        local_mask_sums.append(chunked_loss_mask.sum())

    return torch.stack(local_mask_sums)


def get_cp_local_num_tokens(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
    padded_total_lengths: list[int] | None = None,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
) -> torch.Tensor:
    """Count loss-contributing tokens held by THIS CP rank.

    With context parallelism each sample is zig-zag partitioned across CP ranks,
    so a rank only owns part (or none) of every sample. Summing this CP-local
    count across the CP group (as ``finalize_model_grads`` and the metric
    all-reduce do) counts every token exactly once, independent of how tokens are
    split across ranks. This keeps the per-token normalizer correct even when the
    CP degree differs between micro-batches (dynamic CP).

    Contrast with the full-sample count (``sum(loss_mask.sum())`` on every rank),
    which is summed ``cp_size`` times and therefore requires a matching
    ``* cp_size`` on the loss/metric — a coupling that only cancels when CP is
    uniform across the step.

    For ``cp_size == 1`` this reduces to the total number of unmasked tokens
    (preserving the historical per-sample ``clamp_min(., 1)``).
    """
    local_mask_sums = get_cp_local_mask_sums(
        total_lengths,
        response_lengths,
        loss_masks,
        qkv_format,
        max_seq_lens,
        padded_total_lengths,
        dynamic_cp_size,
        dynamic_cp_rank,
    )
    cp_size = dynamic_cp_size if dynamic_cp_size is not None else mpu.get_context_parallel_world_size()
    if cp_size == 1:
        local_mask_sums = torch.clamp_min(local_mask_sums, 1)
    return local_mask_sums.sum().to(torch.int)


def all_gather_with_cp(
    tensor: torch.Tensor,
    total_length: int,
    response_length: int,
    padded_total_length: int | None = None,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
    dynamic_cp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Gather tensors across all ranks in the context parallel group.

    The first dimension of the output tensor will be the `response_length`.
    """
    cp_group = dynamic_cp_group if dynamic_cp_group is not None else mpu.get_context_parallel_group()
    cp_size = dynamic_cp_size if dynamic_cp_size is not None else mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return tensor

    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length,
        response_length,
        qkv_format,
        max_seq_len,
        padded_total_length=padded_total_length,
        dynamic_cp_size=dynamic_cp_size,
        dynamic_cp_rank=dynamic_cp_rank,
    )

    prompt_length = total_length - response_length

    chunk_0 = tensor[: logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0] :]
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    def zero(len: int) -> torch.Tensor:
        return torch.zeros(
            [len] + list(tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )

    # logprob should be within the range of [prompt_length - 1, total_length - 1]
    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        # all empty
        full_tensor = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        # only first chunk
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        full_tensor = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        # only second chunk
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)

    assert full_tensor.shape[0] == response_length, f"Expected {response_length}, got {full_tensor.shape}"
    full_tensor = dist.nn.all_reduce(full_tensor, group=cp_group)
    return full_tensor


def slice_with_cp(
    tokens: torch.Tensor,
    pad_value: tuple[int, float, Callable],
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
) -> torch.Tensor:
    cp_rank = dynamic_cp_rank if dynamic_cp_rank is not None else mpu.get_context_parallel_rank()
    cp_size = dynamic_cp_size if dynamic_cp_size is not None else mpu.get_context_parallel_world_size()

    if qkv_format == "bshd":
        assert max_seq_len is not None

    def pad_tokens(tokens, pad):
        if isinstance(pad_value, Callable):
            pad_func = pad_value
            tokens = pad_func(tokens, pad)
        else:
            # pad on the first dimension
            pad_tuple = (0, 0) * (tokens.dim() - 1) + (0, pad)
            tokens = F.pad(tokens, pad_tuple, value=pad_value)
        return tokens

    if cp_size == 1:
        if qkv_format == "bshd":
            pad = max_seq_len - tokens.size(0)
            tokens = pad_tokens(tokens, pad)
        return tokens

    token_len = len(tokens)
    if qkv_format == "thd":
        chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    else:
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # pad
    pad = 2 * cp_size * chunk_size - token_len
    tokens = pad_tokens(tokens, pad)

    # get 2 chunk for thd cp
    start_1, end_1 = chunk_size * cp_rank, chunk_size * (cp_rank + 1)
    start_2, end_2 = chunk_size * (2 * cp_size - cp_rank - 1), chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])


def slice_log_prob_with_cp(
    log_prob: list[float] | torch.Tensor,
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_token_len: int | None = None,
    padded_total_length: int | None = None,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
) -> list[float] | torch.Tensor:
    assert len(log_prob) == response_length, (
        f"log_prob length mismatch: len(log_prob)={len(log_prob)}, "
        f"response_length={response_length}, total_length={total_length}"
    )

    cp_size = dynamic_cp_size if dynamic_cp_size is not None else mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return log_prob

    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
        total_length,
        response_length,
        qkv_format,
        max_token_len,
        padded_total_length,
        dynamic_cp_size=dynamic_cp_size,
        dynamic_cp_rank=dynamic_cp_rank,
    )

    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) : logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) : logits_offset[1][1] - (prompt_length - 1)]

    if isinstance(log_prob, list):
        return chunk_1 + chunk_2
    else:
        return torch.cat([chunk_1, chunk_2], dim=0)


def compute_dynamic_cp_size(max_seq_len: int, max_tokens_per_gpu: int) -> int:
    """Compute dynamic CP group size for a micro-batch based on its max
    sequence length.

    Rounds up to the nearest power of 2. No clamp is needed: arguments.py
    derives the static context_parallel_size from rollout_max_context_len (the
    longest possible sequence), so any real micro-batch's result is already <=
    that maximum.
    """
    dynamic_cp_size = (max_seq_len + max_tokens_per_gpu - 1) // max_tokens_per_gpu
    dynamic_cp_size = 1 << (dynamic_cp_size - 1).bit_length()
    return dynamic_cp_size


def dynamic_cp_split_data(batch: dict, max_tokens_per_gpu: int) -> int:
    """Per-micro-batch dynamic CP split, run inside get_batch.

    On entry every rank of the static (size = context_parallel_size) CP group holds
    the SAME micro-batch (data enters replicated per group). This:

    1. picks a CP size from THIS mb's longest sequence (``compute_dynamic_cp_size``,
       same formula as arguments.py);
    2. if DP > 1, all-reduces MAX over the DP group so the mb uses one CP size on
       every rank (within a static CP group the value is already identical, so DP-only
       suffices; with the ascending mb sort in get_data_iterator this MAX is
       well-matched), then grows it (guided by the global-min sample count) so no
       sub-group is empty;
    3. subdivides the mb among the ``static_cp_size // dynamic_cp_size`` sub-groups so
       each trains distinct data — members of a sub-group share one seqlen-balanced
       partition (then hold its zig-zag CP shard); local & deterministic, no comm.

    Returns the per-mb ``dynamic_cp_size``; the caller derives cp_rank / cp_group from
    it. ``batch`` is mutated in place: per-sample fields are replaced by this rank's
    sub-partition.
    """
    from relax.utils import device as device_utils
    from relax.utils.data.seqlen_balancing import get_seqlen_balanced_partitions

    static_cp_size = mpu.get_context_parallel_world_size()
    static_cp_rank = mpu.get_context_parallel_rank()
    device = device_utils.make_current_torch_device()
    total_lengths = batch["total_lengths"]
    num_samples = len(total_lengths)
    max_seq = max(total_lengths)

    dynamic_cp_size = compute_dynamic_cp_size(max_seq, max_tokens_per_gpu)
    min_samples = num_samples

    if mpu.get_data_parallel_world_size(with_context_parallel=False) > 1:
        dp_group = mpu.get_data_parallel_group(with_context_parallel=False)
        cp_t = torch.tensor([dynamic_cp_size], dtype=torch.int, device=device)
        dist.all_reduce(cp_t, op=dist.ReduceOp.MAX, group=dp_group)
        dynamic_cp_size = int(cp_t.item())
        ns_t = torch.tensor([num_samples], dtype=torch.int, device=device)
        dist.all_reduce(ns_t, op=dist.ReduceOp.MIN, group=dp_group)
        min_samples = int(ns_t.item())

    # Grow so no sub-group is empty (num_sub must be <= samples); needed even at dp==1.
    while dynamic_cp_size < static_cp_size and static_cp_size // dynamic_cp_size > min_samples:
        dynamic_cp_size *= 2

    if dynamic_cp_size >= static_cp_size:
        # Whole static group is one CP group; no subdivision.
        return static_cp_size

    # dynamic_cp_size < static_cp_size (incl. 1): subdivide into
    # static_cp_size // dynamic_cp_size parts so each sub-group trains distinct data
    # (dynamic_cp_size == 1 -> pure DP, no CP slicing).
    num_sub = static_cp_size // dynamic_cp_size
    sub_idx = static_cp_rank // dynamic_cp_size
    partitions = get_seqlen_balanced_partitions(total_lengths, num_sub, equal_size=False)
    selected = partitions[sub_idx]
    for key, val in list(batch.items()):
        if isinstance(val, (list, tuple)) and len(val) == num_samples:
            batch[key] = [val[i] for i in selected]
    # Record the full-mb sample order (sub-group by sub-group) AFTER the subdivide loop,
    # so this bookkeeping list (also length num_samples) is not itself subdivided.
    # dynamic_cp_merge_output uses it to restore the original order after gathering.
    batch["_dcp_partition_order"] = [i for p in partitions for i in p]

    return dynamic_cp_size


def _nccl_all_gather_variable_tensors(
    values: list[torch.Tensor],
    world_size: int,
    group: dist.ProcessGroup,
) -> list[list[torch.Tensor]]:
    """All-gather variable-length tensors via NCCL (no pickle, no D2H/H2D).

    Each rank has a list of tensors with potentially different first-dim sizes.
    Returns a list-of-lists: ``result[rank] = [tensor, ...]`` from that rank.

    Every rank in ``group`` must call this with a non-empty ``values`` so the
    collective is symmetric and a device/dtype is available.
    """
    assert values, "_nccl_all_gather_variable_tensors requires a non-empty values list on every rank"
    local_sizes = torch.tensor([v.shape[0] for v in values], dtype=torch.long, device=values[0].device)
    num_samples = torch.tensor([len(values)], dtype=torch.long, device=values[0].device)

    all_num_samples = [torch.zeros_like(num_samples) for _ in range(world_size)]
    dist.all_gather(all_num_samples, num_samples, group=group)

    max_num_samples = max(n.item() for n in all_num_samples)
    padded_sizes = torch.zeros(max_num_samples, dtype=torch.long, device=values[0].device)
    padded_sizes[: len(values)] = local_sizes
    all_sizes = [torch.zeros_like(padded_sizes) for _ in range(world_size)]
    dist.all_gather(all_sizes, padded_sizes, group=group)

    local_cat = torch.cat(values, dim=0)
    max_total = max(s.sum().item() for s in all_sizes)
    max_total = max(max_total, 1)
    extra_dims = list(local_cat.shape[1:])
    padded = torch.zeros([max_total] + extra_dims, dtype=local_cat.dtype, device=local_cat.device)
    padded[: local_cat.shape[0]] = local_cat

    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)

    result: list[list[torch.Tensor]] = []
    for r in range(world_size):
        n = all_num_samples[r].item()
        sizes_r = all_sizes[r][:n].tolist()
        data_r = gathered[r][: sum(sizes_r)]
        result.append(list(data_r.split(sizes_r, dim=0)))
    return result


def dynamic_cp_merge_output(
    forward_data_store: list[dict],
    static_cp_size: int,
    static_cp_rank: int,
) -> list[dict]:
    """Merge forward-only outputs back to full per-mb results for dynamic CP.

    get_batch subdivided each mb inside its static (size ``static_cp_size``) CP group
    into ``static_cp_size // dynamic_cp_size`` sub-groups of ``dynamic_cp_size`` ranks
    and CP-split each sample across its sub-group, so a rank's forward output covers
    only its sub-partition, in CP-local (zig-zag) form. The forward_only write-back
    scatters by the FULL micro_batch_indices, so it needs each mb's full sample set
    back in original order. Per mb (metadata carried in ``_dcp_meta``):

    1. ``all_gather_with_cp`` over the dynamic CP sub-group -> full-response per sample;
    2. if subdivided (``dynamic_cp_size < static_cp_size``), all-gather over the static
       CP group to collect every sub-group's samples (sub-group members hold the same
       samples, so one rank per sub-group is kept), then reorder to the original mb
       order via the recorded ``partition_order``.

    Returns a new list where each mb dict holds the full mb's per-sample tensors in
    original order. Must be called by every rank of the static CP group.
    """
    merged: list[dict] = []
    for mb_result in forward_data_store:
        meta = mb_result.pop("_dcp_meta")
        dynamic_cp_size = meta["dynamic_cp_size"]
        dynamic_cp_rank = meta["dynamic_cp_rank"]
        total_lengths = meta["total_lengths"]
        response_lengths = meta["response_lengths"]
        padded_total_lengths = meta.get("padded_total_lengths")
        partition_order = meta.get("partition_order")

        new_result: dict = {}
        for key, values in mb_result.items():
            # Only per-sample tensor lists get reconstructed: same per-sample length
            # check as dynamic_cp_split_data, plus a tensor check (all_gather needs tensors).
            if not (
                isinstance(values, (list, tuple))
                and len(values) == len(total_lengths)
                and isinstance(values[0], torch.Tensor)
            ):
                new_result[key] = values
                continue

            # 1. reconstruct each sample's full response from its CP-local zig-zag shards.
            if dynamic_cp_size > 1:
                dynamic_cp_group = mpu.get_dynamic_data_context_parallel_groups(group_size=dynamic_cp_size)
                ptls = padded_total_lengths if padded_total_lengths is not None else [None] * len(values)
                values = [
                    all_gather_with_cp(
                        v,
                        tl,
                        rl,
                        padded_total_length=ptl,
                        dynamic_cp_size=dynamic_cp_size,
                        dynamic_cp_rank=dynamic_cp_rank,
                        dynamic_cp_group=dynamic_cp_group,
                    )
                    for v, tl, rl, ptl in zip(values, total_lengths, response_lengths, ptls, strict=False)
                ]

            # 2. collect all sub-groups' samples across the static CP group and reorder.
            if dynamic_cp_size < static_cp_size:
                static_cp_group = mpu.get_context_parallel_group()
                gathered = _nccl_all_gather_variable_tensors(values, static_cp_size, static_cp_group)
                # Sub-group members hold identical samples -> take one rank per sub-group.
                selected_ranks = list(range(static_cp_rank % dynamic_cp_size, static_cp_size, dynamic_cp_size))
                values = [v for r in selected_ranks for v in gathered[r]]
                # A subdivided mb always carries a partition order; reorder back to the
                # original mb sample order so the write-back aligns with micro_batch_indices.
                # Fail loud (not a silent wrong order) if the invariant ever breaks.
                assert partition_order is not None and len(partition_order) == len(values), (
                    "dynamic-CP merge: partition_order missing or length mismatch "
                    f"(order={None if partition_order is None else len(partition_order)}, values={len(values)})"
                )
                reordered: list = [None] * len(values)
                for new_pos, orig_pos in enumerate(partition_order):
                    reordered[orig_pos] = values[new_pos]
                values = reordered

            new_result[key] = values
        merged.append(new_result)
    return merged


def gdn_reassemble_full(
    gathered: list[torch.Tensor],
    cu_seqlens: torch.Tensor | list[int],
    cp_size: int,
) -> torch.Tensor:
    """Reassemble per-CP-rank zig-zag shards into the full sequential sequence.

    ``gathered[r]`` is rank ``r``'s local activation ``[s_local, b, C]``. Each
    sample is split across CP by :func:`slice_with_cp`: rank ``r`` holds chunk
    ``r`` and chunk ``2*cp-1-r`` (each of size ``chunk_size``). This reassembles,
    per sample, the sequential chunk order ``[0, 1, ..., 2*cp-1]``. Pure (no
    collectives) so it is unit-testable. Returns ``[s_full, b, C]``.

    Inverse of taking, per rank, :func:`gdn_cp_slice`.

    ``cu_seqlens`` are the full (x cp) per-sample boundaries. Pass a **Python
    ``list[int]``** on the hot path (precomputed once per micro-batch) to avoid a
    per-layer ``.tolist()`` device sync; a device tensor is also accepted (host
    sync) for standalone/unit-test use.
    """
    cu = cu_seqlens if isinstance(cu_seqlens, list) else cu_seqlens.tolist()
    local_cu = [c // cp_size for c in cu]
    pieces: list[torch.Tensor] = []
    for i in range(len(local_cu) - 1):
        lo, hi = local_cu[i], local_cu[i + 1]
        cs = (hi - lo) // 2  # per-rank chunk_size (each rank holds 2 chunks)
        first_halves = [gathered[r][lo : lo + cs] for r in range(cp_size)]  # chunks 0..cp-1
        second_halves = [gathered[r][lo + cs : hi] for r in range(cp_size)][::-1]  # chunks cp..2cp-1
        pieces.extend(first_halves + second_halves)
    return torch.cat(pieces, dim=0)


def gdn_cp_slice(
    full: torch.Tensor,
    cu_seqlens: torch.Tensor | list[int],
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Slice this CP rank's zig-zag shard out of the full sequential sequence.

    Inverse of :func:`gdn_reassemble_full` for one rank: given the full
    ``[s_full, b, X]`` (sequential order), return this rank's shard
    ``[s_local, b, X]`` = per sample ``[chunk_r, chunk_{2cp-1-r}]``, matching
    :func:`slice_with_cp`. Plain indexing + cat, so autograd scatters the grad
    back into the correct full-sequence positions.

    ``cu_seqlens`` are the full per-sample boundaries. Pass a **Python
    ``list[int]``** on the hot path (precomputed once per micro-batch) to avoid a
    per-layer ``.tolist()`` device sync; a device tensor is also accepted.
    """
    full_cu = cu_seqlens if isinstance(cu_seqlens, list) else cu_seqlens.tolist()
    pieces: list[torch.Tensor] = []
    for i in range(len(full_cu) - 1):
        flo, fhi = full_cu[i], full_cu[i + 1]
        cs = (fhi - flo) // (2 * cp_size)
        c1 = full[flo + cp_rank * cs : flo + (cp_rank + 1) * cs]
        c2 = full[flo + (2 * cp_size - cp_rank - 1) * cs : flo + (2 * cp_size - cp_rank) * cs]
        pieces.append(c1)
        pieces.append(c2)
    return torch.cat(pieces, dim=0)


class _AllGatherFullSequence(torch.autograd.Function):
    """All-gather each CP rank's shard into the full sequence; backward reduce-
    scatters (sums) the gradient.

    The GDN all-gather path is **not** a fully-duplicated computation: after the
    duplicated recurrent scan on the full sequence, each rank slices back *its
    own* zig-zag shard (:func:`gdn_cp_slice`) for the output projection, so every
    rank's downstream loss is different. Because the scan is causal/recurrent,
    rank ``r``'s output positions depend on input positions owned by *other* CP
    ranks, and — symmetrically — rank ``r``'s input positions receive gradient
    from *other* ranks' output losses. The correct grad for a full-sequence
    position is therefore the **sum** over all CP ranks of each rank's local
    backward, scattered back to the owning rank: exactly ``reduce_scatter(sum)``.

    (A plain ``grads[rank]`` backward — correct only when the post-gather compute
    is duplicated *and* the loss is identical on every rank — would silently drop
    these cross-rank contributions and corrupt training gradients.)
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        ctx.world_size = dist.get_world_size(group=group)
        out = [torch.empty_like(x) for _ in range(ctx.world_size)]
        dist.all_gather(out, x.contiguous(), group=group)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        # grads[m] = dL_local/d(gathered[m]); sum across ranks and keep this
        # rank's slot: out = sum_k grads[this_rank](on rank k) = full input grad.
        template = next((g for g in grads if g is not None), None)
        grad_list = [torch.zeros_like(template) if g is None else g.contiguous() for g in grads]
        out = torch.empty_like(grad_list[ctx.rank])
        dist.reduce_scatter(out, grad_list, group=ctx.group)
        return out, None


def gdn_cp_gather_full(
    qkvzba: torch.Tensor,
    cu_seqlens: torch.Tensor | list[int],
    cp_size: int,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Gather the per-CP-rank zig-zag shards into the full sequential sequence.

    All-gathers every rank's local activation (reduce-scatter backward autograd,
    :class:`_AllGatherFullSequence`) then reassembles with
    :func:`gdn_reassemble_full`. ``qkvzba`` is ``[s_local, b, C]``; returns
    ``[s_full, b, C]``. Pass a precomputed host boundary list as ``cu_seqlens`` on
    the hot path to avoid a per-layer device sync.
    """
    gathered = _AllGatherFullSequence.apply(qkvzba, cp_group)
    return gdn_reassemble_full(gathered, cu_seqlens, cp_size)
