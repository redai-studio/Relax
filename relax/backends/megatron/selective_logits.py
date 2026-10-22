# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU-owned selection plans for chunked SFT heads (not vocabulary pruning)."""

from argparse import Namespace

import torch


def prepare_selective_logits(batch: dict, args: Namespace) -> None:
    """Attach per-microbatch indices before pinning/copying the CPU batch.

    The caller restricts this to CP=1, THD text batches. Indices travel with
    the batch/packed metadata, including asynchronous copy stream lifetime. No
    CUDA nonzero, scalar reads, or cross-microbatch mutable state are used.
    """
    masks = batch["loss_masks"]
    full_mask = batch["full_loss_masks"]
    if full_mask.device.type != "cpu" or any(mask.device.type != "cpu" for mask in masks):
        raise ValueError("Selective logits plans must be prepared on CPU")
    if full_mask.ndim != 2 or full_mask.shape[0] != 1:
        raise ValueError("Selective logits plans require a single packed row")
    batch["selective_logit_rows"] = [
        (mask.numel(), torch.nonzero(mask.reshape(-1) != 0, as_tuple=True)[0]) for mask in masks
    ]
    if not getattr(args, "enable_mtp_training", False):
        return

    # Use exactly MCore's packed roll, on CPU with no process group. The model
    # supplies next-token labels and the unrolled full mask to process_mtp_loss.
    from megatron.core.transformer.multi_token_prediction import roll_tensor

    packed = batch["packed_seq_params"]
    rows = []
    for _ in range(args.mtp_num_layers):
        [full_mask] = roll_tensor([full_mask], shifts=-1, dims=-1, cp_group=None, packed_seq_params=packed)
        rows.append(torch.nonzero(full_mask[0] != 0, as_tuple=True)[0])
    packed.relax_sft_selective_mtp_rows = tuple(rows)
    packed.relax_sft_selective_full_rows = full_mask.shape[1]


def prepare_selective_logits_from_cpu_masks(batch: dict, args: Namespace) -> None:
    """Build the same plan for normal PP training without reading GPU
    tensors."""
    from copy import copy

    import torch.nn.functional as F

    from relax.utils.sft_utils import align_loss_mask_for_sft

    packed = batch["packed_seq_params"]
    cpu_packed = copy(packed)
    cpu_packed.cu_seqlens_q = torch.tensor(packed.cu_seqlens_q_cpu, dtype=torch.int32)
    physical = packed.qsa_select_offsets_cpu
    cpu_packed.cu_seqlens_q_padded = torch.tensor(physical, dtype=torch.int32)
    masks, full_masks = [], []
    for i, (mask, total, response) in enumerate(
        zip(batch["selective_loss_masks_cpu"], batch["total_lengths"], batch["response_lengths"], strict=True)
    ):
        if mask.device.type != "cpu":
            raise ValueError("Selective logits masks must stay on CPU")
        if batch.get("unconcat_pre_shifted_labels") is None and total == response:
            mask = align_loss_mask_for_sft(mask)
        masks.append(mask)
        if total != response:
            mask = F.pad(mask, (total - response - 1, 1))
        # allgather CP1 concatenates logical lengths and extends the last
        # physical segment; regular THD may append a separate padding segment.
        width = total if args.allgather_cp else physical[i + 1] - physical[i]
        full_masks.append(F.pad(mask, (0, width - total)))
    full_mask = torch.cat(full_masks)
    full_mask = F.pad(full_mask, (0, batch["tokens"].shape[1] - full_mask.numel()))[None]
    cpu_batch = {"loss_masks": masks, "full_loss_masks": full_mask, "packed_seq_params": cpu_packed}
    prepare_selective_logits(cpu_batch, args)
    batch["selective_logit_rows"] = cpu_batch["selective_logit_rows"]
    if getattr(args, "enable_mtp_training", False):
        packed.relax_sft_selective_mtp_rows = cpu_packed.relax_sft_selective_mtp_rows
        packed.relax_sft_selective_full_rows = cpu_packed.relax_sft_selective_full_rows
