# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""SFT-specific paths in megatron data.py."""

import torch

from relax.utils.sft_utils import align_loss_mask_for_sft as _align_loss_mask_for_sft


def test_align_loss_mask_for_sft_left_shift_one():
    """SFT loss_mask must be shifted left by 1 (next-token alignment) and
    right-padded with 0."""
    full_mask = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.int)

    aligned = _align_loss_mask_for_sft(full_mask)

    # Position i predicts token i+1; we learn position i if token i+1 is assistant.
    expected = torch.tensor([0, 1, 1, 1, 0, 0], dtype=torch.int)
    assert torch.equal(aligned, expected)
    assert aligned.shape == full_mask.shape


def test_batch_loss_masks_aligned_for_sft_consumers():
    """batch["loss_masks"] (consumed by loss.py:sum_of_token) must be the SFT-
    aligned per-sample list, not the raw input. Regression test for the SFT
    off-by-one loss bug fixed in data.py: previously only
    batch["full_loss_masks"] received the _align_loss_mask_for_sft shift, while
    batch["loss_masks"] kept the unaligned raw masks — causing sum_of_token to
    pair the F.pad(...,(0,1),0) dummy last position with a mask=1 entry and
    miss the first real response prediction.

    This test reproduces the relevant slice of get_batch's loss-mask block in-
    process (no Megatron dist init required), exercising both SFT and RL
    samples to make sure the RL per-sample list stays response-only.
    """
    from relax.utils.sft_utils import align_loss_mask_for_sft as _align_loss_mask_for_sft

    # SFT: response_length == total_length; mask spans the whole seq, 1 at
    # response positions. RL: response_length < total_length; mask is
    # response-only.
    sft_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int)  # len=7 SFT
    rl_mask = torch.tensor([1, 1, 1, 0], dtype=torch.int)  # len=4 (response-only), total=10
    raw_loss_masks = [sft_mask.clone(), rl_mask.clone()]
    total_lengths = [7, 10]
    response_lengths = [7, 4]

    per_sample_loss_masks: list[torch.Tensor] = []
    for loss_mask, total_length, response_length in zip(raw_loss_masks, total_lengths, response_lengths, strict=True):
        if response_length == total_length:
            loss_mask = _align_loss_mask_for_sft(loss_mask)
            per_sample_loss_masks.append(loss_mask)
        else:
            per_sample_loss_masks.append(loss_mask)

    # SFT sample: aligned (left-shifted, last=0).
    assert torch.equal(
        per_sample_loss_masks[0],
        torch.tensor([0, 0, 1, 1, 1, 1, 0], dtype=torch.int),
    )
    # RL sample: untouched response-only mask.
    assert torch.equal(per_sample_loss_masks[1], rl_mask)


def test_sft_loss_mask_alignment_with_sum_of_token_semantics():
    """Verify the aligned mask + get_responses-style indexing gives the correct
    next-token loss positions and avoids the dummy-last-position contribution.

    SFT branch in loss.py:102-104:
        logits_chunk = logits[start:end]                       # full seq
        tokens_chunk = F.pad(tokens[1:], (0,1), value=0)       # last is dummy 0
        # log_probs[i] = log P(tokens_chunk[i] | logits[i])
        #             = log P(tokens[i+1] | logits[i])   for i < S-1
        #             = log P(0          | logits[S-1])  for i = S-1  (FAKE)

    Then sum_of_token does Σ log_probs[i] * mask[i]. With the raw (unaligned)
    mask, mask[S-1]=1 lights up the fake position. With the aligned mask,
    mask[S-1]=0 always, and mask[i]=raw[i+1] lines up exactly with the real
    next-token targets.
    """
    from relax.utils.sft_utils import align_loss_mask_for_sft as _align_loss_mask_for_sft

    # response at original positions 3..6 (4 tokens including the last).
    raw_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.int)
    aligned = _align_loss_mask_for_sft(raw_mask)
    S = raw_mask.size(0)

    # Position-by-position: which i in [0, S-1] does each mask light up?
    raw_positions = (raw_mask == 1).nonzero(as_tuple=True)[0].tolist()
    aligned_positions = (aligned == 1).nonzero(as_tuple=True)[0].tolist()

    assert raw_positions == [3, 4, 5, 6], "raw mask is at response positions"
    assert aligned_positions == [2, 3, 4, 5], (
        "aligned mask is shifted left by 1 — these are the logit positions whose next-token target is a response token"
    )
    assert aligned[S - 1].item() == 0, (
        "aligned mask must zero out the dummy last position; otherwise "
        "sum_of_token pairs it with the F.pad(...,(0,1),0) sentinel token"
    )
    # Total count preserved (one position shifts in from the start, one
    # shifts out at the end), so num_tokens reporting is unaffected.
    assert int(raw_mask.sum().item()) == int(aligned.sum().item())
