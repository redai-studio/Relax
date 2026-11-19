# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Opt-in selective MTP CE, fixed integer-count launch, native wider bwd
splits."""


def selected_mtp_head(
    output_layer,
    loss_mask,
    input_,
    *,
    weight=None,
    runtime_gather_output=None,
    output_cross_entropy_loss=False,
    labels=None,
    packed_seq_params=None,
):
    import torch

    assert output_cross_entropy_loss and labels is not None
    assert labels.ndim == loss_mask.ndim == 2 and labels.shape == loss_mask.shape
    assert labels.shape[0] == 1 and input_.ndim == 3 and input_.shape[1] == 1
    sp = bool(output_layer.sequence_parallel)
    if sp:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        # Native LCE sums dHidden over TP; gather backward only selects the SP shard.
        input_ = gather_from_sequence_parallel_region(
            input_, tensor_parallel_output_grad=False, group=output_layer.tp_group
        )
    assert input_.shape[0] == labels.numel()
    # Native process_mtp_loss already rolled this mask at packed sequence boundaries.
    rows = torch.nonzero(loss_mask.reshape(-1) != 0, as_tuple=False).flatten()
    empty = rows.numel() == 0
    selected = rows.new_zeros(1) if empty else rows
    hidden = input_.index_select(0, selected)
    target = labels.reshape(-1).index_select(0, selected).reshape(1, -1)
    try:
        output_layer.sequence_parallel = False
        # Preserve __call__ and therefore DDP's parameter-all-gather prehook.
        losses = output_layer(
            hidden,
            weight=weight,
            runtime_gather_output=runtime_gather_output,
            output_cross_entropy_loss=True,
            labels=target,
        )
    finally:
        output_layer.sequence_parallel = sp
    assert losses.shape == target.shape
    if empty:
        return losses.new_zeros(labels.shape) + losses.sum() * 0
    return losses.new_zeros(labels.numel()).index_copy(0, rows, losses.reshape(-1)).reshape_as(labels)


def install_ce_launches():
    """Validate the statically selected launch; configuration is owned by
    MCore."""
    from megatron.core.fusions.linear_cross_entropy.blackwell import entry

    bwd = entry._get_bwd_config()
    assert not bwd._bwd_kernel and bwd._vocab_per_split == 16384
