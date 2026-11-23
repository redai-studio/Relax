# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Use one gathered supervised-row pool per microbatch for the main SFT
head."""

from functools import wraps


def install(args):
    from relax.backends.megatron import loss
    from relax.utils.env import Envs

    if args.loss_type != "sft" or args.context_parallel_size != 1 or args.qkv_format != "thd":
        raise RuntimeError("Pooled supervision requires packed CP1 SFT")
    native = loss.get_log_probs_and_entropy

    @wraps(native)
    def run(logits, **kwargs):
        head = kwargs.get("lm_head_forward")
        original = kwargs["args"]
        if (
            not Envs.RELAX_SFT_FUSED_LINEAR_CE
            or not callable(getattr(head, "_relax_fused_linear_ce", None))
            or original.allgather_cp
            or kwargs.get("cp_output_is_zigzag")
            or kwargs.get("dynamic_cp_size") not in (None, 1)
            or kwargs.get("with_entropy")
            or kwargs.get("with_topk")
            or kwargs.get("gather_topk_token_ids") is not None
            or not kwargs.get("non_loss_data", True)
            or logits.ndim != 3
            or logits.shape[0] != 1
            or not logits.is_contiguous()
        ):
            return native(logits, **kwargs)
        pooled = pooled_log_probs(logits, **kwargs)
        return native(logits, **kwargs) if pooled is None else pooled

    loss.get_log_probs_and_entropy = run


def pooled_log_probs(
    logits,
    *,
    args,
    lm_head_forward,
    unconcat_tokens,
    total_lengths,
    response_lengths,
    pre_shifted_labels=None,
    selective_logit_rows=None,
    max_seq_lens=None,
    padded_total_lengths=None,
    cp_output_is_zigzag=False,
    dynamic_cp_size=None,
    dynamic_cp_rank=None,
    with_entropy=False,
    with_topk=False,
    gather_topk_token_ids=None,
    non_loss_data=True,
    **unused,
):
    import torch

    from relax.backends.megatron import loss as loss_api
    from relax.utils.env import Envs

    assert args.loss_type == "sft" and args.sft_chunked_logits and args.qkv_format == "thd"
    assert dynamic_cp_size in (None, 1) and loss_api.mpu.get_context_parallel_world_size() == 1
    assert not args.allgather_cp and not cp_output_is_zigzag
    assert non_loss_data and not with_entropy and not with_topk and gather_topk_token_ids is None
    estimator = getattr(args, "advantage_estimator", None)
    assert estimator is None or loss_api.get_algorithm(estimator).policy_loss_fn != "score_centering"
    assert Envs.RELAX_SFT_FUSED_LINEAR_CE
    fused_ce = getattr(lm_head_forward, "_relax_fused_linear_ce", None)
    assert callable(fused_ce), "Pooled CE requires a persistent MFSDP fused head"
    flat = logits.squeeze(0)
    assert logits.ndim == 3 and logits.shape[0] == 1 and flat.is_contiguous()
    width = flat.shape[-1]
    indices, targets, layouts = [], [], []
    for sample, (hidden, labels) in enumerate(
        loss_api.get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            pre_shifted_labels=pre_shifted_labels,
            max_seq_lens=max_seq_lens,
            padded_total_lengths=padded_total_lengths,
            cp_output_is_zigzag=cp_output_is_zigzag,
            dynamic_cp_size=dynamic_cp_size,
            dynamic_cp_rank=dynamic_cp_rank,
        )
    ):
        full_rows = hidden.shape[0]
        assert hidden.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
        assert hidden.stride() == (width, 1)
        delta = hidden.storage_offset() - flat.storage_offset()
        assert delta >= 0 and delta % width == 0
        offset = delta // width
        assert offset + full_rows <= flat.shape[0]
        selection = None
        if Envs.RELAX_SFT_SELECTIVE_LOGITS and selective_logit_rows is not None:
            expected, chosen = selective_logit_rows[sample]
            assert expected == full_rows
            if 0 < chosen.numel() < full_rows:
                selection = chosen
        # Preserve the native all-zero-mask path: it still evaluates all rows.
        local = torch.arange(full_rows, device=flat.device, dtype=torch.long) if selection is None else selection
        indices.append(local + offset)
        targets.append(labels if selection is None else labels.index_select(0, selection))
        layouts.append((full_rows, selection, local.numel()))
    assert len(layouts) == len(unconcat_tokens) and layouts
    selected = torch.cat(indices)
    pooled_hidden = flat.index_select(0, selected)
    pooled_targets = torch.cat(targets)
    chunk_size = int(args.sft_logits_chunk_size)
    assert chunk_size > 0
    chunks = []
    # A single SplitBackward avoids a full-sized zero allocation per slice.
    for hidden, labels in zip(pooled_hidden.split(chunk_size), pooled_targets.split(chunk_size), strict=True):
        if hidden.shape[0] == 0:
            chunks.append(hidden.float().sum(-1))
            continue
        losses = fused_ce(hidden.unsqueeze(1), labels, args.rollout_temperature)
        if losses is None:
            # Instrumented heads and delayed weight gradients must run the original head, including its hooks.
            return None
        chunks.append(-losses.reshape(-1))
    values = torch.cat(chunks)
    outputs = []
    for value, (full_rows, selection, _) in zip(values.split([x[2] for x in layouts]), layouts, strict=True):
        outputs.append(value if selection is None else value.new_zeros(full_rows).index_copy(0, selection, value))
    return logits.new_empty(0, dtype=torch.float32), {"log_probs": outputs}
