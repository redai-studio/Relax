# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bridge's per-document zigzag output must retain next-token loss
alignment."""

from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F


try:
    from relax.backends.megatron import cp_utils
    from relax.backends.megatron import loss as loss_module
    from relax.engine.sft.eval.ppl import compute_sft_eval_step
    from relax.utils.sft_utils import align_loss_mask_for_sft
except (ImportError, AttributeError) as exc:
    pytest.skip(f"Requires the pinned Megatron runtime: {exc}", allow_module_level=True)


def _local_hidden(hidden, lengths, padded, cp_size, rank):
    if cp_size == 1:
        return hidden.unsqueeze(0)
    result = []
    for values, physical in zip(hidden.split(lengths), padded, strict=True):
        values = F.pad(values, (0, 0, 0, physical - len(values)))
        chunks = values.chunk(2 * cp_size)
        result.extend((chunks[rank], chunks[2 * cp_size - rank - 1]))
    return torch.cat(result).unsqueeze(0)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("chunked,recompute", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("dynamic", [False, True])
def test_vl_cp_sft_loss_gradients_and_eval_match_unpartitioned(monkeypatch, cp_size, chunked, recompute, dynamic):
    """Include padding, empty supervision, a one-token document and span
    ends."""
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(loss_module.mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(loss_module, "validate_opd_topk_gather", lambda *_: None)

    def fail_redistribution(*_, **__):
        raise AssertionError("Bridge output is already zigzag; no contiguous redistribution is valid")

    monkeypatch.setattr(loss_module, "_allgather_cp_redistribute", fail_redistribution)

    def log_probs(logits, tokens, *_, **__):
        return -F.cross_entropy(logits, tokens, reduction="none").unsqueeze(-1), None

    monkeypatch.setattr(loss_module, "calculate_log_probs_and_entropy", log_probs)
    torch.manual_seed(17)
    lengths = [7, 10, 1]
    padded = [((length + 4 * cp_size - 1) // (4 * cp_size)) * (4 * cp_size) for length in lengths]
    tokens = [torch.randint(0, 19, (length,)) for length in lengths]
    masks = [torch.tensor([0, 0, 1, 1, 0, 1, 1]), torch.zeros(10, dtype=torch.long), torch.zeros(1)]
    aligned = [align_loss_mask_for_sft(mask) for mask in masks]
    hidden = torch.randn(sum(lengths), 5, requires_grad=True)
    weight = torch.randn(19, 5, requires_grad=True)
    ref = sum(
        (F.cross_entropy((values[:-1] @ weight.T), labels[1:], reduction="none") * mask[1:]).sum()
        for values, labels, mask in zip(hidden.split(lengths), tokens, masks, strict=True)
    )
    expected_grads = torch.autograd.grad(ref, (hidden, weight), retain_graph=True)
    args = Namespace(
        qkv_format="thd",
        loss_type="sft",
        allgather_cp=True,
        rollout_temperature=1.0,
        sft_chunked_logits=chunked,
        recompute_loss_function=recompute,
        sft_logits_chunk_size=3,
        log_probs_chunk_size=3,
    )
    actual = hidden.sum() * 0
    eval_loss, eval_count = 0.0, 0
    for rank in range(cp_size):
        monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda r=rank: r)
        metadata = dict(
            total_lengths=lengths,
            response_lengths=lengths,
            loss_masks=aligned,
            padded_total_lengths=padded if cp_size > 1 else None,
            dynamic_cp_size=cp_size if dynamic else None,
            dynamic_cp_rank=rank if dynamic else None,
        )
        batch = dict(unconcat_tokens=tokens, cp_output_is_zigzag=cp_size > 1, **metadata)
        reducer = cp_utils.get_sum_of_sample_mean(**metadata, calculate_per_token_loss=True)
        local_hidden = _local_hidden(hidden, lengths, padded, cp_size, rank)
        local_logits = local_hidden @ weight.T
        if chunked:
            value, _ = loss_module.sft_loss_function_chunked(
                args, batch, local_hidden, reducer, lm_head_forward=lambda h: (h @ weight.T, None)
            )
        else:
            value, _ = loss_module.sft_loss_function(args, batch, local_logits, reducer)
        actual = actual + value
        _, result = compute_sft_eval_step(local_logits, args=args, **batch)
        eval_loss += result["sum_neg_log_prob"][0].item()
        eval_count += result["num_tokens"][0].item()
    if recompute:
        actual.backward()
        actual_grads = (hidden.grad, weight.grad)
    else:
        actual_grads = torch.autograd.grad(actual, (hidden, weight))
    torch.testing.assert_close(actual, ref)
    for got, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, expected)
    assert eval_loss == pytest.approx(ref.item(), rel=1e-6)
    assert eval_count == sum(mask[1:].sum().item() for mask in masks)


def test_contiguous_allgather_layout_is_preserved_without_bridge_metadata(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_rank", lambda: 1)
    args = Namespace(qkv_format="thd", loss_type="sft", allgather_cp=True, rollout_temperature=1.0)
    chunks = list(
        loss_module.get_responses(
            torch.arange(8, 16, dtype=torch.float32).reshape(1, 8, 1),
            args=args,
            unconcat_tokens=[torch.arange(8), torch.arange(8, 16)],
            total_lengths=[8, 8],
            response_lengths=[4, 4],
        )
    )
    assert chunks[0][0].numel() == 0
    assert chunks[1][0].flatten().tolist() == [11, 12, 13, 14]
    assert chunks[1][1].tolist() == [12, 13, 14, 15]
