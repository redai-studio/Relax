# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Selective MTP head preserves masked loss/gradients and packed-mask
alignment."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


try:
    from relax.models.common import chunked_mtp_loss as patch
except ImportError as exc:
    pytest.skip(f"Megatron unavailable: {exc}", allow_module_level=True)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(11, 5))
        self.rows = []
        self.sequence_parallel = False

    def forward(self, hidden, weight=None, runtime_gather_output=None):
        self.rows.append(hidden.shape[0])
        return F.linear(hidden, self.weight if weight is None else weight), None


def ce(labels, logits):
    return F.cross_entropy(logits[:, 0], labels[0], reduction="none").unsqueeze(0)


def test_fused_mtp_preserves_selective_mask_gradients_and_acceptance(monkeypatch):
    from relax.backends.megatron import fused_linear_ce as fused

    torch.manual_seed(14)
    hidden = torch.randn(10, 1, 5, requires_grad=True)
    head = Head()
    head.gradient_accumulation_fusion = False
    head.tp_group = object()
    labels = torch.randint(11, (1, 10))
    rows = torch.tensor([0, 1, 4, 5, 8])
    mask = torch.zeros(1, 10)
    mask[:, rows] = 0.5
    mask[:, 1] = 0  # Teacher filtering after the CPU row plan.
    calls = []

    def linear(hidden, weight, labels, group, **kwargs):
        calls.append((len(hidden), kwargs["return_logits"]))
        logits = F.linear(hidden, weight)
        losses = F.cross_entropy(logits[:, 0], labels[:, 0], reduction="none").unsqueeze(1)
        return losses, logits[:, 0].detach()

    monkeypatch.setattr(fused, "supported", lambda *_args: True)
    monkeypatch.setattr(fused, "linear_cross_entropy", linear)
    monkeypatch.setattr(
        patch,
        "_COMPUTE_MTP_ACCEPTANCE_COUNTS",
        lambda logits, labels, mask, *_args: (
            ((logits[:, 0].argmax(-1) == labels[0]) & mask[0].bool()).sum(),
            mask.bool().sum(),
        ),
    )
    loss, correct, total = patch._chunked_head_ce(
        hidden,
        labels,
        head,
        None,
        None,
        ce,
        None,
        3,
        loss_mask=mask,
        selected_rows=rows,
    )
    expected_logits = F.linear(hidden, head.weight)
    expected = ce(labels, expected_logits)
    actual_grads = torch.autograd.grad((loss * mask).sum(), (hidden, head.weight))
    expected_grads = torch.autograd.grad((expected * mask).sum(), (hidden, head.weight))
    torch.testing.assert_close(loss * mask, expected * mask)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want)
    assert correct == ((expected_logits[:, 0].argmax(-1) == labels[0]) & mask[0].bool()).sum()
    assert total == mask.bool().sum()
    assert calls == [(3, True), (2, True)] and head.rows == []


@pytest.mark.parametrize("rows", [[], [0], [1, 3, 4, 8], list(range(10))])
@pytest.mark.parametrize("chunk_size", [1, 3, 16])
@pytest.mark.parametrize("teacher_mask", [False, True])
def test_selective_mtp_masked_loss_and_gradients(rows, chunk_size, teacher_mask, monkeypatch):
    torch.manual_seed(12)
    hidden = torch.randn(10, 1, 5, requires_grad=True)
    labels = torch.randint(11, (1, 10))
    mask = torch.zeros(1, 10)
    mask[:, rows] = 0.5  # Preserve mask weights, including nonbinary weights.
    if teacher_mask:
        # The precomputed plan is a superset after teacher masking.
        mask[:, 1::2] = 0
    head = Head()
    counts = []

    def acceptance(logits, labels, mask, *_args):
        counts.append(mask.numel())
        return ((logits[:, 0].argmax(-1) == labels[0]) & mask[0].bool()).sum(), mask.bool().sum()

    monkeypatch.setattr(patch, "_COMPUTE_MTP_ACCEPTANCE_COUNTS", acceptance)
    outputs = []
    for indices in (None, torch.tensor(rows, dtype=torch.long)):
        head.rows.clear()
        loss, correct, total = patch._chunked_head_ce(
            hidden,
            labels,
            head,
            None,
            None,
            ce,
            lambda logits: logits * 0.7,
            chunk_size,
            loss_mask=mask,
            selected_rows=indices,
        )
        grads = torch.autograd.grad((loss * mask).sum(), (hidden, head.weight))
        outputs.append(((loss * mask).detach(), correct, total, grads, sum(head.rows)))
    ref, got = outputs
    for actual, expected in zip(got[:3], ref[:3]):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(got[3], ref[3]):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert got[4] == (len(rows) if rows else 10)


def test_selective_mtp_gathers_sp_before_selection(monkeypatch):
    from megatron.core.tensor_parallel import mappings

    head = Head()
    head.sequence_parallel = True
    shard = torch.randn(5, 1, 5, requires_grad=True)
    labels = torch.randint(11, (1, 10))
    selected = torch.tensor([1, 6, 9])
    group = object()
    calls = []

    def gather(hidden, *, tensor_parallel_output_grad, group):
        calls.append((hidden.shape[0], tensor_parallel_output_grad, group))
        return torch.cat((hidden, hidden), dim=0)

    monkeypatch.setattr(mappings, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(patch, "_COMPUTE_MTP_ACCEPTANCE_COUNTS", None)
    loss, _, _ = patch._chunked_head_ce(
        shard,
        labels,
        head,
        None,
        None,
        ce,
        None,
        2,
        tp_group=group,
        selected_rows=selected,
    )
    loss.sum().backward()
    assert calls == [(5, False, group)]
    assert head.rows == [2, 1]
    assert head.sequence_parallel is True
    assert shard.grad is not None
    assert loss.shape == (1, 10)
    assert torch.count_nonzero(loss).item() == 3


@pytest.mark.parametrize(
    "fallback", [None, "disabled", "eval", "teacher", "derived", "cp2", "bad_length", "bad_depth"]
)
def test_selective_mtp_process_gates_and_rolled_mask(monkeypatch, fallback):
    # Model the pinned image's acceptance API even when running on older MCore.
    monkeypatch.setattr(patch, "_COMPUTE_MTP_ACCEPTANCE_COUNTS", lambda *_args: None)
    monkeypatch.setenv("RELAX_SFT_SELECTIVE_LOGITS", "0" if fallback == "disabled" else "1")
    monkeypatch.setattr(patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(patch, "_main_update_token_normalization_enabled", lambda: False)
    monkeypatch.setattr(patch, "_resolve_chunk_size", lambda: 3)
    monkeypatch.setattr(patch, "_save_mtp_loss_to_tracker", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(patch, "MTPLossAutoScaler", SimpleNamespace(apply=lambda hidden, loss: hidden))
    mask = torch.tensor([[0, 1, 1, 0, 0, 1, 1, 0]], dtype=torch.float)
    # Two independently packed documents [0:4], [4:8]. Each depth rolls again.
    rows = (torch.tensor([0, 1, 4, 5]), torch.tensor([0, 4]))
    params = SimpleNamespace(
        qkv_format="thd",
        relax_sft_selective_mtp_rows=rows[:1] if fallback == "bad_depth" else rows,
        relax_sft_selective_full_rows=7 if fallback == "bad_length" else 8,
    )

    def roll(tensor, **_kwargs):
        out = torch.zeros_like(tensor)
        out[:, :3] = tensor[:, 1:4]
        out[:, 4:7] = tensor[:, 5:8]
        return out, out.sum()

    monkeypatch.setattr(patch, "roll_tensor", lambda tensors, **kwargs: [roll(t, **kwargs)[0] for t in tensors])
    monkeypatch.setattr(patch, "_ROLL_TENSOR_PARAMETERS", {"tensors"})
    seen = []

    def chunked(_hidden, labels, *_args, **kwargs):
        seen.append((kwargs["selected_rows"], kwargs["loss_mask"]))
        return labels.float() * 0, None, None

    monkeypatch.setattr(patch, "_chunked_head_ce", chunked)
    config = SimpleNamespace(
        mtp_num_layers=2,
        mtp_loss_scaling_factor=0.3,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=False,
        cross_entropy_fusion_impl="native",
        mtp_detach_heads=False,
        mtp_detach_lm_head=False,
    )
    labels = torch.zeros(1, 8, dtype=torch.long)
    patch.process_mtp_loss(
        torch.zeros(24, 1, 5),
        None if fallback == "derived" else labels,
        mask,
        Head(),
        None,
        None,
        fallback != "eval",
        ce,
        config,
        cp_group=SimpleNamespace(size=lambda: 2 if fallback == "cp2" else 1),
        packed_seq_params=params,
        input_ids=labels,
        mtp_input_mask=torch.tensor([[1, 1, 0, 1, 1, 0, 1, 1]]) if fallback == "teacher" else None,
    )
    assert len(seen) == 2
    for depth, (selected, rolled_mask) in enumerate(seen):
        if fallback in (None, "teacher"):
            torch.testing.assert_close(selected, rows[depth])
            if fallback == "teacher":
                expected = torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0]]) if depth == 0 else torch.zeros(1, 8)
                torch.testing.assert_close(rolled_mask, expected.to(rolled_mask.dtype))
            else:
                torch.testing.assert_close(selected, rolled_mask[0].nonzero().flatten())
        else:
            assert selected is None


@pytest.mark.parametrize("all_excluded", [False, True])
def test_teacher_mask_multi_depth_loss_gradients_and_acceptance(monkeypatch, all_excluded):
    torch.manual_seed(27)
    monkeypatch.setattr(patch, "_chunked_mtp_enabled", lambda: True)
    monkeypatch.setattr(patch, "_main_update_token_normalization_enabled", lambda: False)
    monkeypatch.setattr(patch, "_resolve_chunk_size", lambda: 3)
    mask = torch.tensor([[0, 1, 1, 1, 0, 1, 1, 1]], dtype=torch.float)
    teacher = torch.zeros_like(mask) if all_excluded else torch.tensor([[1, 1, 0, 1, 1, 1, 1, 0]])

    def roll(tensor, **_kwargs):
        out = torch.zeros_like(tensor)
        out[:, :3] = tensor[:, 1:4]
        out[:, 4:7] = tensor[:, 5:8]
        return out, out.sum()

    monkeypatch.setattr(patch, "roll_tensor", lambda tensors, **kwargs: [roll(t, **kwargs)[0] for t in tensors])
    monkeypatch.setattr(patch, "_ROLL_TENSOR_PARAMETERS", {"tensors"})
    monkeypatch.setattr(
        patch,
        "_COMPUTE_MTP_ACCEPTANCE_COUNTS",
        lambda logits, labels, mask, *_args: (
            ((logits[:, 0].argmax(-1) == labels[0]) & mask[0].bool()).sum(),
            mask.bool().sum(),
        ),
    )
    scaled = []
    tracked = []

    def scale(hidden, loss):
        scaled.append(loss)
        return hidden

    def track(loss, num_tokens, depth, config, correct, total, **_kwargs):
        tracked.append((loss.detach(), num_tokens, correct, total))

    monkeypatch.setattr(patch, "MTPLossAutoScaler", SimpleNamespace(apply=scale))
    monkeypatch.setattr(patch, "_save_mtp_loss_to_tracker", track)
    params = SimpleNamespace(
        qkv_format="thd",
        relax_sft_selective_mtp_rows=(torch.tensor([0, 1, 2, 4, 5, 6]), torch.tensor([0, 1, 4, 5])),
        relax_sft_selective_full_rows=8,
    )
    config = SimpleNamespace(
        mtp_num_layers=2,
        mtp_loss_scaling_factor=0.3,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=False,
        cross_entropy_fusion_impl="native",
        mtp_detach_heads=False,
        mtp_detach_lm_head=False,
    )
    hidden = torch.randn(24, 1, 5, requires_grad=True)
    labels = torch.randint(11, (1, 8))
    head = Head()
    outputs = []
    for enabled in (False, True):
        monkeypatch.setenv("RELAX_SFT_SELECTIVE_LOGITS", str(int(enabled)))
        scaled.clear()
        tracked.clear()
        head.rows.clear()
        patch.process_mtp_loss(
            hidden,
            labels,
            mask,
            head,
            None,
            None,
            True,
            ce,
            config,
            cp_group=SimpleNamespace(size=lambda: 1),
            packed_seq_params=params,
            mtp_input_mask=teacher,
        )
        loss = sum(x.sum() for x in scaled)
        gradients = torch.autograd.grad(loss, (hidden, head.weight))
        outputs.append((loss.detach(), gradients, list(tracked), sum(head.rows)))
    dense, selected = outputs
    torch.testing.assert_close(selected[0], dense[0])
    for got, ref in zip(selected[1], dense[1]):
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
    for got_depth, ref_depth in zip(selected[2], dense[2]):
        for got, ref in zip(got_depth, ref_depth):
            torch.testing.assert_close(got, ref)
    assert dense[3] == 16
    assert selected[3] == 10
