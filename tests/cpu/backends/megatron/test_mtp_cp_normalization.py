# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU/Gloo regressions for native and chunked MTP normalization.

The roll adapter exchanges CPU tensors and removes known packed boundaries;
production loss, autograd, count collectives and tracker accumulation are used.
"""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


pytest.importorskip("megatron.core")


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, 3))

    def forward(self, hidden, weight=None, runtime_gather_output=None):
        return hidden @ (self.weight if weight is None else weight).T, None


def _roll(tensors, *, cp_group=None, **kwargs):
    results = []
    for value in tensors:
        if cp_group is not None:
            parts = [torch.empty_like(value) for _ in range(cp_group.size())]
            dist.all_gather(parts, value, group=cp_group)
            value = torch.cat(parts, dim=-1)
        value = value.roll(-1, dims=-1)
        # Two independent packed sequences, each four tokens long.
        value[..., 3::4] = 0
        if cp_group is not None:
            value = value.chunk(cp_group.size(), dim=-1)[dist.get_rank(cp_group)].contiguous()
        results.append(value)
    return results


def _exercise(cp_group, *, chunked, mode, mask, teacher_mask):
    from megatron.core import parallel_state as mpu
    from megatron.core.transformer import multi_token_prediction as native

    from relax.models.common import chunked_mtp_loss as chunked_loss

    torch.manual_seed(23)
    head = _Head()
    depths = 1 if mode == "main_update_tokens" else 2
    full_hidden = torch.randn(1 + depths, 8, 1, 3)
    full_labels = torch.arange(8).remainder(5).view(1, -1)
    local = slice(None) if cp_group is None else slice(4 * dist.get_rank(cp_group), 4 * (dist.get_rank(cp_group) + 1))
    hidden = full_hidden[:, local].flatten(0, 1).clone().requires_grad_()
    labels = full_labels[:, local].contiguous()
    loss_mask = mask[:, local].contiguous()
    config = SimpleNamespace(
        mtp_num_layers=depths,
        mtp_loss_scaling_factor=0.3,
        calculate_per_token_loss=mode != "microbatch",
        cross_entropy_loss_fusion=False,
        cross_entropy_fusion_impl="native",
        mtp_detach_lm_head=False,
    )
    native.roll_tensor = chunked_loss.roll_tensor = _roll
    chunked_loss._ROLL_TENSOR_PARAMETERS = {"tensors"}
    chunked_loss._chunked_mtp_enabled = lambda: chunked
    chunked_loss._main_update_token_normalization_enabled = lambda: mode == "main_update_tokens"
    chunked_loss._resolve_chunk_size = lambda: 3
    mpu.get_data_parallel_group = lambda with_context_parallel=False: cp_group
    # The native tracker allocates its accumulator on current_device; keep it
    # on CPU for this Gloo test without changing the production tracker logic.
    torch.cuda.current_device = lambda: "cpu"
    native.MTPLossLoggingHelper.tracker.clear()
    native.MTPLossLoggingHelper.configure_acceptance_collection(False)
    native.MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))

    def cross_entropy(target, logits):
        return (
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.T.flatten(), reduction="none")
            .view_as(target.T)
            .T
        )

    function = chunked_loss.process_mtp_loss if chunked else native.process_mtp_loss
    result = function(
        hidden,
        labels,
        loss_mask,
        head,
        None,
        False,
        True,
        cross_entropy,
        config,
        cp_group=cp_group,
        mtp_input_mask=None if teacher_mask is None else teacher_mask[:, local].contiguous(),
    )
    # Backward through the genuine MTP auxiliary-loss autograd injection.
    result.sum().backward()
    gradient = head.weight.grad.clone()
    hidden_gradient = hidden.grad.view(1 + depths, 4 if cp_group is not None else 8, 1, 3)[1:]
    original_tokens = mask.sum().clamp(min=1)
    if cp_group is not None:
        dist.all_reduce(gradient, group=cp_group)
    divisor = original_tokens if mode != "microbatch" else (cp_group.size() if cp_group is not None else 1)
    gradient /= divisor
    hidden_gradient = hidden_gradient / divisor
    tracker = native.MTPLossLoggingHelper.tracker
    losses = tracker["loss_sums"].clone()
    counts = tracker.get("num_tokens")
    if cp_group is not None:
        dist.all_reduce(losses, group=cp_group)
    if counts is None:
        if cp_group is not None:
            losses /= cp_group.size()
    else:
        counts = counts.clone()
        if cp_group is not None:
            dist.all_reduce(counts, group=cp_group)
        losses /= counts.clamp(min=1)
    return gradient, hidden_gradient, losses


def _worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    group = dist.group.WORLD
    try:
        masks = [
            torch.tensor([[1, 1, 1, 1, 0, 1, 0, 1]], dtype=torch.float32),
            torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32),
            torch.zeros(1, 8),
        ]
        teacher_masks = [None, torch.tensor([[1, 0, 1, 1, 1, 1, 0, 1]], dtype=torch.bool)]
        for chunked in (False, True):
            modes = ("microbatch", "per_token", "main_update_tokens") if chunked else ("microbatch", "per_token")
            for mode in modes:
                for mask in masks:
                    for teacher_mask in teacher_masks:
                        expected, expected_hidden, expected_loss = _exercise(
                            None, chunked=chunked, mode=mode, mask=mask, teacher_mask=teacher_mask
                        )
                        actual, actual_hidden, actual_loss = _exercise(
                            group, chunked=chunked, mode=mode, mask=mask, teacher_mask=teacher_mask
                        )
                        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
                        torch.testing.assert_close(
                            actual_hidden, expected_hidden[:, 4 * rank : 4 * (rank + 1)], rtol=2e-6, atol=2e-7
                        )
                        torch.testing.assert_close(actual_loss, expected_loss, rtol=2e-6, atol=2e-7)
    finally:
        dist.destroy_process_group()


def test_native_and_chunked_cp2_match_unsharded_loss_and_gradients(tmp_path, monkeypatch):
    monkeypatch.setenv("RELAX_SFT_SELECTIVE_LOGITS", "0")
    monkeypatch.setenv("RELAX_SFT_MTP_ACCEPTANCE_METRICS", "0")
    mp.spawn(_worker, args=(str(tmp_path / "gloo"),), nprocs=2, join=True)


def test_chunked_cp_requires_the_native_normalization_patch(monkeypatch):
    from megatron.core.transformer import multi_token_prediction as native

    from relax.models.common import chunked_mtp_loss

    monkeypatch.delattr(native, "get_mtp_normalization_counts", raising=False)
    with pytest.raises(RuntimeError, match="mtp-cp-normalization.patch"):
        chunked_mtp_loss._normalization_counts(torch.tensor(1), torch.tensor(2), SimpleNamespace(size=lambda: 2))
    counts = chunked_mtp_loss._normalization_counts(torch.tensor(1), torch.tensor(2), None)
    assert tuple(int(value) for value in counts) == (1, 2, 1)
