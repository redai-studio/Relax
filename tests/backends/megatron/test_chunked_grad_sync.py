# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU asynchronous-buffer regression for chunked-head bucket finalization."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


_PATH = Path(__file__).resolve().parents[3] / "relax/backends/megatron/chunked_grad_sync.py"
_SPEC = importlib.util.spec_from_file_location("chunked_grad_sync", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
defer_head_bucket_grad_sync = _MODULE.defer_head_bucket_grad_sync


class _Bucket:
    """Snapshot at dispatch, then overwrite at wait like an async
    collective."""

    def __init__(self, shape, config=None):
        self.ddp_config = config or SimpleNamespace(overlap_grad_reduce=True)
        self.grad = torch.zeros(shape)
        self.grad_reduce_handle = None
        self.grad_reduce_finished = False
        self.is_first_batch = True
        self.param_gather_dispatched = True
        self.snapshots = []
        self.waits = 0
        self.copybacks = 0
        self.force_choices = []

    def start_grad_sync(self, force_all_reduce=False):
        assert self.grad_reduce_handle is None
        self.force_choices.append(force_all_reduce)
        snapshot = self.grad.clone()
        self.snapshots.append(snapshot)

        def wait():
            self.waits += 1
            self.grad.copy_(2 * snapshot)  # Two identical DP ranks, SUM.

        self.grad_reduce_handle = SimpleNamespace(wait=wait)

    def finish_grad_sync(self, force_all_reduce=False):
        self.param_gather_dispatched = False
        if self.grad_reduce_finished:
            return
        if self.is_first_batch:
            self.start_grad_sync(force_all_reduce=force_all_reduce)
        assert self.grad_reduce_handle is not None
        self.grad_reduce_handle.wait()
        self.grad_reduce_handle = None
        self.copybacks += 1
        self.grad_reduce_finished = True

    def reset(self):
        self.grad.zero_()
        self.grad_reduce_finished = False


@pytest.mark.parametrize("force_all_reduce", [False, True])
@pytest.mark.parametrize("mtp_depth", [0, 1, 2])
def test_chunked_head_variable_counts_accumulate_before_dispatch(force_all_reduce, mtp_depth):
    torch.manual_seed(830)
    weight = torch.nn.Parameter(torch.randn(11, 5))
    bucket = _Bucket(weight.shape)
    defer_head_bucket_grad_sync(bucket)
    defer_head_bucket_grad_sync(bucket)  # Idempotent installation.
    hook_calls = 0

    def accumulate(param):
        nonlocal hook_calls
        hook_calls += 1
        bucket.grad.add_(param.grad)
        param.grad = None
        # Exercise requests before the final chunk, including the old first-step
        # threshold. A premature collective would overwrite later contributions.
        if hook_calls >= 3:
            bucket.start_grad_sync(force_all_reduce=force_all_reduce)

    hook = weight.register_post_accumulate_grad_hook(accumulate)
    try:
        for step, last_chunks in enumerate((3, 5, 2, 4)):
            bucket.reset()
            hook_calls = 0
            expected = torch.zeros_like(weight)
            for chunks in (2, last_chunks):
                hidden = torch.randn(chunks * 3 - 1, 5, requires_grad=True)
                labels = torch.randint(11, (len(hidden),))
                mask = (torch.arange(len(hidden)) % 3 != 0).float()
                reference_weight = weight.detach().clone().requires_grad_()
                reference_hidden = hidden.detach().clone().requires_grad_()
                reference = (
                    F.cross_entropy(F.linear(reference_hidden, reference_weight), labels, reduction="none") * mask
                ).sum()
                mtp_losses = []
                for _ in range(mtp_depth):
                    mtp_hidden = torch.randn_like(hidden, requires_grad=True)
                    mtp_losses.append(0.1 * F.cross_entropy(F.linear(mtp_hidden, weight), labels, reduction="sum"))
                    reference = reference + 0.1 * F.cross_entropy(
                        F.linear(mtp_hidden.detach(), reference_weight), labels, reduction="sum"
                    )
                reference.backward()
                expected += reference_weight.grad

                def chunk_loss(values, targets, loss_mask):
                    return (F.cross_entropy(F.linear(values, weight), targets, reduction="none") * loss_mask).sum()

                actual = sum(
                    checkpoint(
                        chunk_loss,
                        hidden[start : start + 3],
                        labels[start : start + 3],
                        mask[start : start + 3],
                        use_reentrant=True,
                    )
                    for start in range(0, len(hidden), 3)
                ) + sum(mtp_losses)
                actual.backward()
                torch.testing.assert_close(actual, reference)
                torch.testing.assert_close(hidden.grad, reference_hidden.grad)
                assert len(bucket.snapshots) == step
            bucket.start_grad_sync(force_all_reduce=force_all_reduce)  # Explicit schedule request.
            assert len(bucket.snapshots) == step
            bucket.finish_grad_sync(force_all_reduce=force_all_reduce)
            bucket.finish_grad_sync(force_all_reduce=force_all_reduce)
            assert len(bucket.snapshots) == bucket.waits == bucket.copybacks == step + 1
            assert not bucket.param_gather_dispatched
            torch.testing.assert_close(bucket.snapshots[-1], expected)
            torch.testing.assert_close(bucket.grad, 2 * expected)
            assert bucket.force_choices[-1] is force_all_reduce
            with torch.no_grad():
                weight.sub_(0.001 * bucket.grad)
            bucket.is_first_batch = False
            bucket.reset()  # Relax also resets after the optimizer.
    finally:
        hook.remove()


def test_deferred_head_preserves_other_bucket_overlap():
    config = SimpleNamespace(overlap_grad_reduce=True)
    head, other = _Bucket((2,), config), _Bucket((2,), config)
    defer_head_bucket_grad_sync(head)
    head.grad.fill_(3)
    other.grad.fill_(7)
    head.start_grad_sync()
    other.start_grad_sync()
    assert not head.snapshots and len(other.snapshots) == 1
    assert config.overlap_grad_reduce
    other.is_first_batch = False
    other.finish_grad_sync()
    head.finish_grad_sync()
    torch.testing.assert_close(head.grad, torch.full((2,), 6.0))
    torch.testing.assert_close(other.grad, torch.full((2,), 14.0))


def test_deferred_head_rejects_unexpected_inflight_collective():
    bucket = _Bucket((2,))
    bucket.start_grad_sync()
    with pytest.raises(RuntimeError, match="before the first backward"):
        defer_head_bucket_grad_sync(bucket)
    bucket = _Bucket((2,))
    defer_head_bucket_grad_sync(bucket)
    bucket.grad_reduce_handle = object()
    with pytest.raises(RuntimeError, match="before finalization"):
        bucket.finish_grad_sync()
