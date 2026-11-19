# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import pytest

from relax.engine.rollout.deferred import (
    DeferredExecutor,
    DeferredState,
    PublicationReceipt,
    PublicationState,
    seal_batch,
)
from relax.utils.types import Sample


def setup_batch(attempt="attempt-1"):
    args = SimpleNamespace(vocab_size=10)
    sample = Sample(index=1, response_length=1, teacher_log_probs=[-1])
    batch = seal_batch(args, [sample], batch_id="b", rollout_id=0, required_fields=["teacher_log_probs"])
    executor = DeferredExecutor(args)
    handle = executor.submit_deferred(batch, None, operation_id="publish-b", attempt_id=attempt, samples=[sample])
    return executor, handle, sample


@pytest.mark.asyncio
@pytest.mark.parametrize("state", list(PublicationState) + [None])
async def test_deferred_requires_explicit_commit_receipt(state):
    executor, handle, _ = setup_batch()
    calls = []

    async def score(samples):
        return ()

    async def publish(payload, is_last):
        calls.append(payload)
        return None if state is None else PublicationReceipt(handle.operation_id, state)

    executor.start(handle, score=score, publish=publish)
    result = await executor.wait_deferred(handle)
    expected = {
        PublicationState.CONFIRMED: DeferredState.COMPLETED,
        PublicationState.NOT_SUBMITTED: DeferredState.FAILED,
    }.get(state, DeferredState.UNKNOWN)
    assert result.state is expected
    executor.start(handle, score=score, publish=publish)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_deferred_semantic_validation_prevents_publication():
    executor, handle, sample = setup_batch()

    async def score(samples):
        sample.teacher_log_probs = [float("nan")]
        return ()

    async def publish(*args):
        pytest.fail("invalid batch must not publish")

    executor.start(handle, score=score, publish=publish)
    result = await executor.wait_deferred(handle)
    assert result.state is DeferredState.FAILED
    assert "batch=b sample=1 field=teacher_log_probs" in result.error
    assert result.publication is PublicationState.NOT_SUBMITTED


@pytest.mark.asyncio
async def test_deferred_cancellation_preserves_unknown_commit_outcome():
    executor, handle, _ = setup_batch()
    entered = asyncio.Event()

    async def score(samples):
        return ()

    async def publish(*args):
        entered.set()
        await asyncio.Event().wait()

    executor.start(handle, score=score, publish=publish)
    await entered.wait()
    await executor.cancel()
    result = await executor.wait_deferred(handle)
    assert result.state is DeferredState.UNKNOWN
    assert result.publication is PublicationState.UNKNOWN
    other, other_handle, _ = setup_batch("attempt-2")
    assert other_handle.operation_id != handle.operation_id


@pytest.mark.asyncio
async def test_deferred_rejects_swapped_exports_with_same_training_index_before_publish():
    args = SimpleNamespace()
    samples = [Sample(index=3, response_length=1, teacher_log_probs=[value]) for value in (-0.2, -0.9)]
    batch = seal_batch(
        args,
        samples,
        batch_id="exports",
        rollout_id=0,
        required_fields=["teacher_log_probs"],
        scoring_ids=["session/export-a", "session/export-b"],
    )
    executor = DeferredExecutor(args)
    handle = executor.submit_deferred(batch, None, operation_id="b", attempt_id="attempt", samples=samples)

    async def score(rows):
        rows.reverse()
        return ()

    async def publish(*args):
        pytest.fail("reordered exports must not publish")

    executor.start(handle, score=score, publish=publish)
    result = await executor.wait_deferred(handle)
    assert result.state is DeferredState.FAILED
    assert "batch=exports sample=session/export-a field=identity" in result.error
    assert result.publication is PublicationState.NOT_SUBMITTED


@pytest.mark.asyncio
@pytest.mark.parametrize("topk", [False, True])
async def test_deferred_masked_sentinels_produce_finite_actual_opd_loss(topk):
    import numpy as np
    import torch

    from relax.utils.opd.opd_utils import _opd_compute_per_token_signal, reduce_opd_loss

    sample = Sample(index=1, response_length=2, loss_mask=[1, 0], teacher_log_probs=[-0.2, float("nan")])
    required = ["teacher_log_probs"]
    if topk:
        sample.opd_topk_token_ids = np.array([[1, -1], [-1, -1]], dtype=np.int32)
        sample.opd_topk_ksz = np.array([1, 0], dtype=np.int32)
        sample.opd_topk_teacher_log_probs = np.array([[-0.2, float("nan")], [float("inf"), float("nan")]])
        sample.opd_topk_student_log_probs = np.array([[-0.3, float("inf")], [float("nan"), float("nan")]])
        required.extend(["opd_topk_token_ids", "opd_topk_teacher_log_probs", "opd_topk_student_log_probs"])
    args = SimpleNamespace(vocab_size=3)
    batch = seal_batch(args, [sample], batch_id="masked", rollout_id=0, required_fields=required)
    executor = DeferredExecutor(args)
    handle = executor.submit_deferred(batch, None, operation_id="b", attempt_id="a", samples=[sample])
    losses = []

    async def score(samples):
        return ()

    async def publish(payload, is_last):
        assert payload[0] is sample
        if topk:
            ids = torch.as_tensor(sample.opd_topk_token_ids)
            assert ids.tolist() == [[1, -1], [-1, -1]]
            signal = _opd_compute_per_token_signal(
                student_lp_1d=None,
                teacher_lp_1d=None,
                student_lp_2d=torch.as_tensor(sample.opd_topk_student_log_probs),
                teacher_lp_2d=torch.as_tensor(sample.opd_topk_teacher_log_probs),
                token_selection="union",
                kl_type="reverse_kl",
                jsd_alpha=0.5,
                mask=torch.arange(2)[None, :] < torch.as_tensor(sample.opd_topk_ksz)[:, None],
            )
        else:
            signal = _opd_compute_per_token_signal(
                student_lp_1d=torch.tensor([-0.3, -0.4]),
                teacher_lp_1d=torch.tensor(sample.teacher_log_probs),
                token_selection="student_sampled",
                kl_type="reverse_kl",
                jsd_alpha=0.5,
            )
        losses.append(reduce_opd_loss({"response_lengths": [2], "loss_masks": [torch.tensor([1, 0])]}, signal))
        return PublicationReceipt(handle.operation_id, PublicationState.CONFIRMED)

    executor.start(handle, score=score, publish=publish)
    result = await executor.wait_deferred(handle)
    assert result.state is DeferredState.COMPLETED
    assert len(losses) == 1 and torch.isfinite(losses[0])
