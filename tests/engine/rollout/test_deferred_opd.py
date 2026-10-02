# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Deferred OPD scoring: nothing half-scored reaches the training queue.

The batch is sealed at submit time, so the validator compares what came back
against what was promised: presence, order, per-field row counts and the loss
mask. A single unscored sample fails the whole batch rather than publishing
rows whose distillation targets are missing.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from relax.engine.rollout.deferred import (
    DeferredExecutor,
    DeferredState,
    seal_batch,
    validate_scored_batch,
)
from relax.utils.types import Sample


REQUIRED = ("opd_topk_token_ids", "opd_topk_teacher_log_probs")


def build_args(**overrides):
    args = SimpleNamespace(
        opd_token_selection="student_topk",
        opd_teacher_key="data_source",
        use_opd=True,
        opd_type="sglang",
        use_agentic_rollout=False,
        colocate=True,
        hybrid=False,
        resource={"teacher": [1, 4], "actor": [1, 8], "rollout": [1, 4]},
        teacher_hf_checkpoint="/ckpt",
        opd_teacher_routes=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def build_sample(index: int, response_length: int = 3, *, scored: bool = True, k: int = 2) -> Sample:
    sample = Sample(
        index=index,
        group_index=index // 2,
        tokens=list(range(10 + response_length)),
        response_length=response_length,
        loss_mask=[1] * response_length,
        metadata={"data_source": "math"},
    )
    if scored and response_length:
        sample.opd_topk_token_ids = np.zeros((response_length, k), dtype=np.int64)
        sample.opd_topk_teacher_log_probs = np.zeros((response_length, k), dtype=np.float32)
    return sample


def seal(args, samples):
    return seal_batch(args, samples, batch_id="b-1", rollout_id=7, required_fields=REQUIRED)


# ----------------------------------------------------------------------
# Validation.
# ----------------------------------------------------------------------
def test_a_fully_scored_batch_validates():
    args = build_args()
    samples = [build_sample(0), build_sample(1)]
    missing, problems = validate_scored_batch(seal(args, samples), samples)
    assert missing == () and problems == []


def test_an_empty_response_needs_no_scoring_fields():
    args = build_args()
    samples = [build_sample(0), build_sample(1, response_length=0, scored=False)]
    ref = seal(args, samples)
    assert ref.eligible_count == 1
    missing, problems = validate_scored_batch(ref, samples)
    assert missing == () and problems == []


def test_a_missing_scoring_field_fails_the_sample():
    args = build_args()
    samples = [build_sample(0), build_sample(1, scored=False)]
    missing, problems = validate_scored_batch(seal(args, samples), samples)
    assert missing == (1,)
    assert "missing opd_topk_token_ids" in problems[0]


def test_a_field_that_is_one_row_short_fails_rather_than_misaligning():
    args = build_args()
    samples = [build_sample(0)]
    ref = seal(args, samples)
    samples[0].opd_topk_teacher_log_probs = np.zeros((2, 2), dtype=np.float32)
    missing, problems = validate_scored_batch(ref, samples)
    assert missing == (0,)
    assert "has 2 rows, expected 3" in problems[0]


def test_reordered_results_are_rejected():
    args = build_args()
    samples = [build_sample(0), build_sample(1)]
    ref = seal(args, samples)
    missing, problems = validate_scored_batch(ref, list(reversed(samples)))
    assert missing and "order changed" in problems[0]


def test_a_duplicated_sample_is_rejected():
    args = build_args()
    first = build_sample(0)
    ref = seal(args, [first, build_sample(0)])
    missing, problems = validate_scored_batch(ref, [first, first])
    assert missing == (0,)
    assert "appears twice" in problems[0]


def test_a_changed_batch_size_fails_every_sample():
    args = build_args()
    samples = [build_sample(0), build_sample(1)]
    missing, problems = validate_scored_batch(seal(args, samples), samples[:1])
    assert set(missing) == {0, 1}
    assert "batch size changed" in problems[0]


def test_a_loss_mask_that_stopped_matching_is_rejected():
    args = build_args()
    samples = [build_sample(0)]
    ref = seal(args, samples)
    samples[0].loss_mask = [1, 1]
    missing, problems = validate_scored_batch(ref, samples)
    assert missing == (0,)
    assert "loss mask covers 2 of 3" in problems[0]


def test_a_changed_response_length_is_rejected():
    args = build_args()
    samples = [build_sample(0)]
    ref = seal(args, samples)
    samples[0].response_length = 4
    missing, problems = validate_scored_batch(ref, samples)
    assert missing == (0,)
    assert "response length changed" in problems[0]


def test_sealing_records_routing_and_multimodal_facts():
    args = build_args()
    sample = build_sample(0)
    sample.multimodal_inputs = {"images": ["x"]}
    ref = seal(args, [sample])
    assert ref.samples[0].route_key == "math"
    assert ref.samples[0].has_multimodal is True
    assert ref.samples[0].prompt_length == 10
    assert ref.token_selection == "student_topk"
    assert ref.required_fields == REQUIRED


# ----------------------------------------------------------------------
# Executor.
# ----------------------------------------------------------------------
def run(coro):
    return asyncio.run(coro)


def test_executor_publishes_only_after_the_full_fixed_sequence():
    args = build_args()
    samples = [build_sample(0), build_sample(1)]
    executor = DeferredExecutor(args)
    handle = executor.submit_deferred(seal(args, samples), None, operation_id="op-1", samples=samples)
    states: list[DeferredState] = []
    published: list[tuple] = []

    async def score(batch):
        states.append(executor._records["op-1"].snapshot.state)
        return ()

    async def publish(payload, is_last):
        states.append(executor._records["op-1"].snapshot.state)
        published.append((payload, is_last))

    async def main():
        executor.start(handle, score=score, publish=publish)
        return await executor.wait_deferred(handle)

    result = run(main())
    assert result.state is DeferredState.COMPLETED
    assert states == [DeferredState.SCORING, DeferredState.PUBLISHING]
    assert published and published[0][1] is False
    assert result.scored == 2 and result.published == 2


def test_executor_does_not_publish_a_partially_scored_batch():
    args = build_args()
    samples = [build_sample(0), build_sample(1, scored=False)]
    executor = DeferredExecutor(args)
    handle = executor.submit_deferred(seal(args, samples), None, operation_id="op-1", samples=samples)
    published: list = []

    async def score(batch):
        return (1,)

    async def publish(payload, is_last):
        published.append(payload)

    async def main():
        executor.start(handle, score=score, publish=publish)
        return await executor.wait_deferred(handle)

    result = run(main())
    assert result.state is DeferredState.FAILED
    assert published == []
    assert result.missing == (1,)
    # The successful sample is still counted, for diagnosis only.
    assert result.scored == 1


def test_executor_fails_the_batch_when_scoring_raises():
    args = build_args()
    samples = [build_sample(0)]
    executor = DeferredExecutor(args)
    handle = executor.submit_deferred(seal(args, samples), None, operation_id="op-1", samples=samples)
    published: list = []

    async def score(batch):
        raise RuntimeError("teacher unreachable")

    async def publish(payload, is_last):
        published.append(payload)

    async def main():
        executor.start(handle, score=score, publish=publish)
        return await executor.wait_deferred(handle)

    result = run(main())
    assert result.state is DeferredState.FAILED
    assert "teacher unreachable" in result.error
    assert published == []


def test_executor_submit_is_idempotent_and_conflicts_on_other_inputs():
    args = build_args()
    samples = [build_sample(0)]
    other = [build_sample(2)]
    executor = DeferredExecutor(args)
    ref = seal(args, samples)
    handle = executor.submit_deferred(ref, None, operation_id="op-1", samples=samples)
    assert executor.submit_deferred(ref, None, operation_id="op-1", samples=samples) == handle
    with pytest.raises(ValueError, match="different inputs"):
        executor.submit_deferred(seal(args, other), None, operation_id="op-1", samples=other)


def test_wait_timeout_ends_the_wait_not_the_batch():
    args = build_args()
    samples = [build_sample(0)]
    executor = DeferredExecutor(args)
    handle = executor.submit_deferred(seal(args, samples), None, operation_id="op-1", samples=samples)
    release = asyncio.Event()

    async def score(batch):
        await release.wait()
        return ()

    async def publish(payload, is_last):
        pass

    async def main():
        task = executor.start(handle, score=score, publish=publish)
        timed_out = await executor.wait_deferred(handle, timeout_s=0.05)
        assert not timed_out.terminal
        release.set()
        return await task

    assert run(main()).state is DeferredState.COMPLETED


def test_batches_are_scored_one_at_a_time():
    args = build_args()
    executor = DeferredExecutor(args)
    handles = []
    for index in range(2):
        samples = [build_sample(index)]
        handles.append(
            executor.submit_deferred(
                seal_batch(args, samples, batch_id=f"b-{index}", rollout_id=7, required_fields=REQUIRED),
                None,
                operation_id=f"op-{index}",
                samples=samples,
            )
        )
    concurrent = 0
    peak = 0

    async def score(batch):
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.01)
        concurrent -= 1
        return ()

    async def publish(payload, is_last):
        pass

    async def main():
        tasks = [executor.start(handle, score=score, publish=publish) for handle in handles]
        await asyncio.gather(*tasks)

    run(main())
    assert peak == 1
    assert all(executor._records[f"op-{index}"].snapshot.state is DeferredState.COMPLETED for index in range(2))


def test_deferred_opd_active_for_agentic_rollout_on_shared_teacher():
    from relax.engine.rollout.deferred_opd import deferred_opd_active

    shared = build_args(
        use_agentic_rollout=True,
        rollout_num_gpus=8,
        resource={"teacher": [1, 8], "actor": [1, 8], "rollout": [1, 8]},
    )
    split = build_args(use_agentic_rollout=True, rollout_num_gpus=4)
    assert deferred_opd_active(shared)
    assert not deferred_opd_active(split)


def test_deferred_opd_publish_keeps_its_own_rollout_id_after_a_timed_out_wait():
    """A batch still publishing after its wait ended must not pick up the next
    iteration's loop values and land in another TQ partition."""
    from relax.engine.rollout.deferred_opd import DeferredOpdSession

    published = []

    async def record_publish(args, payload, count, rollout_id, client, *, is_last):
        published.append((count, rollout_id))

    opd_manager = SimpleNamespace(schema_opd_transfer_data=lambda: REQUIRED)
    session = DeferredOpdSession(build_args(), 1, None, opd_manager, publish=record_publish)
    pending = []

    def start(handle, *, score, publish):
        pending.append(publish)

    async def wait_deferred(handle, timeout_s=None):
        # The wait ends while the batch is still running, as a timeout would.
        return SimpleNamespace(state=DeferredState.COMPLETED, published=0)

    session.executor.start = start
    session.executor.wait_deferred = wait_deferred

    async def run():
        await session.transfer(None, [build_sample(0), build_sample(1)], 0, 1, None)
        await session.transfer(None, [build_sample(2), build_sample(3)], 1, 2, None)
        await session.flush()
        for publish in pending:
            await publish([], False)

    asyncio.run(run())

    assert published == [(0, 1), (1, 2)]


def test_deferred_opd_empty_last_batch_moves_end_of_stream_to_the_last_staged_batch():
    """A last batch whose groups expand to no samples must not drop the marker
    that closes a streaming partition."""
    from relax.engine.rollout.deferred_opd import DeferredOpdSession

    opd_manager = SimpleNamespace(schema_opd_transfer_data=lambda: REQUIRED)
    session = DeferredOpdSession(build_args(), 1, None, opd_manager, publish=None)

    async def run():
        await session.transfer(None, [build_sample(0), build_sample(1)], 0, 1, None)
        await session.transfer(None, [[], []], 1, 1, None, is_last=True)

    asyncio.run(run())

    assert session.staged_batches == 1
    assert session._staged[0][5] is True


def test_deferred_opd_empty_last_batch_without_staged_data_is_reported(caplog):
    from relax.engine.rollout.deferred_opd import DeferredOpdSession

    opd_manager = SimpleNamespace(schema_opd_transfer_data=lambda: REQUIRED)
    session = DeferredOpdSession(build_args(), 1, None, opd_manager, publish=None)

    with caplog.at_level("ERROR"):
        asyncio.run(session.transfer(None, [[]], 0, 1, None, is_last=True))

    assert session.staged_batches == 0
    assert "end-of-stream" in caplog.text
