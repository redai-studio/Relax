# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exercise the native rollout loop; only inference and external I/O are
replaced."""

import asyncio
from argparse import Namespace
from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.utils.types import Sample


try:
    from relax.engine.rollout import sglang_rollout
except ModuleNotFoundError as exc:
    if exc.name.split(".")[0] not in {"ray", "sglang", "sglang_router", "pybase64"}:
        raise
    pytest.skip(f"Missing inference dependency: {exc.name}", allow_module_level=True)


def _args() -> Namespace:
    return Namespace(
        rollout_global_dataset=True,
        rollout_source_mixture="mixture.yaml",
        source_mixture_quotas={"A": 3, "B": 1},
        opd_teacher_key="data_source",
        dynamic_sampling_filter_path=None,
        fully_async=False,
        num_rollout=2,
        rollout_batch_size=4,
        over_sampling_batch_size=4,
        n_samples_per_prompt=1,
        global_batch_size=4,
        num_iters_per_train_update=1,
        partial_rollout=False,
        use_dynamic_global_batch_size=False,
        debug_rollout_only=False,
        reward_key=None,
    )


def _groups() -> list[list[Sample]]:
    return [
        [
            Sample(
                index=i,
                group_index=i,
                prompt=f"prompt-{i}",
                response="response",
                status=Sample.Status.COMPLETED,
                reward=0.0,
                metadata={"data_source": source, "_source_mixture_batch_id": 0},
            )
        ]
        for i, source in enumerate(("A", "A", "A", "B"))
    ]


@pytest.fixture
def rollout_harness(monkeypatch: pytest.MonkeyPatch) -> Callable:
    def make(groups: list[list[Sample]], *, cached: bool = False, mutate: Callable | None = None) -> SimpleNamespace:
        state = SimpleNamespace(
            last_step_current_deficit=0,
            remaining_batch_size=0,
            prefetched_samples_ref=groups if cached else None,
            pendings=set(),
            protected_pendings=set(),
            tokenizer=None,
            reset=Mock(),
        )
        completed = []

        async def generate(group: list[Sample]) -> list[Sample]:
            # B finishes first. A quota must not be replaced with faster B samples.
            await asyncio.sleep(0.01 if group[0].metadata["data_source"] == "A" else 0)
            if mutate is not None:
                mutate(group)
            completed.append(group[0].index)
            return group

        def submit(batch: list[list[Sample]]) -> None:
            state.remaining_batch_size += len(batch)
            state.pendings.update(asyncio.create_task(generate(group)) for group in batch)

        state.submit_generate_tasks = Mock(side_effect=submit)
        current_batch = []
        puts = []

        async def transfer(args: Namespace, batch: list[list[Sample]], size: int, *pos: object, **kw: object) -> None:
            assert len(completed) == 4  # No partial batch is submitted before all sources finish.
            puts.append(batch)
            current_batch.extend(sample for group in batch for sample in group)

        source = SimpleNamespace(get_samples=SimpleNamespace(remote=Mock(return_value=groups)))
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
        # Expected fatal batches must not leave the process-wide timer running
        # and contaminate later test cases.
        monkeypatch.setattr(sglang_rollout, "Timer", Mock)
        monkeypatch.setattr(sglang_rollout, "ray", SimpleNamespace(get=lambda ref: ref))
        monkeypatch.setattr(sglang_rollout, "start_sglang_profile", AsyncMock())
        monkeypatch.setattr(sglang_rollout, "stop_sglang_profile", AsyncMock())
        monkeypatch.setattr(sglang_rollout, "abort", AsyncMock(return_value=([], [])))
        monkeypatch.setattr(sglang_rollout, "transfer_batch_to_data_system", transfer)
        monkeypatch.setattr(sglang_rollout, "CURRENT_ROLLOUT_BATCH", current_batch)
        monkeypatch.setattr(sglang_rollout, "save_debug_rollout_data", Mock())
        monkeypatch.setattr(sglang_rollout, "_log_rollout_data", Mock())
        monkeypatch.setattr(sglang_rollout, "tqdm", Mock())

        async def run() -> object:
            try:
                return await asyncio.wait_for(
                    sglang_rollout.generate_rollout_async(_args(), 0, source, object()), timeout=5
                )
            finally:
                for task in state.pendings:
                    task.cancel()
                await asyncio.gather(*state.pendings, return_exceptions=True)

        return SimpleNamespace(run=run, source=source, state=state, puts=puts, completed=completed)

    return make


@pytest.mark.parametrize("cached", [False, True])
async def test_source_mixture_rollout_commits_exact_quotas_despite_completion_order(
    rollout_harness: Callable, cached: bool
) -> None:
    harness = rollout_harness(_groups(), cached=cached)
    output, buffered = await harness.run()
    assert harness.completed[0] == 3
    assert len(harness.puts) == 1
    assert Counter(group[0].metadata["data_source"] for group in output.samples) == {"A": 3, "B": 1}
    assert buffered == []
    if cached:
        harness.source.get_samples.remote.assert_not_called()
    else:
        harness.source.get_samples.remote.assert_called_once_with(4, rollout_id=0)


@pytest.mark.parametrize("failure", ["short", "wrong_source", "wrong_batch", "duplicate"])
async def test_source_mixture_rollout_rejects_invalid_prefetch_before_inference(
    rollout_harness: Callable, failure: str
) -> None:
    groups = _groups()
    if failure == "short":
        groups.pop()
    elif failure == "wrong_source":
        groups[0][0].metadata["data_source"] = "B"
    elif failure == "wrong_batch":
        groups[0][0].metadata["_source_mixture_batch_id"] = 1
    else:
        groups[0][0].index = groups[1][0].index
    harness = rollout_harness(groups, cached=True)
    with pytest.raises((ValueError, RuntimeError)):
        await harness.run()
    harness.state.submit_generate_tasks.assert_not_called()
    assert harness.puts == []


@pytest.mark.parametrize("failure", ["aborted", "wrong_source", "unfinished"])
async def test_source_mixture_rollout_rechecks_completed_batch_before_transfer(
    rollout_harness: Callable, failure: str
) -> None:
    def mutate(group: list[Sample]) -> None:
        if group[0].index != 0:
            return
        if failure == "wrong_source":
            group[0].metadata["data_source"] = "B"
        else:
            group[0].status = Sample.Status.ABORTED if failure == "aborted" else Sample.Status.PENDING

    harness = rollout_harness(_groups(), mutate=mutate)
    with pytest.raises((ValueError, RuntimeError)):
        await harness.run()
    assert harness.puts == []


@pytest.mark.parametrize("mixture,rollout_id", [(True, 0), (True, 1), (False, 0)])
def test_source_mixture_prefetch_reserves_next_batch_but_not_after_final_step(
    monkeypatch: pytest.MonkeyPatch, mixture: bool, rollout_id: int
) -> None:
    args = _args()
    if not mixture:
        args.rollout_source_mixture = None
    state = SimpleNamespace(prefetched_samples_ref=None)
    source = SimpleNamespace(get_samples=SimpleNamespace(remote=Mock(return_value="next batch")))
    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
    monkeypatch.setattr(sglang_rollout, "generate_rollout_async", AsyncMock(return_value=("output", [])))
    monkeypatch.setattr(sglang_rollout, "run", asyncio.run)
    assert sglang_rollout.generate_rollout(args, rollout_id, source, object()) == "output"
    if mixture and rollout_id == 1:
        source.get_samples.remote.assert_not_called()
        assert state.prefetched_samples_ref is None
    else:
        source.get_samples.remote.assert_called_once_with(4, **({"rollout_id": 1} if mixture else {}))
        assert state.prefetched_samples_ref == "next batch"
