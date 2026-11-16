# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU regressions for rollout filtering; generation, Ray I/O and logging are
stubbed."""

import asyncio
from argparse import Namespace
from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from relax.engine.filters.dynamic_sampling_filters import check_reward_nonzero_std
from relax.utils.types import Sample


try:
    from relax.engine.rollout import sglang_rollout
except ModuleNotFoundError as exc:
    if exc.name.split(".")[0] not in {"ray", "sglang", "sglang_router", "pybase64"}:
        raise
    pytest.skip(f"Missing inference dependency: {exc.name}", allow_module_level=True)


Group = list[Sample]


def _group(
    index: int,
    rewards: tuple[float | None, float | None] = (0.0, 1.0),
    *,
    reward_key: str | None = None,
    status: Sample.Status = Sample.Status.COMPLETED,
) -> Group:
    return [
        Sample(
            group_index=index,
            reward={reward_key: reward} if reward_key and reward is not None else reward,
            status=status,
            metadata={"_timing": {"generate": 2.0}},
        )
        for reward in rewards
    ]


def _group_ids(groups: list[Group]) -> set[int]:
    return {group[0].group_index for group in groups}


@pytest.fixture
def run_rollout(monkeypatch: pytest.MonkeyPatch) -> Callable:
    async def run(
        batches: list[list[Group]],
        *,
        drained: tuple[list[Group], list[Group]] | None = None,
        fully_async: bool = False,
        partial_rollout: bool = False,
        dynamic_global_batch_size: bool = True,
        reward_key: str | None = None,
        filter_fn: Callable | None = check_reward_nonzero_std,
        rollout_batch_size: int = 1,
        dp_size: int = 1,
    ) -> SimpleNamespace:
        args = Namespace(
            rollout_global_dataset=True,
            dynamic_sampling_filter_path="test_filter" if filter_fn else None,
            fully_async=fully_async,
            num_rollout=2,
            rollout_batch_size=rollout_batch_size,
            over_sampling_batch_size=max(rollout_batch_size, len(batches[0])),
            n_samples_per_prompt=2,
            global_batch_size=2 * rollout_batch_size,
            num_iters_per_train_update=1,
            partial_rollout=partial_rollout,
            use_dynamic_global_batch_size=partial_rollout and dynamic_global_batch_size,
            debug_rollout_only=False,
            reward_key=reward_key,
        )
        state = SimpleNamespace(
            last_step_current_deficit=0,
            remaining_batch_size=0,
            prefetched_samples_ref=None,
            pendings=set(),
            protected_pendings=set(),
            tokenizer=None,
            reset=Mock(),
        )

        async def completed(group: Group) -> Group:
            return group

        def submit(groups: list[Group]) -> None:
            state.remaining_batch_size += len(groups)
            state.pendings.update(asyncio.create_task(completed(group)) for group in groups)

        state.submit_generate_tasks = submit
        current_batch = []
        transferred = []

        async def transfer(
            args: Namespace,
            groups: list[Group],
            batch_size: int,
            rollout_id: int,
            client: object,
            **kwargs: object,
        ) -> None:
            assert batch_size == len(groups)
            transferred.extend(groups)
            current_batch.extend(sample for group in groups for sample in group)

        data_source = SimpleNamespace(get_samples=SimpleNamespace(remote=Mock(side_effect=batches)))
        log = Mock()
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: state)
        monkeypatch.setattr(sglang_rollout, "ray", SimpleNamespace(get=lambda ref: ref))
        monkeypatch.setattr(sglang_rollout, "load_function", lambda path: filter_fn)
        monkeypatch.setattr(sglang_rollout, "start_sglang_profile", AsyncMock())
        monkeypatch.setattr(sglang_rollout, "stop_sglang_profile", AsyncMock())
        monkeypatch.setattr(sglang_rollout, "abort", AsyncMock(return_value=drained or ([], [])))
        monkeypatch.setattr(sglang_rollout, "transfer_batch_to_data_system", transfer)
        monkeypatch.setattr(sglang_rollout, "CURRENT_ROLLOUT_BATCH", current_batch)
        monkeypatch.setattr(sglang_rollout, "compute_dp_size", lambda args: dp_size)
        monkeypatch.setattr(sglang_rollout, "save_debug_rollout_data", Mock())
        monkeypatch.setattr(sglang_rollout, "_log_rollout_data", log)
        monkeypatch.setattr(sglang_rollout, "tqdm", Mock())

        output, buffered = await asyncio.wait_for(
            sglang_rollout.generate_rollout_async(args, 0, data_source, object()), timeout=5
        )
        log.assert_called_once()
        state.reset.assert_called_once()
        return SimpleNamespace(
            output=output,
            buffered=buffered,
            transferred=transferred,
            logged_metrics=log.call_args.args[3],
            fetch=data_source.get_samples.remote,
        )

    return run


@pytest.mark.parametrize("fully_async", [False, True])
@pytest.mark.parametrize("reward_key", [None, "score"])
async def test_sglang_rollout_logs_dynamic_filter_drops_and_preserves_timing(
    run_rollout: Callable, fully_async: bool, reward_key: str | None
) -> None:
    rejected = _group(0, (0.3, 0.3), reward_key=reward_key)
    accepted = _group(1, reward_key=reward_key)
    result = await run_rollout([[rejected], [accepted]], fully_async=fully_async, reward_key=reward_key)

    assert result.output.samples == [accepted]
    assert result.transferred == [accepted]
    assert result.buffered == []
    assert result.fetch.call_count == 2  # A dropped group causes replacement sampling.
    expected = {"rollout/dynamic_filter/drop_zero_std_0.3": 1}
    assert result.output.metrics == expected
    assert result.logged_metrics.items() >= expected.items()
    assert result.logged_metrics["perf_detail/rollout/generate_time/mean"] == 2.0
    assert result.logged_metrics["perf_detail/rollout/generate_time/max"] == 2.0
    assert result.logged_metrics["perf_detail/rollout/get_samples_time/total"] >= 0


@pytest.mark.parametrize("filter_kind", ["structured", "legacy", "disabled"])
async def test_sglang_rollout_filters_completed_drain_before_dp_trim_and_keeps_surplus(
    run_rollout: Callable, filter_kind: str
) -> None:
    # All main-loop groups finish together: two are committed and one is surplus.
    main_groups = [_group(index, reward_key="score") for index in range(3)]
    pending_drop = _group(3, (0.3, 0.3), reward_key="score")
    pending_keep = _group(4, reward_key="score")
    partial = _group(5, (None, None), status=Sample.Status.ABORTED)
    protected_drop = _group(6, (0.7, 0.7), reward_key="score")
    protected_keep = _group(7, reward_key="score", status=Sample.Status.TRUNCATED)

    def dynamic_filter(args: Namespace, group: Group) -> object:
        output = check_reward_nonzero_std(args, group)
        return bool(output.keep) if filter_kind == "legacy" else output

    filter_fn = Mock(side_effect=dynamic_filter) if filter_kind != "disabled" else None
    result = await run_rollout(
        [main_groups],
        drained=([pending_drop, pending_keep, partial], [protected_drop, protected_keep]),
        partial_rollout=True,
        reward_key="score",
        filter_fn=filter_fn,
        rollout_batch_size=2,
        dp_size=2,
    )

    transferred = _group_ids(result.transferred)
    buffered = _group_ids(result.buffered)
    rejected = {3, 6} if filter_fn else set()
    expected_completed = {0, 1, 2, 3, 4, 6, 7} - rejected
    assert transferred == _group_ids(result.output.samples)
    assert len(result.transferred) == (4 if filter_fn else 6)
    assert len(result.transferred) % 2 == 0
    assert {4, 7} <= transferred  # Both pending and protected completion paths.
    assert transferred.isdisjoint(buffered)
    assert (transferred | buffered) == expected_completed | {5}
    assert 5 in buffered
    assert len(buffered & {0, 1, 2}) == 1  # Legal surplus survives DP trimming.
    if filter_fn:
        assert Counter(call.args[1][0].group_index for call in filter_fn.call_args_list) == Counter(
            {index: 1 for index in (0, 1, 2, 3, 4, 6, 7)}
        )  # Incomplete groups are skipped; main-loop surplus is not filtered twice.
    expected_metrics = (
        {"rollout/dynamic_filter/drop_zero_std_0.3": 1, "rollout/dynamic_filter/drop_zero_std_0.7": 1}
        if filter_kind == "structured"
        else {}
    )
    assert result.output.metrics == expected_metrics
    assert result.logged_metrics.items() >= expected_metrics.items()


async def test_sglang_rollout_defers_drain_filter_without_dynamic_batch_size(run_rollout: Callable) -> None:
    accepted = _group(0)
    drained = _group(1, (0.3, 0.3))
    filter_fn = Mock(wraps=check_reward_nonzero_std)
    result = await run_rollout(
        [[accepted]],
        drained=([drained], []),
        partial_rollout=True,
        dynamic_global_batch_size=False,
        filter_fn=filter_fn,
    )

    assert result.transferred == [accepted]
    assert result.buffered == [drained]
    assert result.output.metrics == {}
    assert [call.args[1] for call in filter_fn.call_args_list] == [accepted]
