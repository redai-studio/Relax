# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Map SGLang pre-dispatch HTTP 499 responses to ordinary rollout aborts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from relax.engine.rollout import sglang_rollout
from relax.engine.rollout.sglang_rollout import _post_generation
from relax.utils.types import Sample


def _status_error(status_code: int, *, body) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://test/generate")
    if isinstance(body, dict):
        response = httpx.Response(status_code, request=request, json=body)
    else:
        response = httpx.Response(status_code, request=request, text=body)
    return httpx.HTTPStatusError("boom", request=request, response=response)


_ABORT_499 = lambda: _status_error(499, body={"error": {"message": "Request abc was aborted"}})  # noqa: E731


@pytest.mark.parametrize("fully_async", [True, False])
@pytest.mark.asyncio
async def test_standard_post_maps_strict_499_to_generation_abort(monkeypatch, fully_async):
    async def fake_post(*_args, **_kwargs):
        raise _ABORT_499()

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    with pytest.raises(sglang_rollout.GenerationAborted):
        await _post_generation(
            _router_args(fully_async=fully_async),
            SimpleNamespace(aborted=False, abort_event=asyncio.Event()),
            "http://test/generate",
            {"rid": "stable"},
            None,
            evaluation=False,
        )


@pytest.mark.asyncio
async def test_standard_eval_keeps_sglang_499_visible(monkeypatch):
    error = _ABORT_499()

    async def fake_post(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await _post_generation(
            _router_args(),
            SimpleNamespace(aborted=False, abort_event=asyncio.Event()),
            "http://test/generate",
            {},
            None,
            evaluation=True,
        )
    assert exc_info.value is error


@pytest.mark.asyncio
async def test_standard_post_keeps_unrelated_499_visible(monkeypatch):
    error = _status_error(499, body={"error": {"message": "upstream closed"}})

    async def fake_post(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await _post_generation(
            _router_args(),
            SimpleNamespace(aborted=False, abort_event=asyncio.Event()),
            "http://test/generate",
            {},
            None,
            evaluation=False,
        )
    assert exc_info.value is error


def _router_args(**overrides):
    values = {"fully_async": True, "use_slime_router": False, "router_cb_timeout_duration_secs": -1.99}
    values.update(overrides)
    return SimpleNamespace(**values)


_NO_WORKERS_503 = lambda: _status_error(503, body={"error": {"code": "no_available_workers"}})  # noqa: E731


@pytest.mark.asyncio
async def test_router_no_workers_waits_once_then_retries(monkeypatch):
    calls = []
    payload = {}

    async def fake_post(_url, request_payload, **_kwargs):
        calls.append(request_payload)
        if len(calls) == 1:
            raise _NO_WORKERS_503()
        return {"ok": True}

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    result = await _post_generation(
        _router_args(),
        SimpleNamespace(aborted=False, abort_event=asyncio.Event()),
        "http://test/generate",
        payload,
        None,
        evaluation=False,
    )

    assert result == {"ok": True}
    assert calls == [payload, payload]


@pytest.mark.asyncio
async def test_router_recovery_stops_if_rollout_was_aborted(monkeypatch):
    state = SimpleNamespace(aborted=False, abort_event=asyncio.Event())
    calls = 0

    async def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        state.aborted = True
        state.abort_event.set()
        raise _NO_WORKERS_503()

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    with pytest.raises(sglang_rollout.GenerationAborted):
        await _post_generation(_router_args(), state, "http://test/generate", {}, None, evaluation=False)
    assert calls == 1


@pytest.mark.asyncio
async def test_router_recovery_does_not_retry_after_abort_during_wait(monkeypatch):
    state = SimpleNamespace(aborted=False, abort_event=asyncio.Event())
    calls = 0

    async def fake_post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise _NO_WORKERS_503()

    async def fake_wait_for(awaitable, timeout):
        del timeout
        awaitable.close()
        state.aborted = True
        state.abort_event.set()
        raise asyncio.TimeoutError

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    monkeypatch.setattr(sglang_rollout.asyncio, "wait_for", fake_wait_for)
    with pytest.raises(sglang_rollout.GenerationAborted):
        await _post_generation(_router_args(), state, "http://test/generate", {}, None, evaluation=False)
    assert calls == 1


@pytest.mark.asyncio
async def test_router_recovery_reports_persistent_503(monkeypatch):
    async def fake_post(*_args, **_kwargs):
        raise _NO_WORKERS_503()

    monkeypatch.setattr(sglang_rollout, "post", fake_post)
    with pytest.raises(RuntimeError, match="still has no available workers"):
        await _post_generation(
            _router_args(),
            SimpleNamespace(aborted=False, abort_event=asyncio.Event()),
            "http://test/generate",
            {},
            None,
            evaluation=False,
        )


# --- group-reward ordering: an aborted group must skip group reward -----------


class _GroupState:
    """Minimal GenerateState stand-in for generate_and_rm_group (no
    tokenizer)."""

    def __init__(self) -> None:
        self.aborted = False
        self.opd_manager = None


def _group_args(*, group_rm: bool) -> SimpleNamespace:
    return SimpleNamespace(
        group_rm=group_rm,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        sglang_enable_deterministic_inference=False,
    )


@pytest.mark.parametrize(("status", "expected_rm_calls"), [(Sample.Status.ABORTED, 0), (Sample.Status.COMPLETED, 1)])
@pytest.mark.asyncio
async def test_group_reward_only_runs_for_completed_group(monkeypatch, status, expected_rm_calls):
    async def fake_dispatch(state, args, sample, sampling_params, evaluation=False):
        sample.status = status
        return sample

    rm_calls: list = []

    async def fake_batched_rm(args, group):
        rm_calls.append(group)
        return [1.0] * len(group)

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda args: _GroupState())
    monkeypatch.setattr(sglang_rollout, "_dispatch_generate", fake_dispatch)
    monkeypatch.setattr(sglang_rollout, "batched_async_rm", fake_batched_rm)

    group = [Sample(prompt="hi"), Sample(prompt="hi")]
    result = await sglang_rollout.generate_and_rm_group(_group_args(group_rm=True), group, {"max_new_tokens": 8})

    assert all(sample.status == status for sample in result)
    assert len(rm_calls) == expected_rm_calls


class _RolloutState:
    def __init__(self, last_step_current_deficit=1):
        self.last_step_current_deficit = last_step_current_deficit
        self.remaining_batch_size = 0
        self.prefetched_samples_ref = None
        self.tokenizer = None
        self.pendings = set()
        self.protected_pendings = set()

    def submit_generate_tasks(self, samples):
        self.remaining_batch_size += len(samples)
        self.pendings.update(asyncio.create_task(asyncio.sleep(0, result=group)) for group in samples)

    def reset(self):
        pass


def _rollout_args(**overrides):
    values = {
        "rollout_global_dataset": True,
        "fully_async": True,
        "num_rollout": 1,
        "rollout_batch_size": 1,
        "over_sampling_batch_size": 1,
        "n_samples_per_prompt": 1,
        "dynamic_sampling_filter_path": None,
        "global_batch_size": 1,
        "num_iters_per_train_update": 1,
        "partial_rollout": False,
        "use_dynamic_global_batch_size": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_rollout(monkeypatch, state, transfer):
    async def noop(*_args, **_kwargs):
        pass

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout.ray, "get", lambda ref: ref)
    monkeypatch.setattr(sglang_rollout, "start_sglang_profile", noop)
    monkeypatch.setattr(sglang_rollout, "stop_sglang_profile", noop)
    monkeypatch.setattr(sglang_rollout, "transfer_batch_to_data_system", transfer)
    monkeypatch.setattr(sglang_rollout, "abort", lambda *_args: asyncio.sleep(0, result=([], [])))
    monkeypatch.setattr(sglang_rollout, "CURRENT_ROLLOUT_BATCH", [])


@pytest.mark.asyncio
async def test_final_backfill_replenishes_aborted_groups(monkeypatch):
    state = _RolloutState()
    aborted = [[Sample(index=i, status=Sample.Status.ABORTED)] for i in range(2)]
    completed = [[Sample(index=i, status=Sample.Status.COMPLETED)] for i in range(2, 4)]
    batches = [aborted, completed]
    transfers = []

    async def transfer(_args, batch, _count, rollout_id, _client, *, is_last=False):
        transfers.append((rollout_id, is_last, [group[0].index for group in batch]))

    _patch_rollout(monkeypatch, state, transfer)

    result, carry = await asyncio.wait_for(
        sglang_rollout.generate_rollout_async(
            _rollout_args(),
            rollout_id=1,
            data_source=SimpleNamespace(get_samples=SimpleNamespace(remote=lambda _count: batches.pop(0))),
            data_system_client=None,
        ),
        timeout=2,
    )

    assert len(result.samples) == 1 and transfers[0][:2] == (0, True)
    assert [sample.abort_count for group in carry for sample in group if sample.status == Sample.Status.ABORTED] == [
        1,
        1,
    ]


@pytest.mark.parametrize(("max_staleness", "expected_requests", "expected_deficit"), [(2, 1, 8), (0, 2, 0)])
@pytest.mark.asyncio
async def test_old_partition_is_paid_before_current(monkeypatch, max_staleness, expected_requests, expected_deficit):
    state = _RolloutState()
    first = [[Sample(index=i, status=Sample.Status.COMPLETED if i == 0 else Sample.Status.ABORTED)] for i in range(9)]
    second = [[Sample(index=i, status=Sample.Status.COMPLETED)] for i in range(9, 18)]
    batches = [first, second]
    requests, transfers = [], []

    def get_samples(_count):
        requests.append(None)
        return batches.pop(0)

    async def transfer(_args, batch, _count, rollout_id, _client, *, is_last=False):
        transfers.append((rollout_id, is_last, len(batch)))

    _patch_rollout(monkeypatch, state, transfer)
    args = _rollout_args(
        max_staleness=max_staleness,
        num_rollout=3,
        rollout_batch_size=8,
        over_sampling_batch_size=8,
        global_batch_size=8,
    )

    await asyncio.wait_for(
        sglang_rollout.generate_rollout_async(
            args,
            rollout_id=1,
            data_source=SimpleNamespace(get_samples=SimpleNamespace(remote=get_samples)),
            data_system_client=None,
        ),
        timeout=2,
    )

    assert len(requests) == expected_requests
    assert transfers == ([(0, True, 1)] if max_staleness == 2 else [(0, True, 1), (1, True, 8)])
    assert state.last_step_current_deficit == expected_deficit
