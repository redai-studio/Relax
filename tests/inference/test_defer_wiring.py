# Copyright (c) 2026 Relax Authors. All Rights Reserved.


import asyncio
from types import SimpleNamespace

import pytest

from relax.engine.rollout.on_policy_distillation import OpdManager
from relax.inference import defer as defer_module
from relax.inference.defer import (
    DeferredTransferCollector,
    _score_and_publish,
    capture_deferred_transfer,
    capture_transfers,
    run_deferred_rollout,
    validate_deferred_workload,
    wait_inference_commit,
)


def _args(**values):
    args = dict(
        inference_defer_roles=["teacher"],
        colocate=True,
        fully_async=False,
        use_opd=True,
        opd_type="sglang",
        opd_token_selection="student_sampled",
        opd_teacher_prompt_key=None,
        opd_teacher_image_key=None,
        _inference_run_id="test-run",
        rollout_http_timeout=1.0,
    )
    args.update(values)
    return SimpleNamespace(**args)


def _sample(index):
    return SimpleNamespace(
        index=index,
        group_index=index,
        response_length=1,
        tokens=[1, 2],
        teacher_log_probs=None,
        rollout_log_probs=[-1.0],
        loss_mask=[1],
        weight_versions=["v1"],
        reward=1.0,
        remove_sample=False,
    )


class _ObjectRef:
    def __init__(self, value=None):
        self.value = value

    def __await__(self):
        async def resolve():
            return self.value

        return resolve().__await__()


class _Coordinator:
    def __init__(self, events):
        self.events = events

    def __getattr__(self, method):
        def remote(*args, **kwargs):
            self.events.append((method, args))
            return _ObjectRef()

        return SimpleNamespace(remote=remote)


def test_deferred_context_survives_run_coroutine_threadsafe_and_resets_afterward():
    from relax.utils.async_utils import AsyncLoopThread

    loop = AsyncLoopThread()
    collector = DeferredTransferCollector(7)
    sample = _sample(1)
    args = _args()

    async def capture():
        return capture_deferred_transfer(args, [[sample]], 7)

    try:
        with capture_transfers(collector):
            assert loop.run(capture()) is True
        assert collector.groups == [[sample]]
        with pytest.raises(RuntimeError, match="outside"):
            loop.run(capture())
    finally:
        loop.loop.call_soon_threadsafe(loop.loop.stop)
        loop._thread.join(timeout=1)
        loop.loop.close()


async def test_native_transfer_capture_precedes_conversion_and_tq_write(monkeypatch):
    from relax.utils import utils

    def convert(*args):
        raise AssertionError("Conversion must wait for deferred scoring")

    async def put(**kwargs):
        raise AssertionError("No unscored rows may be published")

    monkeypatch.setattr(utils, "convert_samples_to_train_data", convert)
    collector = DeferredTransferCollector(3)
    rows = [[_sample(1), _sample(2)]]
    with capture_transfers(collector):
        await utils.transfer_batch_to_data_system(_args(), rows, 0, 3, SimpleNamespace(async_put=put), is_last=True)
    assert collector.groups == rows


def test_deferred_capture_checks_partition_and_disabled_mode():
    assert capture_deferred_transfer(_args(inference_defer_roles=[]), [[_sample(1)]], 0) is False
    collector = DeferredTransferCollector(1)
    with capture_transfers(collector):
        with pytest.raises(ValueError, match="cross rollout"):
            capture_deferred_transfer(_args(), [[_sample(1)]], 2)


def test_deferred_finalize_registration_requires_managed_batch():
    callback = lambda _samples: None
    from relax.inference.defer import register_deferred_rollout_finalize

    assert register_deferred_rollout_finalize(_args(inference_defer_roles=[]), 1, callback) is False
    with pytest.raises(RuntimeError, match="outside"):
        register_deferred_rollout_finalize(_args(), 1, callback)

    collector = DeferredTransferCollector(1)
    with capture_transfers(collector):
        assert register_deferred_rollout_finalize(_args(), 1, callback) is True
        with pytest.raises(RuntimeError, match="already"):
            register_deferred_rollout_finalize(_args(), 1, lambda _samples: None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"fully_async": True},
        {"colocate": False},
        {"partial_rollout": True},
        {"debug_rollout_only": True},
        {"debug_train_only": True},
        {"use_dynamic_global_batch_size": True},
        {"dynamic_sampling_filter_path": "filter"},
        {"rollout_sample_filter_path": "filter"},
        {"custom_reward_post_process_path": "hook"},
        {"custom_convert_samples_to_train_data_path": "hook"},
        {"agentic_custom_advantage_path": "hook"},
        {"train_backend": "fsdp"},
        {"loss_type": "sft"},
        {"use_opd": False},
        {"opd_type": "megatron"},
        {"eval_interval": 2, "eval_prompt_data": ["data"]},
        {"rollout_function_path": "custom.generate"},
        {"inference_defer_roles": ["unknown"]},
    ],
)
def test_deferred_preflight_rejects_unsupported_business_dependencies(overrides):
    with pytest.raises(ValueError):
        validate_deferred_workload(_args(**overrides))


def test_deferred_preflight_accepts_supported_native_agentic_and_genrm_adapters():
    validate_deferred_workload(_args())
    validate_deferred_workload(_args(use_agentic_rollout=True))
    validate_deferred_workload(
        _args(
            inference_defer_roles=["genrm"],
            _genrm_instances_resolved={"judge": {}},
            rm_type="dapo-genrm",
            use_opd=False,
        )
    )
    with pytest.raises(ValueError, match="one model"):
        validate_deferred_workload(
            _args(
                inference_defer_roles=["genrm"],
                _genrm_instances_resolved={"one": {}, "two": {}},
                rm_type="dapo-genrm",
                use_opd=False,
            )
        )


@pytest.fixture
def score_environment(monkeypatch):
    events = []

    async def prepare(self, samples):
        events.append(("prepare", tuple(sample.index for sample in samples)))

    async def teacher(self, samples):
        events.append(("teacher", tuple(sample.index for sample in samples)))
        for sample in samples:
            sample.teacher_log_probs = [-float(sample.index)]

    def convert(args, samples):
        assert all(sample.teacher_log_probs is not None for sample in samples)
        events.append(("convert", tuple(sample.index for sample in samples)))
        return list(samples)

    monkeypatch.setattr(OpdManager, "prepare", prepare)
    monkeypatch.setattr(OpdManager, "teacher_prefill", teacher)
    monkeypatch.setattr(defer_module, "convert_samples_to_train_data", convert)
    monkeypatch.setattr(defer_module, "build_rollout_custom_meta", lambda batch: {"rows": len(batch)})

    async def put(**kwargs):
        events.append(("put", kwargs))

    coordinator = _Coordinator(events)
    args = _args(_inference_coordinator=coordinator)
    manager = SimpleNamespace(
        args=args,
        data_system_client=SimpleNamespace(async_put=put),
        _offload_local=lambda: events.append(("offload_local", ())),
        data_source=object(),
    )
    return manager, events


async def test_deferred_score_and_publish_orders_release_conversion_and_commit(score_environment):
    manager, events = score_environment
    rows = [_sample(2), _sample(1)]
    collector = DeferredTransferCollector(9, groups=[rows])

    def finalize(samples):
        assert all(sample.teacher_log_probs is not None for sample in samples)
        events.append(("finalize", tuple(sample.index for sample in samples)))

    collector.register_finalize(finalize)

    await _score_and_publish(manager, collector)

    names = [name for name, _ in events]
    assert names == [
        "prepare",
        "offload_local",
        "acknowledge_rollout_offloaded",
        "activate",
        "teacher",
        "deactivate",
        "convert",
        "put",
        "finalize",
        "commit_batch",
    ]
    put = next(value for name, value in events if name == "put")
    assert put["data"] == rows
    assert put["partition_id"] == "train_9"
    assert put["is_last"] is True
    assert rows[0].teacher_log_probs == [-2.0]
    assert collector.finalized is True


async def test_deferred_tq_failure_does_not_emit_success_telemetry(score_environment):
    manager, events = score_environment
    collector = DeferredTransferCollector(10, groups=[[_sample(1)]])
    collector.register_finalize(lambda _samples: events.append(("finalize", ())))

    async def fail_put(**_kwargs):
        events.append(("put_failed", ()))
        raise RuntimeError("TQ unavailable")

    manager.data_system_client.async_put = fail_put

    with pytest.raises(RuntimeError, match="TQ unavailable"):
        await _score_and_publish(manager, collector)

    assert "finalize" not in [name for name, _value in events]
    assert "commit_batch" not in [name for name, _value in events]


async def test_deferred_finalizer_failure_after_put_clears_uncommitted_partition(score_environment):
    manager, events = score_environment
    collector = DeferredTransferCollector(11, groups=[[_sample(1)]])

    def failing_finalize(_samples):
        raise AssertionError("finalizer invariant failed")

    async def clear(**kwargs):
        events.append(("clear", kwargs))

    collector.register_finalize(failing_finalize)
    manager.data_system_client.async_clear_partition = clear

    with pytest.raises(AssertionError, match="finalizer invariant"):
        await _score_and_publish(manager, collector)

    names = [name for name, _value in events]
    assert names.index("put") < names.index("clear")
    assert ("clear", {"partition_id": "train_11"}) in events
    assert "commit_batch" not in names


def _run_failing_producer(manager, monkeypatch, rollout_id):
    from relax.utils import async_utils

    private_loop = async_utils.AsyncLoopThread()
    monkeypatch.setattr(async_utils, "get_async_loop", lambda: private_loop)

    def produce(args, rollout_id, data_source, data_client, evaluation):
        raise RuntimeError("producer crashed")

    manager.generate_rollout = produce
    try:
        with pytest.raises(RuntimeError, match="producer crashed"):
            run_deferred_rollout(manager, rollout_id)
    finally:
        private_loop.loop.call_soon_threadsafe(private_loop.loop.stop)
        private_loop._thread.join(timeout=1)
        private_loop.loop.close()


def test_run_deferred_rollout_producer_failure_releases_and_acknowledges_rollout(score_environment, monkeypatch):
    manager, events = score_environment

    _run_failing_producer(manager, monkeypatch, 4)

    names = [name for name, _value in events]
    assert names == ["begin_batch", "offload_local", "acknowledge_rollout_offloaded", "fail_batch"]
    assert events[-1][1][0] == 4
    assert "producer crashed" in events[-1][1][1]


def test_run_deferred_rollout_unconfirmed_release_keeps_lease_reserved(score_environment, monkeypatch):
    manager, events = score_environment

    def failing_offload():
        raise TimeoutError("release timed out")

    manager._offload_local = failing_offload

    _run_failing_producer(manager, monkeypatch, 5)

    names = [name for name, _value in events]
    assert names == ["begin_batch", "fail_batch"]
    assert "rollout release unconfirmed" in events[-1][1][1]


def test_run_deferred_rollout_bridges_ray_refs_and_captures_threadsafe_producer(score_environment, monkeypatch):
    manager, events = score_environment
    from relax.utils import async_utils

    private_loop = async_utils.AsyncLoopThread()
    monkeypatch.setattr(async_utils, "get_async_loop", lambda: private_loop)
    rows = [[_sample(1)]]

    def produce(args, rollout_id, data_source, data_client, evaluation):
        assert evaluation is False

        async def transfer():
            assert capture_deferred_transfer(args, rows, rollout_id)

        async_utils.run(transfer())
        return rows

    manager.generate_rollout = produce
    try:
        output = run_deferred_rollout(manager, 3)
    finally:
        private_loop.loop.call_soon_threadsafe(private_loop.loop.stop)
        private_loop._thread.join(timeout=1)
        private_loop.loop.close()

    assert output.samples == rows
    assert events[0][0] == "begin_batch"
    assert events[-1] == ("commit_batch", (3,))


async def test_training_commit_wait_uses_coordinator_and_rejects_missing_gate():
    events = []
    await wait_inference_commit(_args(_inference_coordinator=_Coordinator(events)), 12)
    assert events == [("wait_committed", (12,))]
    with pytest.raises(RuntimeError, match="no commit coordinator"):
        await wait_inference_commit(_args(), 12)


async def test_deferred_reward_exception_waits_for_sibling_before_role_release(score_environment, monkeypatch):
    manager, events = score_environment
    manager.args.inference_defer_roles = ["genrm"]
    completed = asyncio.Event()

    async def score(args, sample):
        assert args.inference_defer_roles == ["genrm"]
        if sample.index == 1:
            raise RuntimeError("judge failed")
        await asyncio.sleep(0)
        events.append(("judge_sibling_done", ()))
        completed.set()
        return {"score": 1.0}

    monkeypatch.setattr(defer_module, "async_compute_score_genrm", score)
    collector = DeferredTransferCollector(4, groups=[[_sample(1), _sample(2)]])

    with pytest.raises(RuntimeError, match="judge failed"):
        await _score_and_publish(manager, collector)

    names = [name for name, _args in events]
    assert completed.is_set()
    assert names.index("judge_sibling_done") < names.index("deactivate")
    assert "convert" not in names
    assert "put" not in names


async def test_deferred_genrm_writeback_uses_selected_reward_field(score_environment, monkeypatch):
    manager, _events = score_environment
    manager.args.inference_defer_roles = ["genrm"]
    manager.args.reward_key = "acc"

    async def score(_args, _sample):
        return {"score": 0.25, "acc": 1, "format_error": ""}

    monkeypatch.setattr(defer_module, "async_compute_score_genrm", score)
    monkeypatch.setattr(defer_module, "convert_samples_to_train_data", lambda _args, samples: list(samples))
    sample = _sample(1)

    await _score_and_publish(manager, DeferredTransferCollector(5, groups=[[sample]]))

    assert sample.reward == {"acc": 1}


async def test_deferred_placeholder_reward_is_numeric_not_an_unawaited_coroutine():
    from relax.engine.rewards import async_rm

    result = await async_rm(_args(inference_defer_roles=["genrm"], reward_key="score"), _sample(1))

    assert result == {"score": 0.0}
