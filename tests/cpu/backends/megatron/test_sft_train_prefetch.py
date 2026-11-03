# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exercise the production CPU prefetch functions without a Ray/GPU runtime."""

import ast
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event, current_thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _load_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    actor = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"), None
    )
    if actor is not None:
        methods = [node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name in names]
        nodes.append(ast.ClassDef(name="Actor", bases=[], keywords=[], body=methods, decorator_list=[]))
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)


@pytest.fixture
def runtime():
    main_thread = current_thread()
    groups = []

    def main_only(value):
        def call(*args, **kwargs):
            assert current_thread() is main_thread
            return value

        return call

    def all_reduce(tensor, *, op, group):
        assert current_thread() is main_thread
        groups.append(group)

    namespace = dict(
        time=time,
        CancelledError=CancelledError,
        Event=Event,
        torch=torch,
        logger=Mock(),
        timer=lambda _: nullcontext(),
        Envs=SimpleNamespace(RELAX_EMPTY_POLL_SLEEP_MS=5),
        is_megatron_main_rank=main_only(True),
        sft_partition_id=lambda args, rollout_id: f"sft_{rollout_id}",
        evaluation_step_for_rollout=lambda args, rollout_id: rollout_id + 1,
        should_run_sft_eval=lambda args, rollout_id: getattr(args, "eval_at", None) == rollout_id,
        should_run_sft_predict=lambda args, rollout_id: False,
        device_utils=SimpleNamespace(make_current_torch_device=main_only(torch.device("cpu"))),
        mpu=SimpleNamespace(
            get_data_parallel_rank=main_only(3),
            get_tensor_and_context_parallel_group=main_only("tp_cp"),
            get_pipeline_model_parallel_world_size=main_only(2),
            get_pipeline_model_parallel_group=main_only("pp"),
        ),
        dist=SimpleNamespace(all_reduce=all_reduce, ReduceOp=torch.distributed.ReduceOp),
    )
    _load_functions(
        "relax/utils/data/stream_dataloader.py",
        {"fetch_data_from_transfer_queue", "_prefetch_data_from_transfer_queue"},
        namespace,
    )
    _load_functions(
        "relax/backends/megatron/actor.py",
        {
            "_should_pause_sft_lookahead",
            "_agree_sft_train_prefetch_result",
            "_start_sft_train_prefetch",
            "_take_sft_train_prefetch",
            "_shutdown_sft_train_prefetch",
        },
        namespace,
    )
    actor = namespace["Actor"]()
    actor.args = SimpleNamespace(num_rollout=100, save=None, rotate_ckpt=False, save_interval=None)
    actor._sft_train_prefetch = None
    actor._sft_train_prefetch_rollout_id = None
    actor._sft_train_prefetch_cancel = None
    actor._sft_train_prefetch_executor = ThreadPoolExecutor(max_workers=1)
    try:
        yield SimpleNamespace(actor=actor, namespace=namespace, groups=groups)
    finally:
        actor._shutdown_sft_train_prefetch()


def _submit(runtime, client, *, cancel=None, timeout=2.0, poll_interval=0.005):
    return runtime.actor._sft_train_prefetch_executor.submit(
        runtime.namespace["_prefetch_data_from_transfer_queue"],
        tq_client=client,
        data_fields=["tokens"],
        batch_size=2,
        partition_id="sft_4",
        task_name="sft_train",
        sampling_config={"dp_rank": 3, "task_name": "sft_train"},
        batch_index=0,
        cancel_event=cancel if cancel is not None else Event(),
        timeout=timeout,
        poll_interval=poll_interval,
    )


def test_prefetch_waits_for_publish_and_fetches_before_main_thread_take(runtime):
    ready, empty_seen, fetched = Event(), Event(), Event()
    requests = []
    data_calls = []
    payload = {"tokens": [1, 2]}
    main_thread = current_thread()

    def get_meta(**kwargs):
        assert current_thread() is not main_thread
        requests.append(kwargs)
        if not ready.is_set():
            empty_seen.set()
            return SimpleNamespace(size=0)
        return SimpleNamespace(size=2)

    def get_data(meta):
        assert current_thread() is not main_thread
        data_calls.append(meta)
        fetched.set()
        return payload

    actor = runtime.actor
    actor.data_system_client = SimpleNamespace(get_meta=get_meta, get_data=get_data)
    actor._start_sft_train_prefetch(3, "sft_train", ["tokens"], 2)
    assert empty_seen.wait(1)
    assert not actor._sft_train_prefetch.done()
    ready.set()
    assert fetched.wait(1), "payload should be fetched while the main thread is still training"
    result, rpc_seconds = actor._take_sft_train_prefetch(4)
    assert result[0] is payload
    assert len(data_calls) == 1
    assert len(requests) >= 2
    assert all(request == requests[0] for request in requests)
    assert requests[0]["sampling_config"] == {
        "dp_rank": 3,
        "task_name": "sft_train",
        "batch_index": 0,
        "partition_id": "sft_4",
    }
    assert rpc_seconds >= 0
    assert runtime.groups == ["tp_cp", "pp"]
    assert actor._sft_train_prefetch is None
    assert actor._sft_train_prefetch_cancel is None
    assert actor._sft_train_prefetch_rollout_id is None


def test_empty_prefetch_timeout_finishes_before_fallback(runtime):
    client = SimpleNamespace(get_meta=Mock(return_value=SimpleNamespace(size=0)), get_data=Mock())
    actor = runtime.actor
    actor._sft_train_prefetch = _submit(runtime, client, timeout=0.03)
    future = actor._sft_train_prefetch
    actor._sft_train_prefetch_rollout_id = 4
    assert actor._take_sft_train_prefetch(4) is None
    assert future.done()
    assert isinstance(future.exception(), TimeoutError)
    client.get_data.assert_not_called()


def test_shutdown_interrupts_empty_poll_without_another_rpc(runtime):
    empty_seen = Event()
    cancel = Event()

    def get_meta(**kwargs):
        empty_seen.set()
        return SimpleNamespace(size=0)

    client = SimpleNamespace(get_meta=Mock(side_effect=get_meta), get_data=Mock())
    actor = runtime.actor
    actor._sft_train_prefetch_cancel = cancel
    actor._sft_train_prefetch_rollout_id = 4
    actor._sft_train_prefetch = _submit(runtime, client, cancel=cancel, timeout=10, poll_interval=10)
    future = actor._sft_train_prefetch
    assert empty_seen.wait(1)
    actor._shutdown_sft_train_prefetch()
    actor._shutdown_sft_train_prefetch()
    assert cancel.is_set()
    assert future.done()
    assert isinstance(future.exception(), CancelledError)
    assert client.get_meta.call_count == 1
    client.get_data.assert_not_called()


def test_nonempty_get_data_failure_does_not_resample(runtime):
    client = SimpleNamespace(
        get_meta=Mock(return_value=SimpleNamespace(size=2)),
        get_data=Mock(side_effect=RuntimeError("payload unavailable")),
    )
    future = _submit(runtime, client)
    with pytest.raises(RuntimeError, match="payload unavailable"):
        future.result(timeout=1)
    assert client.get_meta.call_count == client.get_data.call_count == 1


def test_cancel_during_nonempty_meta_keeps_claimed_payload(runtime):
    cancel = Event()
    payload = {"tokens": [1, 2]}

    def get_meta(**kwargs):
        cancel.set()
        return SimpleNamespace(size=2)

    client = SimpleNamespace(get_meta=Mock(side_effect=get_meta), get_data=Mock(return_value=payload))
    result, _ = _submit(runtime, client, cancel=cancel).result(timeout=1)
    assert result[0] is payload
    assert client.get_meta.call_count == client.get_data.call_count == 1


def test_cancelled_prefetch_does_not_claim_metadata(runtime):
    cancel = Event()
    cancel.set()
    client = SimpleNamespace(get_meta=Mock(), get_data=Mock())
    with pytest.raises(CancelledError):
        _submit(runtime, client, cancel=cancel).result(timeout=1)
    client.get_meta.assert_not_called()
    client.get_data.assert_not_called()


@pytest.mark.parametrize("boundary", ["save", "eval", "last"])
def test_prefetch_respects_step_boundaries(runtime, boundary):
    actor = runtime.actor
    actor.data_system_client = SimpleNamespace(get_meta=Mock(), get_data=Mock())
    if boundary == "save":
        actor.args.save, actor.args.save_interval = "checkpoint", 4
    elif boundary == "eval":
        actor.args.eval_at = 4
    else:
        actor.args.num_rollout = 4
    actor._start_sft_train_prefetch(3, "sft_train", ["tokens"], 2)
    assert actor._sft_train_prefetch is None
    actor.data_system_client.get_meta.assert_not_called()


def test_peer_error_still_forces_replica_fallback(runtime):
    def peer_error(tensor, **kwargs):
        tensor[1] = 1

    runtime.namespace["dist"].all_reduce = peer_error
    payload = [{"tokens": [1]}, SimpleNamespace(size=1)], 0.1
    assert runtime.namespace["_agree_sft_train_prefetch_result"](payload, None) is None


def test_stale_running_prefetch_is_cancelled_and_aborts(runtime):
    entered, release = Event(), Event()
    cancel = Event()

    def get_meta(**kwargs):
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(size=0)

    actor = runtime.actor
    actor._sft_train_prefetch_cancel = cancel
    actor._sft_train_prefetch_rollout_id = 3
    actor._sft_train_prefetch = _submit(runtime, SimpleNamespace(get_meta=get_meta), cancel=cancel)
    future = actor._sft_train_prefetch
    try:
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match="Cannot cancel stale"):
            actor._take_sft_train_prefetch(4)
        assert cancel.is_set()
    finally:
        release.set()
    with pytest.raises(CancelledError):
        future.result(timeout=1)


def test_image_prefetch_retries_same_partition_and_honors_cancellation():
    from relax.engine.sft.image_prefetch import SFTImagePrefetch

    fetch = Mock(
        side_effect=[([None, SimpleNamespace(size=0)], 0.1), ([{"tokens": [1]}, SimpleNamespace(size=1)], 0.2)]
    )
    image = SFTImagePrefetch(SimpleNamespace(is_vl_model=False), fetch)
    kwargs = dict(
        model=None,
        data_fields=["tokens"],
        batch_size=1,
        partition_id="sft_4",
        task_name="sft_train",
        sampling_config={"dp_rank": 0},
    )
    result = image.fetch(4, cancel_event=Event(), **kwargs)
    assert result[0][0] == {"tokens": [1]}
    assert result[1] == pytest.approx(0.3)
    assert fetch.call_args_list[0] == fetch.call_args_list[1]
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        image.fetch(5, cancel_event=cancel, **kwargs)
    assert fetch.call_count == 2
