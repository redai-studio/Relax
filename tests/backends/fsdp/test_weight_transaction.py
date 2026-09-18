# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""FSDP weight-sync transaction validation and failure agreement."""

from __future__ import annotations

import faulthandler
import multiprocessing as mp
import os
import socket
import time
from collections.abc import Callable
from datetime import timedelta
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from relax.backends.fsdp.actor import (
    FSDPTrainRayActor,
    WeightSyncError,
    _collect_weight_sync_errors,
    _raise_weight_sync_errors,
)
from relax.models.generative import FullWeightManifest


class _Ref:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error


class _Remote:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        try:
            return _Ref(value=self.fn(*args, **kwargs))
        except Exception as exc:
            return _Ref(error=exc)


class _Engine:
    def __init__(self, checksum, events, index, checksum_error=None):
        self.checksum = checksum
        self.events = events
        self.index = index
        self.checksum_error = checksum_error
        self.get_weights_checksum = _Remote(self._checksum)
        self.commit_weight_version = _Remote(self._commit)

    def _checksum(self, _modules):
        self.events.append(("checksum", self.index))
        if self.checksum_error is not None:
            raise self.checksum_error
        return self.checksum

    def _commit(self, version, digest):
        self.events.append(("commit", self.index, version, digest))
        return {"active_version": version, "weight_manifest_sha256": digest}


def _fake_ray_get(value):
    def _resolve(ref):
        if ref.error is not None:
            raise ref.error
        return ref.value

    return [_resolve(ref) for ref in value] if isinstance(value, list) else _resolve(value)


def _manifest():
    return FullWeightManifest(1, "qwen_image", "t2i", 3, "base", 2, 8, "bf16", 16, "shape-hash")


def _shell():
    shell = object.__new__(FSDPTrainRayActor)
    shell.args = SimpleNamespace(fsdp_trainable_attr="transformer")
    return shell


def test_all_engines_are_verified_before_any_commit(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", _fake_ray_get)
    events = []
    checksum = {"transformer": "a" * 64, "tensor_count": 2}
    engines = [_Engine(checksum, events, index) for index in range(2)]

    FSDPTrainRayActor._verify_and_commit_engines(_shell(), engines, _manifest())

    assert events[:2] == [("checksum", 0), ("checksum", 1)]
    assert [event[:2] for event in events[2:]] == [("commit", 0), ("commit", 1)]
    assert all(event[3] == _manifest().sha256() for event in events[2:])


@pytest.mark.parametrize(
    "second_checksum",
    [
        {"transformer": "a" * 64},
        {"transformer": "a" * 64, "tensor_count": 1},
        {"transformer": "b" * 64, "tensor_count": 2},
        {"transformer": "not_found", "tensor_count": 2},
    ],
)
def test_secondary_engine_mismatch_prevents_all_commits(monkeypatch, second_checksum):
    import ray

    monkeypatch.setattr(ray, "get", _fake_ray_get)
    events = []
    engines = [
        _Engine({"transformer": "a" * 64, "tensor_count": 2}, events, 0),
        _Engine(second_checksum, events, 1),
    ]

    with pytest.raises(WeightSyncError):
        FSDPTrainRayActor._verify_and_commit_engines(_shell(), engines, _manifest())
    assert not any(event[0] == "commit" for event in events)


def test_secondary_engine_checksum_rpc_failure_prevents_all_commits(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "get", _fake_ray_get)
    events = []
    engines = [
        _Engine({"transformer": "a" * 64, "tensor_count": 2}, events, 0),
        _Engine({}, events, 1, checksum_error=RuntimeError("RPC failed")),
    ]

    with pytest.raises(WeightSyncError, match="checksum RPC failed.*RPC failed"):
        FSDPTrainRayActor._verify_and_commit_engines(_shell(), engines, _manifest())
    assert not any(event[0] == "commit" for event in events)


class _AdapterEngine:
    def __init__(self, adapted_layers, events, index):
        self.adapted_layers = adapted_layers
        self.events = events
        self.index = index
        self.set_lora_from_tensors = _Remote(self._set)
        self.commit_weight_version = _Remote(self._commit)

    def _set(self, _payload, **_kwargs):
        self.events.append(("set", self.index))
        return {"success": True, "adapted_layers": self.adapted_layers}

    def _commit(self, version, digest):
        self.events.append(("commit", self.index))
        return {"active_version": version, "weight_manifest_sha256": digest}


def _adapter_shell():
    shell = object.__new__(FSDPTrainRayActor)
    shell.args = SimpleNamespace(
        weight_sync_wire_dtype="fp32",
        lora_alpha=4,
        fsdp_trainable_attr="transformer",
    )
    shell.device = torch.device("cpu")
    shell._named_sync_params = lambda: []
    shell._rank0_then_agree = lambda work, **_kwargs: work()
    return shell


@pytest.mark.parametrize("adapted_layers", [1, 2])
def test_adapter_transaction_requires_every_engine_layer_before_commit(monkeypatch, adapted_layers):
    import ray

    from relax.backends.fsdp import weight_update

    monkeypatch.setattr(ray, "get", _fake_ray_get)
    gathered = [
        ("block.lora_A.default.weight", torch.randn(2, 4)),
        ("block.lora_B.default.weight", torch.randn(4, 2)),
    ]
    monkeypatch.setattr(weight_update, "iter_full_named_tensors", lambda *_args, **_kwargs: iter(gathered))
    events = []
    engines = [_AdapterEngine(1, events, 0), _AdapterEngine(adapted_layers, events, 1)]

    if adapted_layers == 1:
        FSDPTrainRayActor._run_adapter_transaction(_adapter_shell(), _manifest(), engines)
        assert [event[0] for event in events] == ["set", "set", "commit", "commit"]
    else:
        with pytest.raises(WeightSyncError, match="incomplete LoRA payload"):
            FSDPTrainRayActor._run_adapter_transaction(_adapter_shell(), _manifest(), engines)
        assert not any(event[0] == "commit" for event in events)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _failure_worker_entry(
    rank: int,
    port: int,
    queue: Queue,
    ready: Event,
    start: Event,
    worker: Callable[..., None],
    worker_args: tuple[str, ...],
    startup_timeout: float,
    run_timeout: float,
) -> None:
    # Spawn reimports this module and its optional Megatron/modelopt dependencies
    # before entering here. Do not charge those imports to the hang watchdog.
    ready.set()
    if not start.wait(timeout=startup_timeout):
        raise TimeoutError(f"rank {rank} did not receive the test start signal")
    faulthandler.dump_traceback_later(max(run_timeout - 5, run_timeout / 2))
    try:
        worker(rank, port, *worker_args, queue)
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_failure_workers(
    worker: Callable[..., None],
    *worker_args: str,
    startup_timeout: float = 120,
    run_timeout: float = 20,
) -> list[tuple[int, str]]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    start = ctx.Event()
    ready = [ctx.Event() for _ in range(2)]
    port = _free_port()
    processes = [
        ctx.Process(
            target=_failure_worker_entry,
            args=(rank, port, queue, ready[rank], start, worker, worker_args, startup_timeout, run_timeout),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + startup_timeout
        for rank, (event, process) in enumerate(zip(ready, processes)):
            while not event.wait(timeout=min(0.1, max(0, deadline - time.monotonic()))):
                if process.exitcode is not None:
                    pytest.fail(f"rank {rank} exited during spawn/import with exit code {process.exitcode}")
                if time.monotonic() >= deadline:
                    pytest.fail(f"rank {rank} did not finish spawn/import within {startup_timeout}s")

        # Keep the strict execution deadline, but start it only once both ranks
        # have imported their dependencies and can enter the collective test.
        deadline = time.monotonic() + run_timeout
        start.set()
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        hung_ranks = [rank for rank, process in enumerate(processes) if process.is_alive()]
        if hung_ranks:
            pytest.fail(
                f"{worker.__name__}{worker_args}: ranks {hung_ranks} hung during execution/exit ({run_timeout}s)"
            )
        assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]
        results = sorted(queue.get(timeout=2) for _ in processes)
        assert [rank for rank, _ in results] == [0, 1]
        return results
    finally:
        # Clean up every sibling even if startup, an assertion, or a join fails.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                if not process.is_alive():
                    process.close()
        queue.close()
        queue.join_thread()


def _serialization_failure_worker(rank: int, port: int, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=10))
    from relax.utils import distributed_utils

    distributed_utils.GLOO_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
    try:
        local_error = "RuntimeError: injected serializer failure" if rank == 1 else None
        errors = _collect_weight_sync_errors(local_error)
        _raise_weight_sync_errors(4, "IPC tensor preparation/serialization", errors)
        dist.gather_object(rank, [None, None] if rank == 0 else None, dst=0, group=distributed_utils.GLOO_GROUP)
        queue.put((rank, "unexpected-success"))
    except WeightSyncError as exc:
        queue.put((rank, str(exc)))
    finally:
        dist.destroy_process_group()


def test_non_src_serialization_failure_exits_all_ranks_without_hang():
    results = _run_failure_workers(_serialization_failure_worker)
    assert all("injected serializer failure" in message for _, message in results)


def _local_phase_failure_worker(rank: int, port: int, phase: str, queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=10))
    from relax.utils import distributed_utils

    distributed_utils.GLOO_GROUP = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
    shell = object.__new__(FSDPTrainRayActor)
    shell._rank = rank
    try:
        if phase == "model_load":

            class _Adapter:
                def load_train_model(self, _args):
                    if rank == 1:
                        raise OSError("injected model load failure")
                    return object()

            shell.adapter = _Adapter()
            FSDPTrainRayActor._load_train_model_waved(shell, SimpleNamespace(fsdp_load_wave_size=1))
        else:
            shell.args = SimpleNamespace()
            shell.adapter = object()

            def _read(_rollout_id):
                raise OSError("injected TQ read failure")

            shell._read_train_partition = _read
            FSDPTrainRayActor._load_train_batches(shell, 3, None)
        queue.put((rank, "unexpected-success"))
    except Exception as exc:
        queue.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("phase, message", [("model_load", "model load failure"), ("tq_read", "TQ read failure")])
def test_rank_local_precollective_failure_exits_all_ranks_without_hang(phase, message):
    results = _run_failure_workers(_local_phase_failure_worker, phase)
    assert all(message in result for _, result in results)
