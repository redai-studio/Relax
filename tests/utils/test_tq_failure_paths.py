# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CI-safe timeout, retry, disconnect, fallback, and cleanup contracts."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from relax.utils.tq import lifecycle as tq_lifecycle


def _has_real_submodule(dotted: str) -> bool:
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, ValueError, TypeError):
        return False


_REAL_MOONCAKE_CLIENT = _has_real_submodule("transfer_queue.storage.clients.mooncake_client")


def _raise(error: BaseException) -> None:
    raise error


@pytest.mark.parametrize(
    ("state", "expected", "raises"),
    [
        ("absent", False, None),
        ("healthy", None, RuntimeError),
        ("half", True, None),
        ("timeout", True, None),
        ("dead", True, None),
        ("unknown-error", None, RuntimeError),
    ],
)
def test_reaper_state_matrix(
    monkeypatch: pytest.MonkeyPatch, state: str, expected: bool | None, raises: type[BaseException] | None
) -> None:
    actor = MagicMock()
    killed: list[str] = []
    fake_ray = MagicMock()
    fake_ray.exceptions = tq_lifecycle.ray.exceptions
    if state == "absent":
        fake_ray.get_actor.side_effect = ValueError("missing")
    else:
        fake_ray.get_actor.return_value = actor
    if state == "healthy":
        fake_ray.get.return_value = {"backend": {}}
    elif state == "half":
        fake_ray.get.return_value = None
    elif state == "timeout":
        fake_ray.get.side_effect = fake_ray.exceptions.GetTimeoutError("private")
    elif state == "dead":
        fake_ray.get.side_effect = fake_ray.exceptions.RayActorError(error_msg="private")
    elif state == "unknown-error":
        fake_ray.get.side_effect = fake_ray.exceptions.RayError("private endpoint")
    monkeypatch.setattr(tq_lifecycle, "ray", fake_ray)
    monkeypatch.setattr(tq_lifecycle, "kill_tq_controller_and_wait", lambda **_kwargs: killed.append("killed"))
    context = pytest.raises(raises) if raises else nullcontext()
    with context:
        assert tq_lifecycle.reap_unusable_tq_controller() is expected
    assert killed == (["killed"] if state in {"half", "timeout", "dead"} else [])


def test_controller_cleanup_timeout_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = object()
    killed: list[Any] = []
    monkeypatch.setattr(tq_lifecycle.ray, "get_actor", lambda *_args, **_kwargs: controller)
    monkeypatch.setattr(tq_lifecycle.ray, "kill", lambda handle: killed.append(handle))
    with pytest.raises(tq_lifecycle.TqCleanupTimeout, match="still resolvable"):
        tq_lifecycle.kill_tq_controller_and_wait(timeout=0)
    assert killed == [controller]


@pytest.mark.parametrize(
    ("has_store", "close_error", "expected"),
    [(True, None, ["tq", "store"]), (True, RuntimeError("close failed"), ["tq", "store"]), (False, None, ["tq"])],
)
def test_close_unmount_matrix(
    monkeypatch: pytest.MonkeyPatch,
    has_store: bool,
    close_error: BaseException | None,
    expected: list[str],
) -> None:
    events: list[str] = []
    store = MagicMock()
    manager = SimpleNamespace(storage_client=store) if has_store else SimpleNamespace()
    fake_tq = MagicMock()
    fake_tq.get_client.return_value = MagicMock(storage_manager=manager)

    def close() -> None:
        events.append("tq")
        if close_error:
            raise close_error

    fake_tq.close.side_effect = close
    store.close.side_effect = lambda: events.append("store")
    monkeypatch.setattr(tq_lifecycle, "tq", fake_tq)
    context = pytest.raises(RuntimeError, match="close failed") if close_error else nullcontext()
    with context:
        tq_lifecycle.close_tq_and_unmount()
    assert events == expected


@pytest.mark.parametrize("value", ["soon", "nan", "inf", "-inf"])
def test_attach_timeout_rejects_unusable_env(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RELAX_TQ_ATTACH_TIMEOUT_SECONDS", value)
    with pytest.raises(RuntimeError, match="finite positive") as excinfo:
        tq_lifecycle._resolve_attach_timeout()
    assert value not in str(excinfo.value)


def test_bounded_init_timeout_error_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAX_TQ_ATTACH_TIMEOUT_SECONDS", "12.5")
    assert tq_lifecycle._resolve_attach_timeout() == 12.5
    monkeypatch.setattr(tq_lifecycle.tq, "init", lambda **_kwargs: time.sleep(5))
    with pytest.raises(tq_lifecycle.TqAttachTimeout, match="did not finish"):
        tq_lifecycle._bounded_tq_init({}, time.monotonic() + 0.2, role="test")
    monkeypatch.setattr(tq_lifecycle.tq, "init", lambda **_kwargs: _raise(ValueError("bad conf")))
    with pytest.raises(ValueError, match="bad conf"):
        tq_lifecycle._bounded_tq_init({}, time.monotonic() + 5, role="test")
    monkeypatch.setattr(tq_lifecycle.ray, "get_actor", lambda *_args, **_kwargs: _raise(ValueError("missing")))
    with pytest.raises(tq_lifecycle.TqAttachTimeout, match="attach timed out"):
        tq_lifecycle._await_controller_config(time.monotonic() + 0.2)


class _HandshakeHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, node_ids: list[str] | None = None) -> None:
        self.args: list[tuple[Any, ...]] = []
        self.workers: list[Any] = []
        self.options: dict[str, Any] = {}
        self.strategies: list[Any] = []

        class Task:
            def options(task_self, **options: Any) -> "Task":
                self.strategies.append(options["scheduling_strategy"])
                return task_self

            def remote(task_self, *args: Any) -> object:
                self.args.append(args)
                return object()

        def remote(**options: Any):
            self.options.update(options)

            def decorate(function: Any) -> Task:
                self.workers.append(function)
                return Task()

            return decorate

        monkeypatch.setattr(tq_lifecycle.ray, "remote", remote)
        monkeypatch.setattr(tq_lifecycle, "_alive_node_ids", lambda: node_ids or ["a" * 56])
        monkeypatch.setattr(tq_lifecycle.ray, "wait", lambda refs, **_kwargs: (list(refs), []))
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda _ref: None)


def test_cluster_attach_uses_every_node_hard_affinity_and_one_shot_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    nodes = ["a" * 56, "b" * 56]
    harness = _HandshakeHarness(monkeypatch, nodes)
    assert tq_lifecycle.verify_cluster_attach({}, timeout=0.1) == []
    assert (harness.options["max_calls"], harness.options["max_retries"]) == (1, 0)
    assert [strategy.node_id for strategy in harness.strategies] == nodes
    assert all(strategy.soft is False for strategy in harness.strategies)


@pytest.mark.parametrize(
    ("backend", "verify_mooncake", "assert_error"),
    [
        ("MooncakeStore", True, None),
        ("MooncakeStore", True, RuntimeError("protocol=tcp")),
        ("SimpleStorage", False, None),
    ],
)
def test_handshake_worker_verifies_backend_and_always_detaches(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    verify_mooncake: bool,
    assert_error: BaseException | None,
) -> None:
    harness = _HandshakeHarness(monkeypatch)
    conf = {"backend": {"storage_backend": backend}}
    assert tq_lifecycle.verify_cluster_attach(conf, timeout=0.1) == []
    assert harness.args[0][1] is verify_mooncake
    events: list[str] = []
    monkeypatch.setattr(tq_lifecycle, "attach_tq_client", lambda *_args, **_kwargs: events.append("attach"))
    monkeypatch.setattr(
        tq_lifecycle,
        "assert_mooncake_rdma_configured",
        lambda: events.append("assert") or (_raise(assert_error) if assert_error else None),
    )
    monkeypatch.setattr(tq_lifecycle, "detach_tq_client", lambda: events.append("detach"))
    context = pytest.raises(RuntimeError, match="protocol=tcp") if assert_error else nullcontext()
    with context:
        harness.workers[0](conf, verify_mooncake, 0.1)
    assert events == ["attach", *(["assert"] if verify_mooncake else []), "detach"]


def test_handshake_scheduling_failure_cancels_submitted_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    first_ref = object()
    cancelled: list[object] = []

    class Task:
        calls = 0

        def options(self, **_kwargs: Any) -> "Task":
            return self

        def remote(self, *_args: Any) -> object:
            self.calls += 1
            if self.calls == 1:
                return first_ref
            raise RuntimeError("private scheduling detail")

    monkeypatch.setattr(tq_lifecycle.ray, "remote", lambda **_kwargs: lambda _function: Task())
    monkeypatch.setattr(tq_lifecycle, "_alive_node_ids", lambda: ["a" * 56, "b" * 56])
    monkeypatch.setattr(tq_lifecycle.ray, "cancel", lambda ref, force: cancelled.append(ref))
    monkeypatch.setattr(tq_lifecycle.ray, "wait", lambda refs, **_kwargs: (list(refs), []))
    assert tq_lifecycle.verify_cluster_attach({}, timeout=0.1) == [
        "cluster: handshake scheduling failed (RuntimeError)"
    ]
    assert cancelled == [first_ref]


@pytest.mark.parametrize("cancel_fails", [False, True])
def test_unconfirmed_handshake_worker_aborts_fallback(monkeypatch: pytest.MonkeyPatch, cancel_fails: bool) -> None:
    _HandshakeHarness(monkeypatch)
    monkeypatch.setattr(tq_lifecycle.ray, "wait", lambda refs, **_kwargs: ([], list(refs)))
    if cancel_fails:
        monkeypatch.setattr(tq_lifecycle.ray, "cancel", lambda *_args, **_kwargs: _raise(RuntimeError("private")))
    else:
        monkeypatch.setattr(tq_lifecycle.ray, "cancel", lambda *_args, **_kwargs: None)
    with pytest.raises(tq_lifecycle.TqHandshakeIsolationError, match="could not be confirmed stopped"):
        tq_lifecycle.verify_cluster_attach({}, timeout=0.1)


class MooncakeStorageManager:
    def __init__(self, storage_client: Any) -> None:
        self.storage_client = storage_client


@pytest.mark.parametrize(
    ("manager", "match"),
    [
        (MooncakeStorageManager(MagicMock(protocol="rdma")), None),
        (MagicMock(), "not MooncakeStorageManager"),
        (MooncakeStorageManager(None), "no storage_client"),
        *[(MooncakeStorageManager(MagicMock(protocol=value)), "not configured") for value in ("tcp", None, "")],
    ],
)
def test_configured_mooncake_manager_matrix(monkeypatch: pytest.MonkeyPatch, manager: Any, match: str | None) -> None:
    fake_tq = MagicMock()
    fake_tq.get_client.return_value = MagicMock(storage_manager=manager)
    monkeypatch.setattr(tq_lifecycle, "tq", fake_tq)
    context = pytest.raises(RuntimeError, match=match) if match else nullcontext()
    with context:
        tq_lifecycle.assert_mooncake_rdma_configured()


def test_attach_timeout_leaves_no_reusable_process_global_state() -> None:
    env = os.environ.copy()
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    env.pop("RAY_ADDRESS", None)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with tempfile.TemporaryDirectory(prefix="tq-ray-") as probe_dir:
        result = subprocess.run(
            [sys.executable, "-m", "tests.utils._tq_handshake_timeout_probe", probe_dir],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    assert result.returncode == 0, f"probe stdout:\n{result.stdout}\nprobe stderr:\n{result.stderr}"


def test_worker_detach_closes_clients_resets_handles_and_is_used_by_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_client = MagicMock()
    client = MagicMock(storage_manager=SimpleNamespace(storage_client=storage_client))
    fake_tq = MagicMock()
    fake_tq.get_client.return_value = client
    interface = SimpleNamespace(_TQ_CLIENT=client, _TQ_CONTROLLER=object())
    monkeypatch.setattr(tq_lifecycle.tq, "interface", interface, raising=False)
    monkeypatch.setattr(tq_lifecycle, "tq", fake_tq)
    tq_lifecycle.detach_tq_client()
    storage_client.close.assert_called_once_with()
    client.close.assert_called_once_with()
    assert interface._TQ_CLIENT is None and interface._TQ_CONTROLLER is None

    from relax.components.base import Base

    calls: list[bool] = []
    monkeypatch.setattr(tq_lifecycle, "detach_tq_client", lambda: calls.append(True))
    component = Base()
    component.data_system_client = object()
    component.__del__()
    assert calls == [True] and component.data_system_client is None


def _conf(backend: str) -> dict[str, Any]:
    return {"controller": {}, "backend": {"storage_backend": backend}}


class _InitHarness:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        init_effects: list[Any],
        *,
        start_effect: Any = None,
        reap_effects: list[BaseException | None] | None = None,
    ) -> None:
        self.events: list[str] = []
        effects = iter(init_effects)
        reaps = iter(reap_effects or [])

        def reap() -> None:
            self.events.append("reap")
            effect = next(reaps, None)
            if effect:
                raise effect

        def start(*, timeout: float) -> str:
            self.events.append("start")
            if isinstance(start_effect, BaseException):
                raise start_effect
            return f"owner-{self.events.count('start')}"

        def initialize(owner: Any, conf: dict[str, Any], *, timeout: float) -> Any:
            backend = conf["backend"]["storage_backend"]
            self.events.append(f"init:{backend}")
            effect = next(effects)
            if isinstance(effect, BaseException):
                raise effect
            return tq_lifecycle.TqInitResult(config=conf, owner=owner)

        monkeypatch.setattr(tq_lifecycle, "reap_unusable_tq_controller", reap)
        monkeypatch.setattr(tq_lifecycle, "_start_owner", start)
        monkeypatch.setattr(tq_lifecycle, "_initialize_owner", initialize)


@pytest.mark.parametrize(
    ("mode", "init_effects", "start_effect", "reap_effects", "expected_events", "error"),
    [
        ("off", [None], None, None, ["reap", "start", "init:SimpleStorage"], None),
        (
            "auto",
            [tq_lifecycle.TqInitializationError("failed"), None],
            None,
            None,
            ["reap", "start", "init:MooncakeStore", "reap", "start", "init:SimpleStorage"],
            None,
        ),
        (
            "required",
            [tq_lifecycle.TqInitializationError("failed")],
            None,
            None,
            ["reap", "start", "init:MooncakeStore"],
            tq_lifecycle.TqInitializationError,
        ),
        (
            "auto",
            [tq_lifecycle.TqCleanupTimeout("pending")],
            None,
            None,
            ["reap", "start", "init:MooncakeStore"],
            tq_lifecycle.TqCleanupTimeout,
        ),
        (
            "auto",
            [tq_lifecycle.TqInitializationError("failed")],
            None,
            [None, RuntimeError("dirty")],
            ["reap", "start", "init:MooncakeStore", "reap"],
            RuntimeError,
        ),
        ("auto", [], RuntimeError("schedule failed"), None, ["reap", "start"], RuntimeError),
    ],
)
def test_initialize_and_fallback_transaction_matrix(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    init_effects: list[Any],
    start_effect: Any,
    reap_effects: list[BaseException | None] | None,
    expected_events: list[str],
    error: type[BaseException] | None,
) -> None:
    harness = _InitHarness(
        monkeypatch,
        init_effects,
        start_effect=start_effect,
        reap_effects=reap_effects,
    )
    primary = _conf("SimpleStorage" if mode == "off" else "MooncakeStore")
    context = pytest.raises(error) if error else nullcontext()
    with context:
        result = tq_lifecycle.initialize_tq_with_fallback(primary, mode=mode, fallback_conf=_conf("SimpleStorage"))
        assert result.owner is not None
        if mode == "auto" and len(init_effects) == 2:
            assert result.config["backend"]["storage_backend"] == "SimpleStorage"
    assert harness.events == expected_events


class _RemoteMethod:
    def __init__(self, value: str) -> None:
        self.value = value

    def remote(self, *_args: Any, **_kwargs: Any) -> str:
        return self.value


class _FakeOwner:
    def __init__(self) -> None:
        self.ready = _RemoteMethod("ready")
        self.initialize = _RemoteMethod("initialize")
        self.close = _RemoteMethod("close")
        self.termination_probe = _RemoteMethod("probe")


@pytest.mark.parametrize("outcome", ["success", "timeout"])
def test_owner_start_boundary(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    owner = _FakeOwner()
    stopped: list[Any] = []
    monkeypatch.setattr(tq_lifecycle._TransferQueueOwner, "remote", lambda: owner)
    if outcome == "timeout":
        monkeypatch.setattr(
            tq_lifecycle.ray,
            "get",
            lambda *_args, **_kwargs: _raise(tq_lifecycle.ray.exceptions.GetTimeoutError("timeout")),
        )
        monkeypatch.setattr(tq_lifecycle, "_stop_owner_actor", lambda handle, **_kwargs: stopped.append(handle))
        with pytest.raises(RuntimeError, match="owner creation failed"):
            tq_lifecycle._start_owner(timeout=0.1)
        assert stopped == [owner]
    else:
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda *_args, **_kwargs: None)
        assert tq_lifecycle._start_owner(timeout=0.1) is owner


def test_owner_initialize_success_timeout_and_exception_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = _FakeOwner()
    stored = _conf("SimpleStorage")
    monkeypatch.setattr(tq_lifecycle.ray, "get", lambda *_args, **_kwargs: stored)
    result = tq_lifecycle._initialize_owner(owner, stored, timeout=0.1)
    assert result.config is stored and result.owner is owner

    cleaned: list[Any] = []
    monkeypatch.setattr(tq_lifecycle, "_cleanup_failed_owner", lambda handle: cleaned.append(handle))
    for source, expected in (
        (tq_lifecycle.ray.exceptions.GetTimeoutError("timeout"), tq_lifecycle.TqInitializationTimeout),
        (RuntimeError("private detail"), tq_lifecycle.TqInitializationError),
    ):
        monkeypatch.setattr(tq_lifecycle.ray, "get", lambda *_args, error=source, **_kwargs: _raise(error))
        with pytest.raises(expected):
            tq_lifecycle._initialize_owner(owner, {}, timeout=0.1)
    assert cleaned == [owner, owner]


@pytest.mark.parametrize(
    ("ready", "ray_get", "error"),
    [
        (True, lambda _ref: _raise(tq_lifecycle.ray.exceptions.RayActorError(error_msg="dead")), None),
        (False, lambda _ref: None, tq_lifecycle.TqCleanupTimeout),
        (True, lambda _ref: None, tq_lifecycle.TqCleanupTimeout),
    ],
)
def test_owner_stop_requires_terminal_actor_failure(
    monkeypatch: pytest.MonkeyPatch,
    ready: bool,
    ray_get: Any,
    error: type[BaseException] | None,
) -> None:
    owner = _FakeOwner()
    monkeypatch.setattr(tq_lifecycle.ray, "kill", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tq_lifecycle.ray, "wait", lambda refs, **_kwargs: (list(refs), []) if ready else ([], list(refs))
    )
    monkeypatch.setattr(tq_lifecycle.ray, "get", ray_get)
    context = pytest.raises(error) if error else nullcontext()
    with context:
        tq_lifecycle._stop_owner_actor(owner, timeout=0.1)


def test_owner_cleanup_order_and_fail_closed_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = _FakeOwner()
    events: list[str] = []
    monkeypatch.setattr(tq_lifecycle.ray, "get", lambda *_args, **_kwargs: events.append("close"))
    monkeypatch.setattr(tq_lifecycle, "_stop_owner_actor", lambda *_args, **_kwargs: events.append("terminal"))
    monkeypatch.setattr(tq_lifecycle, "kill_tq_controller_and_wait", lambda **_kwargs: events.append("reap"))
    tq_lifecycle._cleanup_failed_owner(owner)
    assert events == ["close", "terminal", "reap"]

    monkeypatch.setattr(
        tq_lifecycle,
        "_stop_owner_actor",
        lambda *_args, **_kwargs: _raise(tq_lifecycle.TqCleanupTimeout("pending")),
    )
    with pytest.raises(tq_lifecycle.TqCleanupTimeout, match="pending"):
        tq_lifecycle._cleanup_failed_owner(owner)
    assert events == ["close", "terminal", "reap", "close"]


class _FlakyStore:
    def __init__(self, fail_times: int, error: Exception | None = None) -> None:
        self.fail_times = fail_times
        self.error = error
        self.calls: list[list[str]] = []

    def batch_get_into(self, keys: list[str], _ptrs: list[int], _sizes: list[int]) -> list[int]:
        self.calls.append(list(keys))
        if self.error:
            raise self.error
        if self.fail_times:
            self.fail_times -= 1
            return [-800] * len(keys)
        return [0] * len(keys)


def _client_with_store(store: _FlakyStore) -> Any:
    from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient

    client = object.__new__(MooncakeStoreClient)
    client._store = store
    client.replica_config = None
    return client


@pytest.mark.skipif(not _REAL_MOONCAKE_CLIENT, reason="requires real TransferQueue Mooncake client")
@pytest.mark.parametrize("disconnect", [False, True], ids=["retry", "disconnect"])
def test_retry_and_disconnect_surface(monkeypatch: pytest.MonkeyPatch, disconnect: bool) -> None:
    monkeypatch.setattr("transfer_queue.storage.clients.mooncake_client.RETRY_DELAY_SECONDS", 0)
    error = RuntimeError("Failed to open segment for endpoint='<address>'") if disconnect else None
    store = _FlakyStore(0 if disconnect else 2, error)
    client = _client_with_store(store)
    context = pytest.raises(RuntimeError, match="Failed to open segment") if disconnect else nullcontext()
    with context:
        client._batch_get_into_with_retry(["0@f0", "1@f0"], [1, 2], [8, 8])
    assert len(store.calls) == (1 if disconnect else 3)


class _InlineExecutorLoop:
    def run_in_executor(self, _executor: Any, function: Any, *args: Any) -> asyncio.Future[Any]:
        future = asyncio.get_running_loop().create_future()
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)
        return future


@pytest.mark.skipif(not _REAL_MOONCAKE_CLIENT, reason="requires real TransferQueue Mooncake manager")
@pytest.mark.asyncio
@pytest.mark.parametrize("put_fails", [False, True], ids=["success", "failure"])
async def test_production_status_follows_storage_success(monkeypatch: pytest.MonkeyPatch, put_fails: bool) -> None:
    from tensordict import TensorDict
    from transfer_queue.storage.managers.mooncake_manager import MooncakeStorageManager

    monkeypatch.setattr(asyncio, "get_event_loop", lambda: _InlineExecutorLoop())
    calls: list[str] = []
    storage = MagicMock()
    if put_fails:
        storage.put.side_effect = RuntimeError("capacity exhausted")
    else:
        storage.put.side_effect = lambda *_args: calls.append("put")
    manager = object.__new__(MooncakeStorageManager)
    manager.storage_client = storage
    manager.notify_data_update = AsyncMock(side_effect=lambda *_args, **_kwargs: calls.append("notify"))
    manager.controller_handshake_socket = None
    manager.storage_manager_id = "contract-test"
    manager.zmq_context = MagicMock()
    meta = MagicMock(global_indexes=[7], partition_ids=["capacity"], _custom_backend_meta=[{}])
    meta.get_all_custom_meta.return_value = [{}]
    context = pytest.raises(RuntimeError, match="capacity exhausted") if put_fails else nullcontext()
    with context:
        await manager.put_data(TensorDict({"pixel_values": torch.randn(1, 16)}, batch_size=[1]), meta)
    assert calls == ([] if put_fails else ["put", "notify"])
