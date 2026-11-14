# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Control-plane regressions; engine and device execution are tested
separately."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from relax.distributed.weight_sync.admission import AdmissionGate, AdmissionMiddleware, InstallationFence
from relax.distributed.weight_sync.budget import NodeBudget
from relax.distributed.weight_sync.config import PublicationConfig
from relax.distributed.weight_sync.deadline import operation_remaining, remaining, time_budget
from relax.distributed.weight_sync.parallel_io import ordered_map
from relax.distributed.weight_sync.sglang_loader import EmbeddedSGLangGroup
from relax.distributed.weight_sync.sglang_plugin import create_engine, register
from relax.distributed.weight_sync.source_budget import source_demand
from relax.distributed.weight_sync.store import VersionStore
from relax.distributed.weight_sync.transport import TCPTransport


pytest_plugins = ["tests.cpu.backends.sglang.test_router_registration"]


def test_operation_defaults_do_not_override_configured_deadlines(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("relax.distributed.weight_sync.deadline.time.monotonic", lambda: clock[0])
    assert operation_remaining(120) == 120
    with time_budget(1800):
        clock[0] += 200
        assert operation_remaining(120) == 1600
        assert remaining(5) == 5  # Explicit transport/cleanup caps still apply.
        with time_budget(20):
            assert operation_remaining(120) == 20
            clock[0] += 21
            with pytest.raises(TimeoutError):
                operation_remaining(120)


def test_worker_command_receives_operation_budget(tmp_path):
    group = EmbeddedSGLangGroup.__new__(EmbeddedSGLangGroup)
    group._rpc_failed = False
    group.tp_size = group.pp_size = 1
    group.audit_directory = tmp_path
    requests = []

    def send(payload):
        request = json.loads(payload)
        requests.append(request)
        (Path(request["receipts"]) / "rank-0.json").write_text(
            json.dumps(
                {
                    "request_id": request["request_id"],
                    "success": True,
                    "rank": 0,
                    "identity": {"instance": "worker"},
                }
            )
        )

    group._send_rpc = send
    with time_budget(1800):
        assert group._rpc("identify") == {"0": {"instance": "worker"}}
    assert 1700 < requests[0]["timeout"] <= 1800


@pytest.mark.parametrize(
    "path",
    [
        "/pause_generation",
        "/continue_generation",
        "/load_lora_adapter",
        "/unload_lora_adapter",
    ],
)
@pytest.mark.parametrize("state", ["unclaimed", "active", "installing"])
def test_native_controls_preserve_engine_behavior_and_only_wait_for_installation(path, state):
    async def run():
        gate, fence = AdmissionGate(), InstallationFence()
        if state != "unclaimed":
            fence.claim({"epoch": 1, "workers": {"0": "w"}})
        if state == "active":
            await gate.open()
        called, messages = [], []

        async def app(scope, receive, send):
            called.append(scope["path"])
            await send({"type": "http.response.start", "status": 200})

        async def send(message):
            messages.append(message)

        middleware = AdmissionMiddleware(app, gate, fence, asyncio.Lock())
        await middleware({"type": "http", "path": path}, None, send)
        if state == "installing":
            assert not called
            assert messages[0]["status"] == 409
            assert b"after activation" in messages[1]["body"]
        else:
            assert called == [path]
            assert messages[0]["status"] == 200

    asyncio.run(run())


@pytest.mark.parametrize("claimed", [False, True])
def test_native_weight_mutation_is_guarded_only_after_version_management_begins(claimed):
    async def run():
        gate, fence = AdmissionGate(), InstallationFence()
        if claimed:
            fence.claim({"epoch": 1, "workers": {"0": "w"}})
        messages = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})

        async def send(message):
            messages.append(message)

        await AdmissionMiddleware(app, gate, fence, asyncio.Lock())(
            {"type": "http", "path": "/update_weights_from_disk"},
            None,
            send,
        )
        assert messages[0]["status"] == (409 if claimed else 200)

    asyncio.run(run())


@pytest.mark.parametrize("failure", [False, True])
def test_plugin_activation_merges_and_restores_existing_environment(monkeypatch, failure):
    monkeypatch.setattr(
        "importlib.metadata.entry_points", lambda **kw: [SimpleNamespace(name="relax_weight_versions")]
    )
    monkeypatch.setenv("SGLANG_PLUGINS", "existing_plugin")
    monkeypatch.setenv("RELAX_WEIGHT_VERSION_PLUGIN", "original")
    module = ModuleType("sglang")

    def engine(**options):
        assert os.environ["SGLANG_PLUGINS"] == "existing_plugin,relax_weight_versions"
        assert os.environ["RELAX_WEIGHT_VERSION_PLUGIN"] == "1"
        if failure:
            raise RuntimeError("engine startup")
        return "engine"

    module.Engine = engine
    monkeypatch.setitem(sys.modules, "sglang", module)
    if failure:
        with pytest.raises(RuntimeError, match="engine startup"):
            create_engine()
    else:
        assert create_engine() == "engine"
    assert os.environ["SGLANG_PLUGINS"] == "existing_plugin"
    assert os.environ["RELAX_WEIGHT_VERSION_PLUGIN"] == "original"


def test_registration_uses_actual_scheduler_interface_not_distribution_label(monkeypatch):
    monkeypatch.setenv("RELAX_WEIGHT_VERSION_PLUGIN", "1")
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.5.17.post1")
    module = ModuleType("sglang.srt.managers.scheduler")
    module.Scheduler = type("Scheduler", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    register()
    assert callable(module.Scheduler.relax_weight_version)


def test_larger_worker_configuration_preserves_content_and_scales_memory(tmp_path):
    configuration = PublicationConfig("unused", "s", "unused", 1 << 24, 1024, encode_workers=12)
    assert configuration.encode_workers == 12
    assert list(ordered_map(lambda n: n * 2, range(30), 12)) == list(range(0, 60, 2))
    store = VersionStore(tmp_path / "store", disk_budget=1 << 24, max_weight_bytes=1024)
    raw = tmp_path / "weights"
    raw.write_bytes(bytes(256))
    manifest = store.publish("s", 0, {"w": ("BF16", [128], raw)}, encode_workers=12)
    assert store.has_verified_version("s", 0, manifest["root"])
    assert source_demand(0, 0, 0, adaptive=False, bridge=False, encode_workers=12)["ram_bytes"] == 12 * 64 * 1024**2
    assert TCPTransport(("127.0.0.1", 1), retries=12).retries == 12


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5])
def test_invalid_workers_are_still_rejected(bad):
    with pytest.raises(ValueError):
        list(ordered_map(str, range(1), bad))


def test_independent_consumers_share_capacity_without_global_serialization(tmp_path):
    budget = NodeBudget(tmp_path, {"host": 100}, {"host": 10})
    with budget.operation("a", {"host": 40}, quiescent=lambda: True):
        with time_budget(0.2):
            with budget.operation("b", {"host": 40}, quiescent=lambda: True):
                assert len(list(tmp_path.glob("reservation-*.json"))) == 2
                with pytest.raises(OSError, match="capacity unavailable"):
                    with budget.operation("c", {"host": 20}, quiescent=lambda: True):
                        pytest.fail("Capacity limit bypassed")
    assert not list(tmp_path.glob("reservation-*.json"))


def test_completed_operation_survives_reservation_cleanup_failure(tmp_path, monkeypatch):
    budget = NodeBudget(tmp_path, {"host": 100}, {"host": 10})
    original = Path.unlink

    def fail(path, *args, **kwargs):
        if path.name == "reservation-a.json":
            raise PermissionError("busy reservation file")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail)
        with budget.operation("a", {"host": 40}, quiescent=lambda: True):
            pass
    assert (tmp_path / "reservation-a.json").exists()
    with budget.operation("a", {"host": 40}, quiescent=lambda: True):
        pass
    assert not (tmp_path / "reservation-a.json").exists()


def test_status_query_does_not_initialize_or_pause_an_engine(sglang_engine_module):
    actor = SimpleNamespace(_weight_consumer_enabled=True)
    assert sglang_engine_module.SGLangEngine.weight_version_status(actor)["status"] == "NOT_INITIALIZED"
    assert not hasattr(actor, "_version_consumer")


def test_long_installation_is_not_reported_as_a_failure_only_due_to_elapsed_time(monkeypatch):
    gate = AdmissionGate()
    monkeypatch.setattr("relax.distributed.weight_sync.admission.time.monotonic", lambda: gate.closed_at + 300)
    assert gate.unavailable()["status"] == "INSTALLING"
    assert gate.unavailable()["action"] == "retry_after_activation"
    gate.failure = "device operation failed"
    assert gate.unavailable()["action"] == "inspect_consumer_status_and_recover_checkpoint"


@pytest.mark.parametrize("enabled,explicit,expected", [(True, None, 1600), (True, 5, 5), (False, None, None)])
def test_managed_requests_honor_operation_budget_and_preserve_legacy_defaults(
    monkeypatch, sglang_engine_module, enabled, explicit, expected
):
    from unittest.mock import Mock

    import requests

    actor = SimpleNamespace(node_rank=0, server_host="unused", server_port=1, _weight_consumer_enabled=enabled)
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    clock = [10.0]
    monkeypatch.setattr("relax.distributed.weight_sync.deadline.time.monotonic", lambda: clock[0])
    with time_budget(1800):
        clock[0] += 200
        sglang_engine_module.SGLangEngine._make_request(actor, "get_server_info", timeout=explicit)
    assert post.call_args.kwargs["timeout"] == expected
