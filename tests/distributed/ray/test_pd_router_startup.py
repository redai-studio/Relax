# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import HAS_DEPS, make_engine_group, make_mock_args, make_mock_engine, mock_ray_get


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _specs():
    return [
        {"url": "http://p:8000", "worker_type": "prefill", "bootstrap_port": 9000},
        {"url": "http://d1:8000", "worker_type": "decode", "bootstrap_port": None},
        {"url": "http://d2:8000", "worker_type": "decode", "bootstrap_port": None},
    ]


def _registry():
    return [{**spec, "id": f"worker-{i}", "is_healthy": True} for i, spec in enumerate(_specs())]


class _Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def rollout_module(monkeypatch):
    from relax.distributed.ray import rollout

    monkeypatch.setattr(rollout, "_LAUNCHED_ROUTER_PROCESSES", [])
    monkeypatch.setattr(rollout, "time", _Clock())
    return rollout


@pytest.mark.parametrize("invalid", ["missing_p", "one_d", "duplicate", "role", "bootstrap", "url"])
def test_static_pd_specs_reject_invalid_topology(rollout_module, invalid):
    specs = _specs()
    if invalid == "missing_p":
        specs.pop(0)
    elif invalid == "one_d":
        specs.pop()
    elif invalid == "duplicate":
        specs[-1]["url"] = specs[1]["url"] + "/"
    elif invalid == "role":
        specs[-1]["worker_type"] = "regular"
    elif invalid == "bootstrap":
        specs[0]["bootstrap_port"] = None
    else:
        specs[0]["url"] = "not-an-http-url"
    with pytest.raises(ValueError):
        rollout_module._validate_initial_pd_workers(specs)


@pytest.mark.parametrize("invalid", ["duplicate", "extra", "role", "id", "bootstrap"])
def test_static_pd_registry_rejects_ambiguous_workers(monkeypatch, rollout_module, invalid):
    import requests

    workers = _registry()
    if invalid == "duplicate":
        workers.append(dict(workers[0]))
    elif invalid == "extra":
        workers.append({**workers[1], "url": "http://unexpected:8000"})
    elif invalid == "role":
        workers[0]["worker_type"] = "decode"
    elif invalid == "id":
        workers[0]["id"] = "rank@0"
    else:
        workers[0]["bootstrap_port"] = 9001
    monkeypatch.setattr(
        requests,
        "get",
        Mock(
            return_value=SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"workers": workers},
            )
        ),
    )
    with pytest.raises(ValueError):
        rollout_module._wait_pd_router_workers("router", 3000, _specs(), timeout=1)


@pytest.mark.parametrize("unready", ["missing", "unhealthy"])
def test_static_pd_registry_waits_for_every_healthy_worker(monkeypatch, rollout_module, unready):
    import requests

    workers = _registry()
    pending = workers[:-1] if unready == "missing" else [{**w, "is_healthy": False} for w in workers]
    responses = [pending, workers]
    get = Mock(
        side_effect=lambda *a, **k: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"workers": responses.pop(0)},
        )
    )
    monkeypatch.setattr(requests, "get", get)
    result = rollout_module._wait_pd_router_workers("router", 3000, _specs(), timeout=1)
    assert set(result) == {spec["url"] for spec in _specs()}
    assert get.call_count == 2


def test_static_pd_registry_times_out_without_publication(monkeypatch, rollout_module):
    import requests

    monkeypatch.setattr(
        requests,
        "get",
        Mock(
            return_value=SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"workers": []},
            )
        ),
    )
    with pytest.raises(TimeoutError):
        rollout_module._wait_pd_router_workers("router", 3000, _specs(), timeout=1)


def _router_args(**changes):
    return make_mock_args(
        sglang_router_ip=None,
        sglang_router_port=3000,
        use_slime_router=False,
        sglang_router_request_timeout_secs=60,
        router_decode_policy="power_of_two",
        **changes,
    )


def test_managed_pd_keeps_endpoint_published_to_agentic_clients(monkeypatch, rollout_module):
    from sglang_router.launch_router import RouterArgs

    args = _router_args()
    monkeypatch.setattr(rollout_module, "get_host_info", lambda: ("host", "192.0.2.1"))
    monkeypatch.setattr(rollout_module, "Envs", SimpleNamespace(SLIME_HOST_IP=None))
    monkeypatch.setattr(rollout_module, "find_available_port", lambda port: port)
    process = Mock()
    process.is_alive.return_value = True
    constructor = Mock(return_value=process)
    monkeypatch.setattr(rollout_module.multiprocessing, "Process", constructor)
    router_args = SimpleNamespace()
    monkeypatch.setattr(RouterArgs, "from_cli_args", lambda *a, **k: router_args)

    endpoint = rollout_module._start_router(args, has_pd_disaggregation=True)
    constructor.assert_not_called()
    args.sglang_router_ip, args.sglang_router_port = endpoint
    monkeypatch.setattr(rollout_module, "get_host_info", lambda: ("owner", "192.0.2.2"))

    assert rollout_module._start_router(args, has_pd_disaggregation=True, initial_pd_workers=_specs()) == endpoint
    assert router_args.host == "192.0.2.1" and router_args.port == 3000
    assert router_args.prefill_urls == [("http://p:8000", 9000)]
    assert router_args.decode_urls == ["http://d1:8000", "http://d2:8000"]
    assert rollout_module._LAUNCHED_ROUTER_PROCESSES == [process]


def test_external_router_is_reused_without_static_mutation(rollout_module):
    args = _router_args()
    args.sglang_router_ip = "external"
    assert rollout_module._start_router(args, has_pd_disaggregation=True) == ("external", 3000)
    with pytest.raises(ValueError, match="external router"):
        rollout_module._start_router(args, has_pd_disaggregation=True, initial_pd_workers=_specs())
    assert not rollout_module._LAUNCHED_ROUTER_PROCESSES


def test_managed_pd_adopts_registry_before_publishing_groups(monkeypatch, rollout_module):
    entries = []
    for spec in _specs():
        engine = make_mock_engine()
        engine.get_pd_router_worker_spec.remote.return_value = spec
        engine.bind_registered_pd_worker.remote.return_value = True
        group = make_engine_group(engines=[engine], worker_type=spec["worker_type"])
        group.skip_router_registration = True
        group.engine_init_records[0] = rollout_module._EngineInitRecord(
            {"url": spec["url"], "pid": 1, "node_id": "node"},
            router_cleanup=rollout_module.RouterWorkerCleanup("actor"),
        )
        entries.append((group, engine))
    groups = [group for group, _ in entries]
    monkeypatch.setattr(rollout_module.ray, "get", mock_ray_get)
    monkeypatch.setattr(rollout_module, "_start_router", lambda *a, **k: ("router", 3000))
    monkeypatch.setattr(rollout_module, "_wait_pd_router_workers", lambda *a, **k: {w["url"]: w for w in _registry()})

    assert rollout_module._start_managed_pd_router(_router_args(), groups, force_new=False) == ("router", 3000)
    for group, engine in entries:
        assert not group.skip_router_registration
        assert (group.router_ip, group.router_port) == ("router", 3000)
        engine.bind_registered_pd_worker.remote.assert_called_once()
        engine.register_to_router.remote.assert_not_called()
        assert group.engine_init_records[0].router_cleanup.worker_id is not None
        assert group.engine_init_records[0].router_cleanup.pd_worker_type == group.worker_type

    entries[-1][1].bind_registered_pd_worker.remote.return_value = False
    for group in groups:
        group.skip_router_registration = True
    with pytest.raises(RuntimeError, match="adopt"):
        rollout_module._start_managed_pd_router(_router_args(), groups, force_new=False)
    assert all(group.skip_router_registration for group in groups)


@pytest.mark.parametrize("phase", ["engine_init", "registry", "adoption"])
def test_startup_failure_rolls_back_only_owned_resources(monkeypatch, rollout_module, phase):
    existing, created = Mock(), Mock()
    existing.is_alive.return_value = created.is_alive.return_value = True
    created.join.side_effect = lambda **kwargs: setattr(created.is_alive, "return_value", False)
    rollout_module._LAUNCHED_ROUTER_PROCESSES.append(existing)
    group = make_engine_group(engines=[make_mock_engine()])
    shutdown = Mock()
    monkeypatch.setattr(group, "shutdown_engines", shutdown)

    def fail(args, pg, groups):
        groups.append(group)
        rollout_module._LAUNCHED_ROUTER_PROCESSES.append(created)
        raise RuntimeError(phase)

    monkeypatch.setattr(rollout_module, "_start_rollout_servers_impl", fail)
    with pytest.raises(RuntimeError, match=phase):
        rollout_module.start_rollout_servers(_router_args(), None)
    shutdown.assert_called_once_with({0})
    created.terminate.assert_called_once()
    existing.terminate.assert_not_called()
    assert rollout_module._LAUNCHED_ROUTER_PROCESSES == [existing]


def test_router_cleanup_retains_failed_process_for_retry(rollout_module):
    process = Mock()
    process.is_alive.return_value = True
    process.terminate.side_effect = OSError("temporarily unavailable")
    rollout_module._LAUNCHED_ROUTER_PROCESSES.append(process)
    assert rollout_module.stop_launched_routers() == 0
    assert rollout_module._LAUNCHED_ROUTER_PROCESSES == [process]
    process.terminate.side_effect = lambda: setattr(process.is_alive, "return_value", False)
    assert rollout_module.stop_launched_routers() == 1
    assert rollout_module._LAUNCHED_ROUTER_PROCESSES == []


def test_rollout_owner_stops_router_even_when_engine_disposal_fails(monkeypatch, rollout_module):
    cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(cls)
    manager._stop_eviction_monitor = Mock()
    manager._health_monitors = [Mock()]
    manager._shutdown_all_engines = Mock(side_effect=RuntimeError("engine shutdown failed"))
    stop = Mock()
    monkeypatch.setattr(rollout_module, "stop_launched_routers", stop)
    with pytest.raises(RuntimeError, match="engine shutdown failed"):
        manager.dispose()
    stop.assert_called_once()


def test_engine_shutdown_uses_one_deadline_for_all_pd_workers(monkeypatch, rollout_module):
    cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(cls)
    engines = [make_mock_engine() for _ in range(4)]
    manager.servers = {"default": SimpleNamespace(engine_groups=[SimpleNamespace(all_engines=engines)])}
    timeouts = []

    def get(ref, timeout):
        timeouts.append(timeout)
        rollout_module.time.sleep(timeout)
        raise TimeoutError("engine unavailable")

    killed = Mock()
    monkeypatch.setattr(rollout_module.ray, "get", get)
    monkeypatch.setattr(rollout_module.ray, "kill", killed)
    manager._shutdown_all_engines(timeout=15.0)
    assert timeouts == [15.0, 0.0, 0.0, 0.0]
    assert rollout_module.time.now == 15.0
    assert killed.call_count == 4


@pytest.mark.parametrize("removed", [False, True])
def test_pd_recovery_confirms_old_worker_removal_before_recreation(monkeypatch, rollout_module, removed):
    from conftest import make_rollout_server

    group = make_engine_group(engines=[None], worker_type="prefill")
    group.require_router_registration = True
    group.pg = (object(), [0], [0])
    group.engine_init_records[0] = rollout_module._EngineInitRecord(
        {"url": "http://p:8000", "pid": 1, "node_id": "node"},
        router_url="http://p:8000",
        router_cleanup=rollout_module.RouterWorkerCleanup(
            "dead-actor",
            worker_id="static-p",
            pd_worker_type="prefill",
            bootstrap_port=9000,
        ),
    )
    events = []

    def cleanup(*args):
        events.append("remove")
        return removed

    def start(cursors):
        assert events == ["remove"]
        events.append("start")
        group.all_engines[0] = make_mock_engine()
        group.num_new_engines = 1
        return [], cursors

    monkeypatch.setattr(rollout_module, "remove_dead_router_worker", cleanup)
    monkeypatch.setattr(group, "start_engines", start)
    server = make_rollout_server(engine_groups=[group])
    if removed:
        server.recover()
        assert events == ["remove", "start"]
    else:
        with pytest.raises(RuntimeError, match="old router registration"):
            server.recover()
        assert events == ["remove"]
        assert group.all_engines == [None]


def test_owner_signals_all_monitors_before_bounded_join(monkeypatch, rollout_module):
    cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(cls)
    events = []

    def stop(name, timeout):
        events.append((name, timeout))
        rollout_module.time.sleep(timeout)

    manager._stop_eviction_monitor = lambda timeout: stop("eviction", timeout)
    manager._health_monitors = [
        SimpleNamespace(stop=lambda timeout, i=i: stop(f"health-{i}", timeout)) for i in range(4)
    ]
    manager._shutdown_all_engines = lambda: events.append(("engines", None))
    monkeypatch.setattr(rollout_module, "stop_launched_routers", lambda: events.append(("routers", None)))
    manager.dispose()
    assert events[:5] == [("eviction", 0.0)] + [(f"health-{i}", 0.0) for i in range(4)]
    assert rollout_module.time.now == 5.0
    assert events[-2:] == [("engines", None), ("routers", None)]
