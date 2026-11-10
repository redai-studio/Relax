# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import sys
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


def test_managed_pd_defers_endpoint_until_owner_starts_router(monkeypatch, rollout_module):
    from sglang_router.launch_router import RouterArgs

    args = _router_args()
    monkeypatch.setattr(rollout_module, "get_host_info", lambda: ("host", "192.0.2.1"))
    monkeypatch.setattr(rollout_module, "Envs", SimpleNamespace(SLIME_HOST_IP=None))
    find_port = Mock(side_effect=lambda port: port)
    monkeypatch.setattr(rollout_module, "find_available_port", find_port)
    process = Mock()
    process.is_alive.return_value = True
    constructor = Mock(return_value=process)
    monkeypatch.setattr(rollout_module.multiprocessing, "Process", constructor)
    router_args = SimpleNamespace()
    monkeypatch.setattr(RouterArgs, "from_cli_args", lambda *a, **k: router_args)

    assert rollout_module._start_router(args, has_pd_disaggregation=True) == (None, None)
    constructor.assert_not_called()
    find_port.assert_not_called()
    assert (args.sglang_router_ip, args.sglang_router_port) == (None, 3000)
    monkeypatch.setattr(rollout_module, "get_host_info", lambda: ("owner", "192.0.2.2"))

    assert rollout_module._start_router(args, has_pd_disaggregation=True, initial_pd_workers=_specs()) == (
        "192.0.2.2",
        3000,
    )
    assert router_args.host == "192.0.2.2" and router_args.port == 3000
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


@pytest.mark.parametrize("fail_at", ["engines", "publication", None])
def test_startup_publishes_ready_router_or_rolls_back_owned_resources(monkeypatch, rollout_module, fail_at):
    existing, created = Mock(), Mock()
    existing.is_alive.return_value = created.is_alive.return_value = True
    created.join.side_effect = lambda **kwargs: setattr(created.is_alive, "return_value", False)
    rollout_module._LAUNCHED_ROUTER_PROCESSES.append(existing)
    group = make_engine_group(engines=[make_mock_engine()])
    shutdown = Mock()
    monkeypatch.setattr(group, "shutdown_engines", shutdown)

    def start(args, pg, groups):
        groups.append(group)
        rollout_module._LAUNCHED_ROUTER_PROCESSES.append(created)
        if fail_at == "engines":
            raise RuntimeError("startup failed")
        args.sglang_router_ip, args.sglang_router_port = "ready-router", 3000
        return {"default": group}

    publish = Mock(side_effect=RuntimeError("startup failed") if fail_at == "publication" else None)
    monkeypatch.setitem(sys.modules, "relax.agentic.rollout", SimpleNamespace(bind_agentic_router=publish))
    monkeypatch.setattr(rollout_module, "_start_rollout_servers_impl", start)
    args = _router_args(use_agentic_rollout=True)
    if fail_at is None:
        assert rollout_module.start_rollout_servers(args, None) == {"default": group}
        shutdown.assert_not_called()
        created.terminate.assert_not_called()
        assert rollout_module._LAUNCHED_ROUTER_PROCESSES == [existing, created]
    else:
        with pytest.raises(RuntimeError, match="startup failed"):
            rollout_module.start_rollout_servers(args, None)
        shutdown.assert_called_once_with({0})
        created.terminate.assert_called_once()
        assert rollout_module._LAUNCHED_ROUTER_PROCESSES == [existing]
    existing.terminate.assert_not_called()
    if fail_at == "engines":
        publish.assert_not_called()
    else:
        publish.assert_called_once_with("ready-router", 3000)


@pytest.mark.parametrize("defer_router", [False, True])
def test_start_rollout_servers_initializes_models_before_publishing_router(monkeypatch, rollout_module, defer_router):
    args = _router_args(use_agentic_rollout=True, sglang_hf_checkpoint=None)
    args.router_decode_policy = "power_of_two" if defer_router else "consistent_hashing"
    models = [
        rollout_module.ModelConfig(
            name=name,
            engine_groups=[
                rollout_module.EngineGroupConfig("prefill", 2),
                rollout_module.EngineGroupConfig("decode", 4),
            ],
        )
        for name in ("actor", "other")
    ]
    monkeypatch.setattr(rollout_module, "_resolve_sglang_config", lambda _: rollout_module.SglangConfig(models))
    events = []

    def start_engines(group, cursors):
        assert group.skip_router_registration is defer_router
        assert group.require_router_registration is defer_router
        events.append("engines")
        return [object()], cursors

    def wait(handles, *, model_name, **kwargs):
        assert len(handles) == 2
        events.append(f"ready-{model_name}")

    def start_router(args, *, force_new, **kwargs):
        name = "other" if force_new else "actor"
        events.append(f"router-{name}")
        return f"router-{name}", 3001 if force_new else 3000

    def start_managed(args, groups, *, force_new):
        assert [group.worker_type for group in groups] == ["prefill", "decode"]
        name = "other" if force_new else "actor"
        assert events[-1] == f"ready-{name}"
        return start_router(args, force_new=force_new)

    def publish(host, port):
        assert (host, port) == (args.sglang_router_ip, args.sglang_router_port) == ("router-actor", 3000)
        events.append("publish")

    monkeypatch.setattr(rollout_module.EngineGroup, "start_engines", start_engines)
    monkeypatch.setattr(rollout_module, "_wait_engine_init_with_progress", wait)
    dynamic = Mock(side_effect=start_router)
    managed = Mock(side_effect=start_managed)
    monkeypatch.setattr(rollout_module, "_start_router", dynamic)
    monkeypatch.setattr(rollout_module, "_start_managed_pd_router", managed)
    monkeypatch.setitem(sys.modules, "relax.agentic.rollout", SimpleNamespace(bind_agentic_router=publish))

    servers = rollout_module.start_rollout_servers(args, object())

    assert [(server.router_ip, server.router_port) for server in servers.values()] == [
        ("router-actor", 3000),
        ("router-other", 3001),
    ]
    if defer_router:
        dynamic.assert_not_called()
        assert managed.call_count == 2
        assert events == [
            "engines",
            "engines",
            "ready-actor",
            "router-actor",
            "engines",
            "engines",
            "ready-other",
            "router-other",
            "publish",
        ]
    else:
        managed.assert_not_called()
        assert dynamic.call_count == 2
        assert events == [
            "router-actor",
            "engines",
            "engines",
            "ready-actor",
            "router-other",
            "engines",
            "engines",
            "ready-other",
            "publish",
        ]


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


@pytest.mark.parametrize("failure", [None, "engine", "detach"])
def test_rollout_owner_releases_tq_and_router_on_disposal(monkeypatch, rollout_module, failure):
    cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(cls)
    manager._stop_eviction_monitor = Mock()
    manager._health_monitors = [Mock()]
    manager.data_system_client = Mock()
    events = []

    def cleanup(stage):
        events.append(stage)
        if stage == failure:
            raise RuntimeError(f"{stage} shutdown failed")

    manager._shutdown_all_engines = lambda: cleanup("engine")
    monkeypatch.setattr(rollout_module, "detach_tq_client", lambda: cleanup("detach"))
    monkeypatch.setattr(rollout_module, "stop_launched_routers", lambda: cleanup("router"))
    if failure is None:
        manager.dispose()
    else:
        with pytest.raises(RuntimeError, match=f"{failure} shutdown failed"):
            manager.dispose()
    assert events == ["engine", "detach", "router"]
    assert manager.data_system_client is None


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


@pytest.mark.parametrize("nodes_per_engine", [1, 2])
def test_pd_recovery_rolls_back_failed_replacements_and_retries(monkeypatch, rollout_module, nodes_per_engine):
    from conftest import make_rollout_server

    args = make_mock_args(num_gpus_per_node=1)
    surviving = [make_mock_engine() for _ in range(nodes_per_engine)]
    group = make_engine_group(
        args=args,
        engines=surviving + [None] * (2 * nodes_per_engine),
        num_gpus_per_engine=nodes_per_engine,
        worker_type="decode",
    )
    group.require_router_registration = True
    group.pg = (object(), list(range(3 * nodes_per_engine)), list(range(3 * nodes_per_engine)))
    slots = list(range(nodes_per_engine, 3 * nodes_per_engine))
    records = {
        slot: rollout_module._EngineInitRecord(
            object(),
            router_url=f"http://worker-{slot}:8000",
            router_cleanup=rollout_module.RouterWorkerCleanup(f"old-{slot}", pd_worker_type="decode"),
        )
        for slot in slots
    }
    group.engine_init_records.update(records)
    cleanup = Mock(return_value=True)
    monkeypatch.setattr(rollout_module, "remove_dead_router_worker", cleanup)
    pending, failed = object(), object()
    replacements = [make_mock_engine() for _ in slots]
    # A shutdown queued behind pending init can time out; every actor must still be killed.
    shutdown_ref = object()
    replacements[0].shutdown.remote.return_value = shutdown_ref
    attempts = []

    def start(cursors):
        assert group.all_engines == surviving + [None] * len(slots)
        assert cleanup.call_count == 2 * (len(attempts) + 1)
        for slot, engine in zip(slots, replacements, strict=True):
            group.all_engines[slot] = engine
            records[slot] = rollout_module._EngineInitRecord(
                failed if slot == slots[-1] else pending,
                router_url=f"http://worker-{slot}:8000",
                router_cleanup=rollout_module.RouterWorkerCleanup(f"new-{slot}", pd_worker_type="decode"),
            )
            group.engine_init_records[slot] = records[slot]
        group.num_new_engines = len(slots)
        attempts.append(1)
        return [record.value for record in records.values()], cursors

    failure = RuntimeError("Required PD router registration failed")

    def get(refs, timeout=None):
        if isinstance(refs, list) and failed in refs and len(attempts) == 1:
            raise failure
        if refs is shutdown_ref:
            assert timeout == 10
            raise rollout_module.ray.exceptions.GetTimeoutError("shutdown queued behind pending init")
        return mock_ray_get(refs, timeout)

    monkeypatch.setattr(group, "start_engines", start)
    monkeypatch.setattr(rollout_module.ray, "get", get)
    kill = Mock()
    monkeypatch.setattr(rollout_module.ray, "kill", kill)
    server = make_rollout_server(engine_groups=[group])

    with pytest.raises(RuntimeError, match="Required PD router registration failed") as caught:
        server.recover()

    assert caught.value is failure
    assert group.all_engines == surviving + [None] * len(slots)
    assert server.num_new_engines == 0
    assert group.engine_init_records == records
    assert [call.args[0] for call in kill.call_args_list] == replacements
    for engine in surviving:
        engine.shutdown.remote.assert_not_called()

    server.recover()
    assert group.all_engines == surviving + replacements
    assert server.num_new_engines == len(slots)
    assert [call.args[2].actor_id for call in cleanup.call_args_list] == [
        f"{generation}-{slot}" for generation in ("old", "new") for slot in slots[::nodes_per_engine]
    ]


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
