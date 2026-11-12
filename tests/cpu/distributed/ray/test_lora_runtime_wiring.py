# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import HAS_DEPS, create_test_manager, make_mock_engine, mock_ray_get


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


@pytest.fixture
def wiring(monkeypatch):
    from relax.agentic import rollout as agentic
    from relax.distributed.ray import rollout
    from relax.engine.lora_versions import runtime

    events = []
    engine = make_mock_engine()
    server = SimpleNamespace(engines=[engine], engine_groups=[])
    manager = create_test_manager(servers={"default": server})
    manager.args.use_agentic_rollout = True
    manager.args.use_fault_tolerance = False
    manager._lora_runtime = None
    manager._lora_configuring = False
    manager._lora_training_config = None
    manager._generation_started = False
    manager._stop_eviction_monitor = lambda **_kwargs: None
    manager._shutdown_all_engines = lambda: events.append("engines-close")
    host = SimpleNamespace(
        start=SimpleNamespace(remote=lambda *args: events.append("start")),
        publish=SimpleNamespace(remote=lambda path: events.append("publish")),
        close=SimpleNamespace(remote=lambda: events.append("sessions-and-host-close")),
    )
    factory = Mock()
    factory.options.return_value.remote.return_value = host
    monkeypatch.setattr(runtime, "LoRAVersionRuntime", factory)
    monkeypatch.setattr(rollout, "with_control_plane_affinity", lambda args, options: options)
    monkeypatch.setattr(rollout.ray, "get", mock_ray_get)
    monkeypatch.setattr(rollout.ray, "kill", lambda actor: events.append("host-kill"))
    monkeypatch.setattr(rollout, "stop_launched_routers", lambda: events.append("routers-close"))
    monkeypatch.setattr(agentic, "bind_agentic_lora_runtime", lambda handle: events.append("sessions-bind"))
    return manager, host, events, agentic, factory


def test_lora_configuration_publishes_before_session_injection_and_disposes_in_order(wiring):
    manager, host, events, _agentic, _factory = wiring
    assert manager.configure_lora_versions("artifact", [1]) is host
    assert events == ["start", "publish", "sessions-bind"]
    assert manager.get_lora_version_runtime() is host
    with pytest.raises(RuntimeError, match="fixed engine pool"):
        manager.create_scale_out_request(num_replicas=2)
    with pytest.raises(RuntimeError, match="fixed engine pool"):
        manager.create_scale_in_request(num_replicas=1)
    manager.dispose()
    assert events[3:] == ["sessions-and-host-close", "host-kill", "engines-close", "routers-close"]


def test_lora_partial_configuration_retains_owner_and_blocks_pool_changes(wiring, monkeypatch):
    manager, host, _events, agentic, _factory = wiring
    monkeypatch.setattr(agentic, "bind_agentic_lora_runtime", Mock(side_effect=RuntimeError("injection failed")))
    with pytest.raises(RuntimeError, match="injection failed"):
        manager.configure_lora_versions("artifact", [1])
    assert manager._lora_runtime is host
    assert not manager._lora_configuring
    assert manager._lora_configuration_failed
    with pytest.raises(RuntimeError, match="incomplete"):
        manager._begin_generation()
    with pytest.raises(RuntimeError, match="already configured"):
        manager.configure_lora_versions("artifact", [1])
    manager.dispose()


def test_lora_actor_creation_failure_clears_configuring_but_fails_closed(wiring):
    manager, _host, _events, _agentic, factory = wiring
    factory.options.side_effect = RuntimeError("actor creation failed")
    with pytest.raises(RuntimeError, match="actor creation failed"):
        manager.configure_lora_versions("artifact", [1])
    assert not manager._lora_configuring
    with pytest.raises(RuntimeError, match="fixed engine pool"):
        manager._require_unmanaged_engine_pool()


def test_scale_out_rechecks_lora_mode_after_slow_endpoint_validation(wiring):
    manager, _host, _events, _agentic, _factory = wiring

    def collect(_server):
        manager._lora_configuring = True
        return set()

    manager._collect_existing_engine_addrs = collect
    with pytest.raises(RuntimeError, match="fixed engine pool"):
        manager.create_scale_out_request(engine_urls=["http://new:8000"])
    assert not manager._scale_out_requests


def test_lora_configuration_rejects_generation_that_already_started(wiring):
    manager, _host, events, _agentic, _factory = wiring
    manager._begin_generation()
    with pytest.raises(RuntimeError, match="before generation"):
        manager.configure_lora_versions("artifact", [1])
    assert not events


def test_lora_configuration_rejects_global_health_recovery(wiring):
    manager, _host, events, _agentic, factory = wiring
    manager.args.use_health_check = True
    with pytest.raises(ValueError, match="disabled automatic recovery"):
        manager.configure_lora_versions("artifact", [1])
    assert not events and manager._lora_runtime is None
    factory.options.assert_not_called()


def prepare_training(manager, tmp_path):
    manager.args.fully_async = True
    manager.args.hybrid = manager.args.colocate = manager.args.offload_rollout = False
    manager.args.lora_adapter_mode = True
    next(iter(manager.servers.values())).engines[0]._actor_id.hex.return_value = "engine-id"
    return manager.configure_lora_training(str(tmp_path), "frozen-base", [1])


def test_training_configuration_blocks_generation_until_initial_publication(wiring, tmp_path):
    manager, host, events, _agentic, _factory = wiring
    config = prepare_training(manager, tmp_path)
    assert config.engine_actor_ids == ("engine-id",)
    assert manager.get_lora_training_config() == config
    assert not events
    with pytest.raises(RuntimeError, match="has not completed"):
        manager._begin_generation()
    with pytest.raises(RuntimeError, match="fixed engine pool"):
        manager.create_scale_out_request(num_replicas=2)
    with pytest.raises(RuntimeError, match="owns the initial"):
        manager.configure_lora_versions("artifact", [1])

    def publish(path):
        events.append("publish")
        return SimpleNamespace(identity=SimpleNamespace(version_id=0))

    host.publish.remote = publish
    assert manager.publish_training_lora("artifact") == {"status": "published", "version_id": 0}
    assert events == ["start", "publish", "sessions-bind"]
    manager._begin_generation()
    host.publish_for_training = SimpleNamespace(remote=Mock(return_value={"status": "capacity_skipped"}))
    assert manager.publish_training_lora("next") == {"status": "capacity_skipped"}
    host.publish_for_training.remote.assert_called_once_with("next")
    assert events == ["start", "publish", "sessions-bind"]


def test_failed_training_bootstrap_cannot_resume_unmanaged_generation(wiring, tmp_path):
    manager, host, _events, _agentic, _factory = wiring
    prepare_training(manager, tmp_path)
    host.publish.remote = Mock(side_effect=RuntimeError("first publication failed"))
    assert manager.publish_training_lora("artifact")["status"] == "failed"
    assert manager.publish_training_lora("artifact")["status"] == "failed"
    host.publish.remote.assert_called_once()
    with pytest.raises(RuntimeError, match="incomplete"):
        manager._begin_generation()


@pytest.mark.parametrize(
    "argument,value",
    [
        ("fully_async", False),
        ("hybrid", True),
        ("colocate", True),
        ("offload_rollout", True),
        ("train_backend", "fsdp"),
        ("use_health_check", True),
    ],
)
def test_training_configuration_rejects_unsupported_modes(wiring, tmp_path, argument, value):
    manager, _host, events, _agentic, _factory = wiring
    manager.args.fully_async = True
    manager.args.hybrid = manager.args.colocate = manager.args.offload_rollout = False
    manager.args.lora_adapter_mode = True
    setattr(manager.args, argument, value)
    with pytest.raises(ValueError, match="fully-async"):
        manager.configure_lora_training(str(tmp_path), "base", [1])
    assert manager._lora_training_config is None and not events


def test_publication_and_disposal_share_a_group_separate_from_generation():
    from relax.distributed.ray.rollout import RolloutManager

    groups = RolloutManager.__ray_metadata__.method_meta.concurrency_group_for_methods
    publication_group = groups["publish_training_lora"]
    assert all(
        groups[method] == publication_group
        for method in ("configure_lora_training", "configure_lora_versions", "dispose")
    )
    assert groups.get("generate") != publication_group
