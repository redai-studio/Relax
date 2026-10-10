# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Characterization of ``EngineGroup``: what it asks Ray for when it brings up,
switches, checks and shuts down rollout engines.

These tests pin current behavior so the engine-pool unification can be shown
not to change it. They must keep passing unmodified across that refactor.
"""

import pytest


try:
    from relax.distributed.ray import rollout

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import make_engine_group, make_mock_args, make_mock_engine, make_rollout_server, mock_ray_get

from relax.engine.inference.discovery import EngineState


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


def _start(monkeypatch, group):
    """Run ``group.start_engines()`` against a fake actor class.

    Returns ``(creations, handles, port_cursors, allocation_kwargs)`` where
    each creation records the ``.options()`` kwargs, constructor args and the
    fake engine handle.
    """
    creations = []
    allocation = {}

    class FakeRolloutRayActor:
        @classmethod
        def options(cls, **kwargs):
            creations.append({"options": kwargs})
            return cls

        @classmethod
        def remote(cls, *args, **kwargs):
            engine = make_mock_engine()
            engine.init.remote.side_effect = lambda **init_kwargs: ("init-handle", init_kwargs)
            creations[-1].update(ctor_args=args, ctor_kwargs=kwargs, engine=engine)
            return engine

    def fake_allocate(**kwargs):
        allocation.update(kwargs)
        ports = {
            rank: {"host": "10.0.0.1", "port": 15000 + rank, "nccl_port": 16000 + rank, "dist_init_addr": "10.0.0.1:1"}
            for rank, _engine in kwargs["rollout_engines"]
        }
        return ports, {0: 15900}

    monkeypatch.setattr(rollout.ray, "remote", lambda cls: FakeRolloutRayActor)
    monkeypatch.setattr(rollout, "PlacementGroupSchedulingStrategy", lambda **kwargs: kwargs)
    monkeypatch.setattr(rollout, "get_ray_accelerator_kwargs", lambda num_gpus: {"accelerator_gpus": num_gpus})
    monkeypatch.setattr(rollout, "_allocate_rollout_engine_addr_and_ports_normal", fake_allocate)

    handles, port_cursors = group.start_engines()
    return creations, handles, port_cursors, allocation


def _group(pg_size=8, **kwargs):
    group = make_engine_group(**kwargs)
    group.pg = ("pg", [100 + i for i in range(pg_size)], list(range(pg_size)))
    return group


def test_engine_group_start_requests_one_actor_per_empty_slot(monkeypatch):
    args = make_mock_args()
    group = _group(args=args, engines=[None, None], num_gpus_per_engine=2, rank_offset=4)
    group.gpu_offset = 2
    group.sglang_overrides = {"mem_fraction_static": 0.5}

    creations, handles, port_cursors, allocation = _start(monkeypatch, group)

    assert len(creations) == 2
    first, second = creations
    # Ray resources and placement: 0.2 CPU/GPU, bundle and GPU id taken at gpu_offset + i * gpus_per_engine.
    assert first["options"]["num_cpus"] == 0.2
    assert first["options"]["accelerator_gpus"] == 0.2
    assert first["options"]["scheduling_strategy"] == {
        "placement_group": "pg",
        "placement_group_capture_child_tasks": True,
        "placement_group_bundle_index": 102,
    }
    assert second["options"]["scheduling_strategy"]["placement_group_bundle_index"] == 104
    # Constructor: global rank = rank_offset + i, base GPU id from the placement group.
    assert first["ctor_args"] == (args,)
    assert first["ctor_kwargs"] == {
        "rank": 4,
        "worker_type": "regular",
        "base_gpu_id": 2,
        "sglang_overrides": {"mem_fraction_static": 0.5},
        "num_gpus_per_engine": 2,
        "register_sigterm_handler": False,
    }
    assert (second["ctor_kwargs"]["rank"], second["ctor_kwargs"]["base_gpu_id"]) == (5, 4)
    # Slots are filled in place and counted.
    assert group.all_engines == [first["engine"], second["engine"]]
    assert group.num_new_engines == 2
    # Port allocation is asked for exactly the new engines, with the group's layout.
    assert [rank for rank, _engine in allocation["rollout_engines"]] == [4, 5]
    assert (allocation["worker_type"], allocation["num_gpus_per_engine"], allocation["rank_offset"]) == (
        "regular",
        2,
        4,
    )
    assert port_cursors == {0: 15900}
    # init() is fired without waiting; one handle per new engine.
    assert [handle[0] for handle in handles] == ["init-handle", "init-handle"]
    assert handles[0][1] == {
        "host": "10.0.0.1",
        "port": 15004,
        "nccl_port": 16004,
        "dist_init_addr": "10.0.0.1:1",
        "router_ip": "127.0.0.1",
        "router_port": 3000,
        "skip_dcs_registration": False,
        "skip_router_registration": False,
    }


def test_engine_group_start_sets_rollout_engine_env(monkeypatch):
    group = _group(engines=[None], num_gpus_per_engine=2)

    creations, _handles, _cursors, _allocation = _start(monkeypatch, group)

    env_vars = creations[0]["options"]["runtime_env"]["env_vars"]
    assert env_vars["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] == "false"
    assert env_vars["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "true"
    assert env_vars["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] == "0"
    assert env_vars["SLIME_ENABLE_PROFILING"] == "true"
    for name in rollout.NOSET_VISIBLE_DEVICES_ENV_VARS_LIST:
        assert env_vars[name] == "1"


def test_engine_group_start_passes_registration_flags_and_scaled_out_marker(monkeypatch):
    group = _group(engines=[None], num_gpus_per_engine=2, is_scaled_out=True)
    group.skip_dcs_registration = True
    group.skip_router_registration = True

    creations, handles, _cursors, _allocation = _start(monkeypatch, group)

    assert creations[0]["ctor_kwargs"]["register_sigterm_handler"] is True
    assert handles[0][1]["skip_dcs_registration"] is True
    assert handles[0][1]["skip_router_registration"] is True


def test_engine_group_start_keeps_live_engines(monkeypatch):
    survivor = make_mock_engine()
    group = _group(engines=[survivor, None], num_gpus_per_engine=2)

    creations, handles, _cursors, allocation = _start(monkeypatch, group)

    assert len(creations) == 1 and len(handles) == 1
    assert group.all_engines[0] is survivor
    assert creations[0]["ctor_kwargs"]["rank"] == 1
    assert [rank for rank, _engine in allocation["rollout_engines"]] == [1]
    assert group.num_new_engines == 1


def test_engine_group_start_is_a_noop_for_placeholder_and_train_only(monkeypatch):
    placeholder = _group(engines=[], worker_type="placeholder")
    creations, handles, _cursors, _allocation = _start(monkeypatch, placeholder)
    assert (creations, handles, placeholder.num_new_engines) == ([], [], 0)

    train_only = _group(args=make_mock_args(debug_train_only=True), engines=[None])
    creations, handles, _cursors, _allocation = _start(monkeypatch, train_only)
    assert (creations, handles, train_only.all_engines) == ([], [], [None])


def test_engine_group_multi_node_engine_exposes_only_head_slots(monkeypatch):
    # 16 GPUs per engine on 8-GPU nodes: two slots per logical engine.
    group = _group(pg_size=32, engines=[None] * 4, num_gpus_per_engine=16)

    creations, _handles, _cursors, _allocation = _start(monkeypatch, group)

    assert group.nodes_per_engine == 2
    assert [creation["ctor_kwargs"]["base_gpu_id"] for creation in creations] == [0, 8, 16, 24]
    assert group.engines == [creations[0]["engine"], creations[2]["engine"]]


def test_engine_group_caches_actual_head_urls_only_after_initialization(monkeypatch):
    group = _group(pg_size=32, engines=[None] * 4, num_gpus_per_engine=16, rank_offset=4)
    monkeypatch.setattr(rollout.ray, "get", mock_ray_get)
    creations, _handles, _cursors, _allocation = _start(monkeypatch, group)
    # Effective URLs may differ from allocated addresses (overrides/custom engines).
    creations[0]["engine"].get_url.remote.return_value.value = "http://[2001:db8::1]:18000"
    creations[2]["engine"].get_url.remote.return_value.value = "http://custom-engine:19000"

    assert group.lifecycle_states(EngineState.READY) == [EngineState.STARTING] * 2
    assert group.lifecycle_states(EngineState.READY) == [EngineState.STARTING] * 2
    group._cache_engine_urls()

    assert group._engine_urls == {0: "http://[2001:db8::1]:18000", 2: "http://custom-engine:19000"}
    assert group.lifecycle_states(EngineState.READY) == [EngineState.READY] * 2
    creations[1]["engine"].get_url.remote.assert_not_called()
    creations[3]["engine"].get_url.remote.assert_not_called()


def test_engine_group_recovery_refreshes_only_rebuilt_engine_urls(monkeypatch):
    group = _group(engines=[None, None])
    monkeypatch.setattr(rollout.ray, "get", mock_ray_get)
    creations, _handles, _cursors, _allocation = _start(monkeypatch, group)
    for index, creation in enumerate(creations):
        creation["engine"].get_url.remote.return_value.value = f"http://original:{18000 + index}"
    group._cache_engine_urls()
    survivor = group.all_engines[1]
    survivor.get_url.remote.reset_mock()
    group.all_engines[0] = None
    waiting_states = []

    def wait_for_init(refs, timeout=None):
        if refs and isinstance(refs[0], tuple) and refs[0][0] == "init-handle":
            waiting_states.append(group.lifecycle_states(EngineState.READY))
            assert 0 not in group._engine_urls
        return mock_ray_get(refs, timeout=timeout)

    monkeypatch.setattr(rollout.ray, "get", wait_for_init)
    make_rollout_server(engine_groups=[group]).recover()

    assert waiting_states == [[EngineState.STARTING, EngineState.READY]]
    assert group._engine_urls == {0: "http://localhost:30000", 1: "http://original:18001"}
    assert group.lifecycle_states(EngineState.READY) == [EngineState.READY] * 2
    survivor.get_url.remote.assert_not_called()


def test_engine_group_offload_and_onload_fan_out_to_head_engines():
    head_a, follower_a, head_b, follower_b = (make_mock_engine() for _ in range(4))
    group = make_engine_group(engines=[head_a, follower_a, head_b, follower_b], num_gpus_per_engine=16)
    head_a.release_memory_occupation.remote.return_value = "release-a"
    head_b.release_memory_occupation.remote.return_value = "release-b"
    head_a.resume_memory_occupation.remote.return_value = "resume-a"
    head_b.resume_memory_occupation.remote.return_value = "resume-b"

    assert group.offload() == ["release-a", "release-b"]
    assert group.onload(tags=["weights"]) == ["resume-a", "resume-b"]

    head_a.resume_memory_occupation.remote.assert_called_once_with(tags=["weights"])
    follower_a.release_memory_occupation.remote.assert_not_called()
    follower_b.resume_memory_occupation.remote.assert_not_called()


def test_engine_group_offload_skips_dead_slots():
    alive = make_mock_engine()
    alive.release_memory_occupation.remote.return_value = "release"
    group = make_engine_group(engines=[None, alive], num_gpus_per_engine=2)

    assert group.offload() == ["release"]


def test_engine_group_healthcheck_reports_failed_indices(monkeypatch):
    healthy, broken = make_mock_engine(), make_mock_engine()
    broken.health_generate.remote.side_effect = RuntimeError("down")
    group = make_engine_group(engines=[healthy, None, broken], num_gpus_per_engine=2)
    monkeypatch.setattr(rollout.ray, "get", mock_ray_get)

    assert group.healthcheck_engines(timeout=3.0) == {2}
    healthy.health_generate.remote.assert_called_once_with(timeout=3.0)


def test_engine_group_shutdown_unregisters_kills_and_clears_slot(monkeypatch):
    doomed, kept = make_mock_engine(), make_mock_engine()
    group = make_engine_group(engines=[doomed, kept], num_gpus_per_engine=2)
    killed = []
    monkeypatch.setattr(rollout.ray, "get", mock_ray_get)
    monkeypatch.setattr(rollout.ray, "kill", killed.append)

    group.shutdown_engines({0})

    doomed.shutdown.remote.assert_called_once_with()
    doomed.unregister_dcs.remote.assert_called_once_with()
    doomed.unregister_from_router.remote.assert_called_once_with()
    assert killed == [doomed]
    assert group.all_engines == [None, kept]
    kept.shutdown.remote.assert_not_called()
