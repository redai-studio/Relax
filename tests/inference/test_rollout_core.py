# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from types import SimpleNamespace

import pytest

from relax.distributed.ray import inference_manager as module


class _Engine:
    def __init__(self, name, runtime):
        self.name = name
        self.runtime = runtime

    def __getattr__(self, method):
        def remote(**kwargs):
            self.runtime.events.append(("submit", self.name, method, kwargs))
            return self, method, kwargs

        return SimpleNamespace(remote=remote)


class _Group:
    def __init__(self, engines, *, rank=0, offset=0, pg="baseline", owned=False, scaled=False, status="ACTIVE"):
        self.all_engines = list(engines)
        self.nodes_per_engine = 2
        self.num_gpus_per_engine = 4
        self.num_new_engines = len(engines) // 2
        self.gpu_offset = offset
        self.rank_offset = rank
        self.pg = (pg, [], []) if pg is not None else None
        self.pg_owned = owned
        self.is_scaled_out = scaled
        self.lifecycle_status = status

    @property
    def engines(self):
        return self.all_engines[:: self.nodes_per_engine]


class _Server:
    def __init__(self, groups):
        self.engine_groups = list(groups)
        self.model_name = "policy"
        self.recovery = lambda: None

    @property
    def engines(self):
        return [engine for group in self.engine_groups for engine in group.engines]

    @property
    def num_new_engines(self):
        return sum(group.num_new_engines for group in self.engine_groups)

    @property
    def engine_gpu_counts(self):
        return [group.num_gpus_per_engine for group in self.engine_groups for engine in group.engines]

    @property
    def engine_gpu_offsets(self):
        return [
            group.gpu_offset + index * group.num_gpus_per_engine
            for group in self.engine_groups
            for index, _engine in enumerate(group.engines)
        ]

    def recover(self):
        self.recovery()


@pytest.fixture
def runtime(monkeypatch):
    runtime = SimpleNamespace(events=[], failures={}, removed=[])

    def get(ref, timeout=None):
        assert timeout is not None and timeout > 0
        if isinstance(ref, list):
            return [get(item, timeout) for item in ref]
        engine, method, kwargs = ref
        runtime.events.append(("ack", engine.name, method, kwargs))
        if (engine.name, method) in runtime.failures:
            raise runtime.failures[engine.name, method]
        return True

    monkeypatch.setattr(module.ray, "get", get)
    monkeypatch.setattr(module.ray, "kill", lambda engine: runtime.events.append(("kill", engine.name)))
    monkeypatch.setattr(module, "remove_placement_group", runtime.removed.append)
    return runtime


def _owner(runtime):
    engines = [_Engine(str(index), runtime) for index in range(4)]
    group = _Group(engines)
    server = _Server([group])
    args = SimpleNamespace(offload_rollout=True, _inference_preserve_rollout_weights=True)
    owner = module.InferenceManager.for_rollout(args, {"policy": server})
    for key, workers in owner.rollout_replicas().items():
        owner._inference_observation.initialized(key, workers, weights_ready=True)
    return owner, server, group, engines


def test_rollout_core_initializes_topology_once_and_preserves_five_tuple(runtime):
    engines = [_Engine(str(index), runtime) for index in range(4)]
    server = _Server([_Group(engines, offset=8)])
    servers = {"policy": server}
    owner = module.InferenceManager.for_rollout(SimpleNamespace())
    starts = []

    def start(args, pg):
        starts.append(pg)
        return servers

    assert owner.initialize_rollout(start, "borrowed") is servers
    assert owner.initialize_rollout(start, "borrowed") is servers
    assert starts == ["borrowed"]
    lock = object()
    assert owner.get_rollout_engines_and_lock(None, lock) == ([engines[0], engines[2]], lock, 2, [4, 4], [8, 12])
    assert owner.get_rollout_engines_and_lock("missing", lock) == ([], lock, 0, [], [])
    assert len(owner.rollout_replicas()) == 2


def test_rollout_core_adoption_updates_only_owner_topology_and_ownership_is_fixed(runtime):
    owner, server, _group, _engines = _owner(runtime)
    elastic = _Group(
        [_Engine("elastic-head", runtime), _Engine("elastic-follower", runtime)], rank=4, offset=8, pg="elastic"
    )
    owner.adopt_group("policy", elastic, owned_pg=True)
    owner.adopt_group("policy", elastic, owned_pg=True)

    assert owner.servers["policy"] is server
    assert server.engine_groups[-1] is elastic
    assert len(server.engine_groups) == 2
    assert elastic.pg_owned is True
    with pytest.raises(ValueError, match="ownership"):
        owner.adopt_group("policy", elastic, owned_pg=False)


def test_rollout_core_memory_calls_logical_heads_and_resumes_only_missing_tags(runtime):
    owner, _server, _group, _engines = _owner(runtime)
    owner.offload()
    owner.onload(["weights"])
    owner.onload(["weights"])
    owner.onload()
    owner.onload()

    calls = [event for event in runtime.events if event[0] == "submit"]
    assert {event[1] for event in calls} == {"0", "2"}
    assert [event[3]["tags"] for event in calls if event[1:3] == ("0", "resume_memory_occupation")] == [
        ["weights"],
        ["cuda_graph", "kv_cache"],
    ]
    assert owner.status == "onload"
    assert all(entry["state"] == "READY" for entry in owner._inference_observation.entries.values())


def test_rollout_core_excludes_nonactive_elastic_groups_from_production_operations(runtime):
    owner, server, _group, engines = _owner(runtime)
    draining = _Group(
        [_Engine("draining-head", runtime), _Engine("draining-follower", runtime)],
        rank=4,
        offset=8,
        pg="elastic",
        owned=True,
        scaled=True,
        status="DRAINING",
    )
    owner.adopt_group("policy", draining, owned_pg=True)

    assert set(owner.rollout_replicas()) == {"policy/group-0/replica-0", "policy/group-0/replica-1"}
    assert owner._active_rollout_groups(server) == [server.engine_groups[0]]
    assert owner.get_rollout_engines_and_lock(None, "lock") == (
        [engines[0], engines[2]],
        "lock",
        2,
        [4, 4],
        [0, 4],
    )

    owner.offload()

    submissions = [event for event in runtime.events if event[0] == "submit"]
    assert not any(event[1].startswith("draining-") for event in submissions)
    assert draining in server.engine_groups


def test_rollout_core_failed_drain_keeps_failed_state_and_does_not_claim_offload(runtime):
    owner, _server, _group, _engines = _owner(runtime)
    runtime.failures["0", "release_memory_occupation"] = RuntimeError("release failed")
    with pytest.raises(RuntimeError, match="release failed"):
        owner.offload()
    assert owner.status == "failed"
    assert {entry["state"] for entry in owner._inference_observation.entries.values()} == {"FAILED", "SLEEPING"}


def test_rollout_core_shutdown_preserves_borrowed_pg_and_removes_owned_pg_once(runtime):
    owner, _server, group, engines = _owner(runtime)
    elastic = _Group([_Engine("elastic-head", runtime), _Engine("elastic-follower", runtime)], rank=4, pg="elastic")
    owner.adopt_group("policy", elastic, owned_pg=True)

    owner.shutdown_rollout()
    owner.shutdown_rollout()

    assert runtime.removed == ["elastic"]
    assert owner.servers == {}
    assert group.all_engines == [None] * len(engines)
    assert elastic.all_engines == [None, None]
    assert {event[1] for event in runtime.events if event[0] == "kill"} == {
        "0",
        "1",
        "2",
        "3",
        "elastic-head",
        "elastic-follower",
    }


def test_rollout_core_unconfirmed_shutdown_preserves_actor_and_pg_for_retry(runtime):
    owner, _server, group, _engines = _owner(runtime)
    group.pg_owned = True
    runtime.failures["1", "shutdown"] = RuntimeError("child remains alive")

    with pytest.raises(module.InferenceCleanupError):
        owner.shutdown_rollout()

    assert group.all_engines[1] is not None
    assert ("kill", "1") not in runtime.events
    assert runtime.removed == []
    del runtime.failures["1", "shutdown"]
    owner.shutdown_rollout()
    assert runtime.removed == ["baseline"]


def test_rollout_core_recovery_never_republishes_rebuilt_weights_as_ready(runtime):
    owner, server, group, _engines = _owner(runtime)
    original = next(iter(owner._inference_observation.entries.values()))["generation"]

    def recover():
        group.all_engines[0] = _Engine("replacement", runtime)

    server.recovery = recover

    owner.recover_rollout("policy")

    entry = next(iter(owner._inference_observation.entries.values()))
    assert entry["generation"] > original
    assert entry["weights_ready"] is False
    assert entry["admission"] is False
    assert entry["state"] != "READY"


def test_rollout_core_missing_follower_cannot_acknowledge_complete_offload(runtime):
    owner, _server, group, _engines = _owner(runtime)
    group.all_engines[1] = None

    with pytest.raises(RuntimeError):
        owner.offload()

    assert owner.status != "offload"


def test_rollout_core_failed_onload_compensates_resumed_heads_without_ready_publication(runtime):
    owner, _server, _group, _engines = _owner(runtime)
    owner.offload()
    runtime.events.clear()
    runtime.failures["2", "resume_memory_occupation"] = RuntimeError("onload failed")

    with pytest.raises(RuntimeError, match="onload failed"):
        owner.onload()

    submissions = [(event[1], event[2]) for event in runtime.events if event[0] == "submit"]
    assert ("0", "release_memory_occupation") in submissions
    assert owner.status != "onload"
    assert all(entry["state"] != "READY" for entry in owner._inference_observation.entries.values())


def test_rollout_core_remove_group_requires_cleanup_and_removes_owned_pg(runtime):
    owner, server, _group, _engines = _owner(runtime)
    elastic = _Group([_Engine("elastic-head", runtime), _Engine("elastic-follower", runtime)], rank=4, pg="elastic")
    owner.adopt_group("policy", elastic, owned_pg=True)
    with pytest.raises(module.InferenceCleanupError, match="unconfirmed"):
        owner.remove_group("policy", elastic)
    assert runtime.removed == []
    elastic.all_engines = [None, None]

    owner.remove_group("policy", elastic)
    owner.remove_group("policy", elastic)

    assert runtime.removed == ["elastic"]
    assert elastic not in server.engine_groups
