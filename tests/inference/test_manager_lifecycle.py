# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from types import SimpleNamespace

import pytest

from relax.distributed.ray import inference_manager as module


class _Method:
    def __init__(self, actor, name):
        self.actor = actor
        self.name = name

    def remote(self, **kwargs):
        if (self.actor.rank, self.name) in self.actor.runtime.submit_failures:
            raise ValueError(f"{self.name} submission failure")
        self.actor.runtime.events.append(("submit", self.actor.rank, self.name, kwargs))
        return self.actor, self.name, kwargs


class _Actor:
    def __init__(self, runtime, rank):
        self.runtime = runtime
        self.rank = rank

    def __getattr__(self, name):
        return _Method(self, name)


class _Pool(module.InferenceManager):
    def __init__(self, runtime, *, slots=2, nodes=1, owned=True, shared=True):
        self.runtime = runtime
        self.owned = owned
        self.shared = shared
        self._inference_preserves_weights = True
        super().__init__(
            SimpleNamespace(), num_slots=slots, nodes_per_engine=nodes, engine_actor_cls=_Actor, skip_init=True
        )

    def _resolve_placement(self, rank):
        if self.runtime.failpoint == ("placement", rank):
            raise ValueError("placement failure")
        pg = "shared-pg" if self.shared else f"pg-{rank}"
        return (pg, [0], [0]), self.owned, 0

    def _ray_resource_kwargs(self, rank):
        return {}

    def _build_engine_env_vars(self):
        return {}

    def _allocate_engine_addr_and_ports(self, *, new_engines):
        if self.runtime.failpoint == ("ports", 0):
            raise ValueError("ports failure")
        return {rank: {"host": "engine.example", "port": 15000 + rank} for rank, _engine in new_engines}


@pytest.fixture
def runtime(monkeypatch):

    runtime = SimpleNamespace(
        events=[],
        failpoint=None,
        submit_failures=set(),
        rpc_failures={},
        kill_failures=set(),
        removed=[],
        remove_failures=0,
        timeouts=[],
    )

    class ActorClass:
        @staticmethod
        def options(**kwargs):
            return ActorClass

        @staticmethod
        def remote(args, *, rank, **kwargs):
            if runtime.failpoint == ("actor", rank):
                raise ValueError("actor failure")
            actor = _Actor(runtime, rank)
            runtime.events.append(("create", rank))
            return actor

    def get(ref, timeout=None):
        runtime.timeouts.append(timeout)
        if isinstance(ref, list):
            return [get(item, timeout=timeout) for item in ref]
        actor, name, kwargs = ref
        failure = runtime.rpc_failures.get((actor.rank, name))
        if failure is not None:
            raise failure
        runtime.events.append(("ack", actor.rank, name, kwargs))
        if name == "get_url":
            return f"http://engine-{actor.rank}.example:15000"
        return True

    def kill(actor):
        if actor.rank in runtime.kill_failures:
            raise ValueError("kill failed")
        runtime.events.append(("kill", actor.rank))

    def remove(pg):
        if runtime.remove_failures:
            runtime.remove_failures -= 1
            raise ValueError("PG removal failed")
        runtime.removed.append(pg)
        runtime.events.append(("remove", pg))

    monkeypatch.setattr(module.ray, "remote", lambda cls: ActorClass)
    monkeypatch.setattr(module.ray, "get", get)
    monkeypatch.setattr(module.ray, "kill", kill)
    monkeypatch.setattr(module, "remove_placement_group", remove)
    return runtime


@pytest.mark.parametrize("phase", ["placement", "actor", "ports", "init_submission", "warmup"])
def test_initialization_failure_rolls_back_every_acquired_resource(runtime, phase):
    pool = _Pool(runtime, shared=False)
    if phase == "init_submission":
        runtime.submit_failures.add((1, "init"))
    elif phase == "warmup":
        runtime.rpc_failures[(1, "init")] = ValueError("warmup failure")
    else:
        runtime.failpoint = (phase, 0 if phase == "ports" else 1)

    with pytest.raises(ValueError, match="failure"):
        pool._init_engines([0, 1])

    expected = ["pg-0"] if phase == "placement" else ["pg-0", "pg-1"]
    assert runtime.removed == expected
    assert pool.all_engines == [None, None]
    assert not pool._engine_placements
    assert not pool._cleanup_pending
    shutdown_ack = [i for i, event in enumerate(runtime.events) if event[:3:2] == ("ack", "shutdown")]
    kills = [i for i, event in enumerate(runtime.events) if event[0] == "kill"]
    assert shutdown_ack and max(shutdown_ack) < min(kills)


def test_multinode_shutdown_stops_all_workers_before_kills_and_removes_pg_once(runtime):
    pool = _Pool(runtime, slots=4, nodes=2)
    pool._init_engines(list(range(4)))
    runtime.events.clear()

    pool.shutdown()
    pool.shutdown()

    assert runtime.removed == ["shared-pg"]
    assert sorted(event[1] for event in runtime.events if event[0] == "kill") == [0, 1, 2, 3]
    assert pool.all_engines == [None] * 4
    last_shutdown = max(i for i, event in enumerate(runtime.events) if len(event) > 2 and event[2] == "shutdown")
    first_kill = min(i for i, event in enumerate(runtime.events) if event[0] == "kill")
    assert last_shutdown < first_kill


def test_borrowed_pg_survives_startup_rollback_and_shutdown(runtime):
    pool = _Pool(runtime, owned=False)
    runtime.failpoint = ("ports", 0)
    with pytest.raises(ValueError):
        pool._init_engines([0, 1])
    pool.shutdown()
    assert runtime.removed == []


def test_pg_ever_borrowed_cannot_be_deleted_by_a_later_conflicting_owner_record(runtime, monkeypatch):
    pool = _Pool(runtime, slots=2)
    monkeypatch.setattr(pool, "_resolve_placement", lambda rank: (("shared-pg", [0], [0]), rank == 1, 0))
    pool._init_engines([0, 1])

    pool.shutdown()

    assert runtime.removed == []


def test_partial_restore_resumes_only_missing_tags_and_opens_admission_after_completion(runtime):
    pool = _Pool(runtime, slots=1)
    pool._init_engines([0])
    runtime.events.clear()
    pool.offload()
    pool.onload(tags=["weights"])
    assert not pool.is_onloaded()
    pool.onload(tags=["weights"])
    pool.onload()
    pool.onload()

    resumes = [event[3]["tags"] for event in runtime.events if event[:3:2] == ("submit", "resume_memory_occupation")]
    assert resumes == [["weights"], ["cuda_graph", "kv_cache"]]
    assert pool.is_onloaded()
    assert sum(event[:3:2] == ("submit", "continue_generation") for event in runtime.events) == 1
    assert all(timeout is not None and timeout > 0 for timeout in runtime.timeouts)


def test_partial_offload_is_not_skipped_when_pool_is_not_fully_onloaded(runtime):
    pool = _Pool(runtime, slots=1)
    pool._init_engines([0])
    pool.offload()
    pool.onload(tags=["weights"])
    assert not pool.is_onloaded()

    pool.offload()

    releases = [event for event in runtime.events if event[:3:2] == ("submit", "release_memory_occupation")]
    assert len(releases) == 2
    entry = next(iter(pool._inference_observation.entries.values()))
    assert entry["state"] == "SLEEPING"


def test_retry_after_mixed_offload_failure_skips_already_released_replica(runtime):
    pool = _Pool(runtime, slots=2, owned=False)
    pool._init_engines([0, 1])
    runtime.rpc_failures[(1, "release_memory_occupation")] = ValueError("release failed")

    with pytest.raises(ValueError, match="release failed"):
        pool.offload()

    entries = pool._inference_observation.entries
    assert entries["__default__/replica-0"]["state"] == "SLEEPING"
    assert entries["__default__/replica-1"]["state"] == "FAILED"
    runtime.rpc_failures.clear()
    pool.offload()
    releases = [event[1] for event in runtime.events if event[:3:2] == ("submit", "release_memory_occupation")]
    assert releases.count(0) == 1
    assert releases.count(1) == 2


def test_onload_after_unconfirmed_offload_cannot_reopen_admission(runtime):
    pool = _Pool(runtime, slots=1, owned=False)
    pool._init_engines([0])
    runtime.rpc_failures[(0, "release_memory_occupation")] = ValueError("release failed")
    with pytest.raises(ValueError):
        pool.offload()
    runtime.rpc_failures.clear()

    with pytest.raises(RuntimeError, match="Cannot onload"):
        pool.onload()

    assert not any(event[:3:2] == ("submit", "continue_generation") for event in runtime.events)


def test_failed_partial_resume_rolls_back_before_later_recovery(runtime):
    pool = _Pool(runtime, slots=1)
    pool._init_engines([0])
    pool.offload()
    original = pool.all_engines[0]
    runtime.rpc_failures[(0, "resume_memory_occupation")] = ValueError("partial resume failed")

    with pytest.raises(ValueError, match="partial resume failed"):
        pool.onload(tags=["weights"])

    assert pool.all_engines == [None]
    assert not pool._cleanup_pending
    runtime.rpc_failures.clear()
    pool.onload()
    assert pool.all_engines[0] is not original
    assert pool.is_onloaded()


def test_failed_drain_and_failed_shutdown_keep_resources_until_cleanup_retry(runtime):
    pool = _Pool(runtime, slots=1)
    pool._init_engines([0])
    engine = pool.all_engines[0]
    runtime.rpc_failures[(0, "release_memory_occupation")] = TimeoutError("drain timed out")
    runtime.rpc_failures[(0, "shutdown")] = TimeoutError("process unconfirmed")

    with pytest.raises(module.InferenceCleanupError, match="unconfirmed"):
        pool.offload()

    assert pool.all_engines[0] is engine
    assert pool._cleanup_pending == {0}
    assert runtime.removed == []
    assert not any(event[0] == "kill" for event in runtime.events)
    assert next(iter(pool._inference_observation.entries.values()))["state"] == "FAILED"
    runtime.rpc_failures.clear()
    pool.shutdown()
    assert pool.all_engines == [None]
    assert runtime.removed == ["shared-pg"]


def test_kill_retry_does_not_repeat_confirmed_shutdown(runtime):
    pool = _Pool(runtime, slots=1)
    pool._init_engines([0])
    runtime.kill_failures.add(0)
    with pytest.raises(module.InferenceCleanupError):
        pool.shutdown()
    runtime.kill_failures.clear()

    pool.shutdown()

    assert sum(event[:3:2] == ("submit", "shutdown") for event in runtime.events) == 1
    assert runtime.removed == ["shared-pg"]


def test_pg_remove_failure_retains_record_for_retry(runtime):
    pool = _Pool(runtime, slots=1)
    pool._init_engines([0])
    runtime.remove_failures = 1
    with pytest.raises(module.InferenceCleanupError):
        pool.shutdown()
    assert pool.all_engines == [None]
    assert pool._engine_placements

    pool.shutdown()

    assert runtime.removed == ["shared-pg"]
    assert not pool._engine_placements


def test_follower_recovery_rebuilds_whole_replica_without_touching_other_replica(runtime):
    pool = _Pool(runtime, slots=4, nodes=2, owned=False)
    pool._init_engines(list(range(4)))
    survivor = pool.all_engines[2]
    old_head = pool.all_engines[0]
    pool.all_engines[1] = None

    rebuilt = pool.recover()

    assert rebuilt == {0, 1}
    assert pool.all_engines[0] is not old_head
    assert pool.all_engines[2] is survivor
    assert runtime.removed == []


def test_failed_recovery_does_not_reclassify_untouched_sleeping_replica(runtime):
    pool = _Pool(runtime, slots=2, owned=False)
    pool._init_engines([0, 1])
    pool.offload()
    pool._retire_engines([0])
    runtime.failpoint = ("actor", 0)

    assert pool.recover() == set()

    entries = pool._inference_observation.entries
    assert entries["__default__/replica-1"]["state"] == "SLEEPING"


@pytest.mark.parametrize("phase", ["submission", "result"])
def test_dead_actor_shutdown_releases_slot_for_recovery(runtime, monkeypatch, phase):
    pool = _Pool(runtime, slots=2, owned=False)
    pool._init_engines([0, 1])
    dead = pool.all_engines[0]
    if phase == "submission":
        original_remote = _Method.remote

        def dead_submission(self, **kwargs):
            if self.actor is dead and self.name == "shutdown":
                raise module.ray.exceptions.ActorDiedError()
            return original_remote(self, **kwargs)

        monkeypatch.setattr(_Method, "remote", dead_submission)
    else:
        runtime.rpc_failures[(0, "shutdown")] = module.ray.exceptions.ActorDiedError()

    pool._retire_engines([0])
    runtime.rpc_failures.clear()

    assert pool.all_engines[0] is None
    assert not pool._cleanup_pending
    assert pool.recover() == {0}
    assert pool.all_engines[0] is not None and pool.all_engines[0] is not dead


def test_unavailable_actor_shutdown_remains_unconfirmed(runtime):
    pool = _Pool(runtime, slots=1, owned=False)
    pool._init_engines([0])
    engine = pool.all_engines[0]
    runtime.rpc_failures[(0, "shutdown")] = module.ray.exceptions.ActorUnavailableError("restarting", None)

    with pytest.raises(module.InferenceCleanupError, match="unconfirmed"):
        pool._retire_engines([0])

    assert pool.all_engines[0] is engine
    assert pool._cleanup_pending == {0}


def test_health_check_covers_followers_and_retires_dead_replica(runtime):
    pool = _Pool(runtime, slots=4, nodes=2, owned=False)
    pool._init_engines(list(range(4)))
    survivor = pool.all_engines[2:]
    assert pool.health_check()
    probed = {(event[1], event[2]) for event in runtime.events if event[0] == "submit"}
    assert {(1, "health_process"), (3, "health_process")} <= probed
    assert (1, "health_generate") not in probed

    runtime.rpc_failures[(1, "health_process")] = module.ray.exceptions.ActorDiedError()
    runtime.rpc_failures[(1, "shutdown")] = module.ray.exceptions.ActorDiedError()
    assert not pool.health_check()

    assert pool.all_engines[:2] == [None, None]
    assert pool.all_engines[2:] == survivor
    runtime.rpc_failures.clear()
    assert pool.recover() == {0, 1}


def test_health_check_transient_failure_does_not_retire_replica(runtime):
    pool = _Pool(runtime, slots=2, nodes=2, owned=False)
    pool._init_engines([0, 1])
    workers = list(pool.all_engines)
    runtime.rpc_failures[(0, "health_generate")] = TimeoutError("busy")

    assert not pool.health_check()

    assert pool.all_engines == workers
