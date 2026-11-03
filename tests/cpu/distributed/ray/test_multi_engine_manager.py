# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Core lifecycle behavior of ``MultiEngineManager``, exercised without a Ray
cluster: fake engine handles stand in for Ray ObjectRefs, and ``ray.get``/
``ray.kill`` are patched onto the module directly.

This is the shared skeleton behind both ``GenRMManager`` and
``TeacherManager``, so a regression here silently breaks both judge serving
and OPD teacher recovery/offload-onload.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


try:
    import ray  # noqa: F401

    from relax.distributed.ray.multi_engine_manager import MultiEngineManager

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    MultiEngineManager = object


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="requires ray")


class _RemoteCall:
    """Fakes ``engine.method.remote()``: returns a token that the patched
    ``ray.get`` resolves to whatever the engine was configured to return (or
    raises, for a dead engine)."""

    def __init__(self, engine: "_FakeEngine", method: str):
        self._engine = engine
        self._method = method

    def remote(self, **kwargs):
        return (self._engine, self._method)


class _FakeEngine:
    def __init__(
        self,
        name: str,
        *,
        dead_methods: frozenset = frozenset(),
        actor_dead_methods: frozenset = frozenset(),
    ):
        self.name = name
        self.dead_methods = dead_methods
        self.actor_dead_methods = actor_dead_methods
        self.calls: list[str] = []

    def __getattr__(self, method: str):
        return _RemoteCall(self, method)


class _FakeManager(MultiEngineManager):
    """A minimal concrete manager: one engine per rank, no placement group
    (hooks return sentinel values that the test never inspects)."""

    def __init__(
        self, num_slots: int, *, nodes_per_engine: int = 1, owns_pg: bool = False, log_prefix: str = "[fake]"
    ):
        self._made: list[_FakeEngine] = []
        self._dead_at_init: set[int] = set()
        self._removed_pg = False
        self._instance_owns_pg = owns_pg
        super().__init__(
            SimpleNamespace(),
            num_slots=num_slots,
            nodes_per_engine=nodes_per_engine,
            engine_actor_cls=_FakeEngineActorCls,
            log_prefix=log_prefix,
        )

    def _resolve_placement(self, rank):
        return ((f"pg-{rank}", [0], [0]), self._instance_owns_pg, 0)

    def _ray_resource_kwargs(self, rank):
        return {}

    def _allocate_engine_addr_and_ports(self, *, new_engines):
        return {rank: {"host": "h", "port": 1} for rank, _ in new_engines}

    def _build_engine_env_vars(self):
        return {}


class _FakeEngineActorCls:
    """Stand-in engine "actor class"; ``ray.remote(cls)`` in the base class
    just needs something ``.options(...).remote(...)`` works on."""


@pytest.fixture(autouse=True)
def _patch_ray(monkeypatch):
    """Patch the ray module used by multi_engine_manager: ``ray.remote`` wraps
    our fake class into something whose ``.options().remote()`` returns a
    _FakeEngine; ``ray.get`` resolves _RemoteCall tokens; ``ray.kill`` is a no-
    op recorder."""
    import relax.distributed.ray.multi_engine_manager as mem

    created: list[_FakeEngine] = []
    killed: list[_FakeEngine] = []
    state = SimpleNamespace(created=created, killed=killed, fail_create_rank=None)

    def fake_remote(cls):
        if cls is not _FakeEngineActorCls:
            return cls  # pass through decorators applied to real classes elsewhere

        class _Options:
            @staticmethod
            def options(**kwargs):
                class _Ctor:
                    @staticmethod
                    def remote(args, *, rank, worker_type, base_gpu_id, **ctor_kwargs):
                        if rank == state.fail_create_rank:
                            raise RuntimeError("actor creation failed")
                        engine = _FakeEngine(f"engine-{rank}")
                        created.append(engine)
                        return engine

                return _Ctor

        return _Options

    def fake_get(handle_or_list, timeout=None):
        if isinstance(handle_or_list, list):
            return [fake_get(h) for h in handle_or_list]
        engine, method = handle_or_list
        if method in engine.actor_dead_methods:
            raise RuntimeError(f"{engine.name} actor is gone")
        if method in engine.dead_methods:
            raise ConnectionError(f"{engine.name} is dead for {method}")
        engine.calls.append(method)
        return True

    def fake_kill(engine):
        killed.append(engine)

    monkeypatch.setattr(mem.ray, "remote", fake_remote)
    monkeypatch.setattr(mem.ray, "get", fake_get)
    monkeypatch.setattr(mem.ray, "kill", fake_kill)
    monkeypatch.setattr(mem.ray.exceptions, "RayActorError", RuntimeError, raising=False)
    monkeypatch.setattr(
        sys.modules["ray.util.placement_group"], "placement_group_table", lambda pg: {"state": "REMOVED"}
    )

    yield state


def test_fanout_isolates_one_dead_engine_from_the_rest(_patch_ray):
    manager = _FakeManager(num_slots=3)
    for engine in manager.all_engines:
        engine.calls.clear()  # drop the init() calls made during construction
    dead_engine = manager.all_engines[1]
    dead_engine.dead_methods = frozenset({"release_memory_occupation"})

    dead_ranks = manager._fanout("release_memory_occupation")

    assert dead_ranks == [1]
    # The other two engines still got the call.
    assert manager.all_engines[0].calls == ["release_memory_occupation"]
    assert manager.all_engines[2].calls == ["release_memory_occupation"]


def test_fanout_reraises_non_dead_exceptions(_patch_ray, monkeypatch):
    import relax.distributed.ray.multi_engine_manager as mem

    manager = _FakeManager(num_slots=1)

    def raise_value_error(handle_or_list, timeout=None):
        raise ValueError("a real bug, not a dead engine")

    monkeypatch.setattr(mem.ray, "get", raise_value_error)

    with pytest.raises(ValueError, match="a real bug"):
        manager._fanout("release_memory_occupation")


def test_offload_onload_are_idempotent_when_state_unchanged(_patch_ray):
    manager = _FakeManager(num_slots=2)
    for engine in manager.all_engines:
        engine.calls.clear()  # drop the init() calls made during construction
    assert manager.is_onloaded()

    manager.offload()
    assert not manager.is_onloaded()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation"]

    # A second offload while already offloaded must not re-fire the RPC.
    manager.offload()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation"]

    manager.onload()
    assert manager.is_onloaded()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation", "resume_memory_occupation"]

    # A second onload (no tags) while already onloaded must not re-fire the RPC.
    manager.onload()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation", "resume_memory_occupation"]


def test_retire_engines_kills_and_nulls_the_slot(_patch_ray):
    manager = _FakeManager(num_slots=2)
    dead_engine = manager.all_engines[0]

    manager._retire_engines([0])

    assert manager.all_engines[0] is None
    assert manager.all_engines[1] is not None  # untouched
    assert dead_engine in _patch_ray.killed


def test_dead_ray_actor_is_acknowledged_and_recoverable(_patch_ray, monkeypatch):
    placement_group = sys.modules["ray.util.placement_group"]
    removed = []
    monkeypatch.setattr(placement_group, "remove_placement_group", removed.append)
    manager = _FakeManager(num_slots=1, owns_pg=True)
    dead_engine = manager.all_engines[0]
    dead_engine.actor_dead_methods = frozenset({"shutdown"})

    # The fixture represents RayActorError as RuntimeError so this remains a
    # dependency-light test while exercising the real dead-actor branch.
    manager._cleanup_slots([0])

    assert manager.all_engines == [None]
    assert not manager._cleanup_pending
    assert removed == ["pg-0"]
    assert dead_engine in _patch_ray.killed

    rebuilt = manager.recover()
    assert rebuilt == {0}
    assert manager.all_engines[0] is not dead_engine


def test_recover_rebuilds_only_the_dead_slot(_patch_ray):
    manager = _FakeManager(num_slots=2)
    original_engine_1 = manager.all_engines[1]
    manager._retire_engines([0])

    rebuilt = manager.recover()

    assert rebuilt == {0}
    assert manager.all_engines[0] is not None
    assert manager.all_engines[0] is not original_engine_1
    assert manager.all_engines[1] is original_engine_1  # untouched


def test_recover_raises_on_total_wipeout(_patch_ray, monkeypatch):
    import relax.distributed.ray.multi_engine_manager as mem

    manager = _FakeManager(num_slots=1)
    manager._retire_engines([0])

    def failing_options(**kwargs):
        class _Ctor:
            @staticmethod
            def remote(*args, **kwargs2):
                raise RuntimeError("scheduling failed, node is gone")

        return _Ctor

    class _FailingActor:
        options = staticmethod(failing_options)

    monkeypatch.setattr(mem.ray, "remote", lambda cls: _FailingActor)

    with pytest.raises(RuntimeError, match="scheduling failed"):
        manager.recover()


def test_recover_failure_with_live_peer_closes_discovery(_patch_ray, monkeypatch):
    manager = _FakeManager(num_slots=2)
    manager.all_engines[0] = None

    def fail_ports(self, *, new_engines):
        raise RuntimeError("replacement placement unavailable")

    monkeypatch.setattr(_FakeManager, "_allocate_engine_addr_and_ports", fail_ports)

    with pytest.raises(RuntimeError, match="replacement placement unavailable"):
        manager.recover()

    assert manager.all_engines[1] is not None
    assert manager.inference.state == "failed"
    snapshot = manager.get_inference_snapshot()
    assert snapshot["models"]["default"]["state"] == "failed"
    assert all(not engine["direct_eligible"] for engine in snapshot["models"]["default"]["engines"])


def test_shutdown_removes_owned_placement_group_but_not_borrowed_one(_patch_ray, monkeypatch):
    # ray.util.placement_group is shadowed as an attribute by a same-named
    # function on the ray.util package, so it must be patched via sys.modules
    # (where the actual submodule lives) rather than a dotted monkeypatch path.
    placement_group_submodule = sys.modules["ray.util.placement_group"]
    removed_pgs = []
    monkeypatch.setattr(placement_group_submodule, "remove_placement_group", lambda pg: removed_pgs.append(pg))

    owning_manager = _FakeManager(num_slots=1, owns_pg=True)
    owning_manager.shutdown()
    assert removed_pgs == ["pg-0"]

    removed_pgs.clear()
    borrowing_manager = _FakeManager(num_slots=1, owns_pg=False)
    borrowing_manager.shutdown()
    assert removed_pgs == []


def test_startup_port_failure_rolls_back_all_candidates(_patch_ray, monkeypatch):
    def failed_ports(self, **kwargs):
        raise RuntimeError("port allocation failed")

    monkeypatch.setattr(_FakeManager, "_allocate_engine_addr_and_ports", failed_ports)
    with pytest.raises(RuntimeError, match="port allocation"):
        _FakeManager(num_slots=3)
    assert _patch_ray.killed == _patch_ray.created
    assert all(e.calls == ["shutdown"] for e in _patch_ray.created)


def test_multinode_discovery_exposes_heads_and_shutdown_kills_followers(_patch_ray):
    manager = _FakeManager(num_slots=4)
    manager.nodes_per_engine = 2
    first = manager.get_inference_snapshot()
    assert len(first["models"]["default"]["engines"]) == 2
    manager.offload()
    sleeping = manager.get_inference_snapshot()
    assert all(not e["direct_eligible"] for e in sleeping["models"]["default"]["engines"])
    assert sleeping["topology_revision"] > first["topology_revision"]
    manager.shutdown()
    manager.shutdown()
    assert len(_patch_ray.killed) == 4
    assert all(e is None for e in manager.all_engines)


def test_partial_onload_full_resume_and_rollback(_patch_ray):
    manager = _FakeManager(num_slots=1)
    engine = manager.all_engines[0]
    manager.offload()
    manager.onload(tags=["weights"])
    assert manager.get_inference_snapshot()["models"]["default"]["state"] == "onloading"
    manager.onload()
    assert engine.calls.count("resume_memory_occupation") == 2
    manager.offload()
    manager.onload(tags=["weights"])
    manager.offload()
    assert engine.calls.count("release_memory_occupation") == 3


@pytest.mark.parametrize("failed_operation", ["shutdown", "kill"])
def test_failed_retirement_preserves_handle_and_owned_pg_until_retry(_patch_ray, monkeypatch, failed_operation):
    import relax.distributed.ray.multi_engine_manager as mem

    removed = []
    monkeypatch.setattr(sys.modules["ray.util.placement_group"], "remove_placement_group", removed.append)
    manager = _FakeManager(num_slots=1, owns_pg=True)
    engine = manager.all_engines[0]
    engine.dead_methods = frozenset({"release_memory_occupation", "shutdown"})
    if failed_operation == "kill":
        engine.dead_methods = frozenset({"release_memory_occupation"})
        monkeypatch.setattr(mem.ray, "kill", lambda actor: (_ for _ in ()).throw(OSError("kill failed")))

    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        manager.offload()
    assert manager.inference.state == "failed"
    assert manager.all_engines == [engine]
    assert manager._cleanup_pending == {0}
    assert manager._engine_placements and not removed
    assert not _patch_ray.killed
    with pytest.raises(RuntimeError, match="cleanup pending"):
        manager.onload()
    with pytest.raises(RuntimeError, match="cleanup pending"):
        manager.recover()

    engine.dead_methods = frozenset()
    monkeypatch.setattr(mem.ray, "kill", _patch_ray.killed.append)
    manager.offload()
    assert manager.inference.state == "sleeping"
    assert manager.all_engines == [None]
    assert not manager._cleanup_pending
    assert removed == ["pg-0"]


def test_offload_retries_pending_follower_after_head_is_removed(_patch_ray):
    manager = _FakeManager(num_slots=2)
    manager.nodes_per_engine = 2
    head, follower = manager.all_engines
    head.dead_methods = frozenset({"release_memory_occupation"})
    follower.dead_methods = frozenset({"shutdown"})
    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        manager.offload()
    assert manager.all_engines == [None, follower]
    assert manager._cleanup_pending == {1}
    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        manager.offload()
    assert manager.inference.state == "failed"
    follower.dead_methods = frozenset()
    manager.offload()
    assert manager.inference.state == "sleeping"
    assert not manager._cleanup_pending
    assert _patch_ray.killed == [head, follower]


async def test_failed_manager_cleanup_retains_phase_and_blocks_next_model(_patch_ray):
    from relax.inference.lifecycle import LifecycleCoordinator

    manager = _FakeManager(num_slots=1)
    manager.all_engines[0].dead_methods = frozenset({"release_memory_occupation", "shutdown"})
    coordinator = LifecycleCoordinator()

    async def idle():
        return None

    async def release():
        manager.offload()

    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        await coordinator.run_phase("teacher", idle, release, idle)
    assert coordinator.phase == "teacher"
    with pytest.raises(RuntimeError, match="lease retained"):
        await coordinator.run_phase("genrm", idle, idle, idle)


def test_failed_pg_removal_keeps_ownership_for_shutdown_retry(_patch_ray, monkeypatch):
    placement = sys.modules["ray.util.placement_group"]
    manager = _FakeManager(num_slots=1, owns_pg=True)
    monkeypatch.setattr(placement, "remove_placement_group", lambda pg: (_ for _ in ()).throw(OSError("PG failed")))
    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        manager.shutdown()
    assert manager.all_engines == [None]
    assert 0 in manager._engine_placements
    assert manager._cleanup_pending == {0}
    removed = []
    monkeypatch.setattr(placement, "remove_placement_group", removed.append)
    manager.shutdown()
    assert manager.inference.state == "dead"
    assert not manager._engine_placements and not manager._cleanup_pending
    assert removed == ["pg-0"]


def test_startup_cleanup_failure_does_not_skip_other_candidates(_patch_ray, monkeypatch):
    def fail_ports(self, *, new_engines):
        new_engines[0][1].dead_methods = frozenset({"shutdown"})
        raise RuntimeError("port allocation failed")

    monkeypatch.setattr(_FakeManager, "_allocate_engine_addr_and_ports", fail_ports)
    manager = _FakeManager.__new__(_FakeManager)
    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        manager.__init__(num_slots=3)
    assert manager.all_engines == [_patch_ray.created[0], None, None]
    assert _patch_ray.killed == _patch_ray.created[1:]
    assert manager._cleanup_pending == {0}


def test_recovery_never_counts_uncleaned_startup_as_rebuilt(_patch_ray, monkeypatch):
    manager = _FakeManager(num_slots=1)
    manager._retire_engines([0])

    def fail_ports(self, *, new_engines):
        new_engines[0][1].dead_methods = frozenset({"shutdown"})
        raise RuntimeError("port allocation failed")

    monkeypatch.setattr(_FakeManager, "_allocate_engine_addr_and_ports", fail_ports)
    with pytest.raises(RuntimeError, match="cleanup pending"):
        manager.recover()
    assert manager.inference.state == "failed"
    assert manager._cleanup_pending == {0}


def test_shutdown_cleans_follower_slots_and_preserves_borrowed_placement_group(_patch_ray, monkeypatch):
    removed_pgs = []
    monkeypatch.setattr(sys.modules["ray.util.placement_group"], "remove_placement_group", removed_pgs.append)
    manager = _FakeManager(num_slots=4, nodes_per_engine=2)
    engines = list(manager.all_engines)
    for engine in engines:
        engine.calls.clear()

    manager.shutdown()

    assert all(engine.calls == ["shutdown"] for engine in engines)
    assert _patch_ray.killed == engines
    assert manager.all_engines == [None] * 4
    assert manager._engine_placements == {}
    assert removed_pgs == []


def test_offload_onload_rebuilds_whole_multinode_group_and_preserves_other_group(_patch_ray):
    manager = _FakeManager(num_slots=4, nodes_per_engine=2)
    original_engines = list(manager.all_engines)
    for engine in original_engines:
        engine.calls.clear()
    original_engines[0].dead_methods = frozenset({"release_memory_occupation"})

    manager.offload()

    assert manager.all_engines == [None, None, *original_engines[2:]]
    assert _patch_ray.killed == original_engines[:2]
    assert all("shutdown" in engine.calls for engine in original_engines[:2])

    manager.onload()

    for rank in (0, 1):
        assert manager.all_engines[rank] is not original_engines[rank]
        assert manager.all_engines[rank].calls == ["init"]
    assert manager.all_engines[2:] == original_engines[2:]
    assert original_engines[2].calls == ["release_memory_occupation", "resume_memory_occupation"]
    assert original_engines[3].calls == []


def test_multinode_shutdown_stops_followers_before_removing_owned_group(_patch_ray, monkeypatch):
    manager = _FakeManager(num_slots=2, nodes_per_engine=2)
    engines = list(manager.all_engines)
    shared_pg = ("shared-pg", [0, 1], [0, 0])
    manager._engine_placements = {0: (shared_pg, True), 1: (shared_pg, False)}
    removed_pgs = []

    def remove_pg(pg):
        assert all(engine in _patch_ray.killed for engine in engines)
        assert all("shutdown" in engine.calls for engine in engines)
        removed_pgs.append(pg)

    monkeypatch.setattr(sys.modules["ray.util.placement_group"], "remove_placement_group", remove_pg)
    manager.shutdown()

    assert removed_pgs == ["shared-pg"]
    assert manager.all_engines == [None, None]
    assert manager._engine_placements == {}


@pytest.mark.parametrize("failure", ["actor_creation", "address", "init"])
def test_multinode_recovery_failure_cleans_group_and_preserves_healthy_engine(_patch_ray, monkeypatch, failure):
    import relax.distributed.ray.multi_engine_manager as mem

    manager = _FakeManager(num_slots=4, nodes_per_engine=2)
    healthy_engines = manager.all_engines[2:]
    manager.all_engines[0].dead_methods = frozenset({"release_memory_occupation"})
    manager.offload()
    _patch_ray.created.clear()
    _patch_ray.killed.clear()
    shared_pg = ("shared-pg", [0, 1], [0, 0])
    removed_pgs = []

    def resolve_placement(rank):
        return shared_pg, rank == 0, rank

    def remove_pg(pg):
        assert all(engine in _patch_ray.killed for engine in _patch_ray.created)
        assert all("shutdown" in engine.calls for engine in _patch_ray.created)
        removed_pgs.append(pg)

    monkeypatch.setattr(manager, "_resolve_placement", resolve_placement)
    monkeypatch.setattr(sys.modules["ray.util.placement_group"], "remove_placement_group", remove_pg)
    if failure == "actor_creation":
        _patch_ray.fail_create_rank = 0
    elif failure == "address":

        def fail_address(**kwargs):
            raise ConnectionError("engine unavailable during port allocation")

        monkeypatch.setattr(manager, "_allocate_engine_addr_and_ports", fail_address)
    elif failure == "init":
        original_get = mem.ray.get

        def fail_init(handle_or_list, timeout=None):
            if isinstance(handle_or_list, list):
                raise RuntimeError("follower initialization failed")
            return original_get(handle_or_list, timeout=timeout)

        monkeypatch.setattr(mem.ray, "get", fail_init)

    with pytest.raises((RuntimeError, ConnectionError)):
        manager.recover()

    assert removed_pgs == ["shared-pg"]
    assert manager.all_engines == [None, None, *healthy_engines]
    assert set(manager._engine_placements) == {2, 3}
    assert all(engine not in _patch_ray.killed for engine in healthy_engines)


def test_owned_head_transfers_pg_to_failed_borrowed_follower_until_retry(_patch_ray, monkeypatch):
    manager = _FakeManager(num_slots=2, nodes_per_engine=2)
    head, follower = manager.all_engines
    pg = ("shared-pg", [0, 1], [0, 0])
    manager._engine_placements = {0: (pg, True), 1: (pg, False)}
    follower.dead_methods = frozenset({"shutdown"})
    removed = []
    monkeypatch.setattr(sys.modules["ray.util.placement_group"], "remove_placement_group", removed.append)
    with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
        manager.shutdown()
    assert manager.all_engines == [None, follower]
    assert manager._engine_placements == {1: (pg, True)}
    assert removed == []
    follower.dead_methods = frozenset()
    manager.shutdown()
    assert removed == ["shared-pg"]
    assert manager._engine_placements == {}
    assert _patch_ray.killed == [head, follower]
