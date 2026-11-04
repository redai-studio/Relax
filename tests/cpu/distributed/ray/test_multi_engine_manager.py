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
    def __init__(self, name: str, *, dead_methods: frozenset = frozenset()):
        self.name = name
        self.dead_methods = dead_methods
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

    with pytest.raises(RuntimeError, match="could not be rebuilt"):
        manager.recover()


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

    assert manager.recover() == set()

    assert removed_pgs == ["shared-pg"]
    assert manager.all_engines == [None, None, *healthy_engines]
    assert set(manager._engine_placements) == {2, 3}
    assert all(engine not in _patch_ray.killed for engine in healthy_engines)
