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

    def __init__(self, num_slots: int, *, owns_pg: bool = False, log_prefix: str = "[fake]"):
        self._made: list[_FakeEngine] = []
        self._dead_at_init: set[int] = set()
        self._removed_pg = False
        self._instance_owns_pg = owns_pg
        super().__init__(
            SimpleNamespace(),
            num_slots=num_slots,
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

    def fake_remote(cls):
        if cls is not _FakeEngineActorCls:
            return cls  # pass through decorators applied to real classes elsewhere

        class _Options:
            @staticmethod
            def options(**kwargs):
                class _Ctor:
                    @staticmethod
                    def remote(args, *, rank, worker_type, base_gpu_id, **ctor_kwargs):
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

    yield SimpleNamespace(created=created, killed=killed)


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
    manager.all_engines[0] = None  # simulate a prior retirement

    rebuilt = manager.recover()

    assert rebuilt == {0}
    assert manager.all_engines[0] is not None
    assert manager.all_engines[0] is not original_engine_1
    assert manager.all_engines[1] is original_engine_1  # untouched


def test_recover_raises_on_total_wipeout(_patch_ray, monkeypatch):
    import relax.distributed.ray.multi_engine_manager as mem

    manager = _FakeManager(num_slots=1)
    manager.all_engines[0] = None

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


class _RecordingCall(_RemoteCall):
    def remote(self, **kwargs):
        self._engine.remote_kwargs[self._method] = kwargs
        return super().remote(**kwargs)


class _RecordingEngine(_FakeEngine):
    def __init__(self, name: str):
        super().__init__(name)
        self.remote_kwargs: dict[str, dict] = {}

    def __getattr__(self, method: str):
        return _RecordingCall(self, method)


class _HookedManager(MultiEngineManager):
    """Every hook returns a distinctive value so the test can see where it ends
    up."""

    def __init__(self):
        self.pg_tuple = ("shared-pg", [40, 41, 42, 43], [4, 5, 6, 7])
        super().__init__(
            SimpleNamespace(marker="args"),
            num_slots=2,
            engine_actor_cls=_FakeEngineActorCls,
            log_prefix="[hooked]",
        )

    def _resolve_placement(self, rank):
        return self.pg_tuple, False, rank * 2

    def _ray_resource_kwargs(self, rank):
        return {"num_cpus": 0.3, "num_gpus": 0.1}

    def _allocate_engine_addr_and_ports(self, *, new_engines):
        return {
            rank: {"host": "h", "port": 9000 + rank, "nccl_port": 9100 + rank, "dist_init_addr": f"h:{9200 + rank}"}
            for rank, _ in new_engines
        }

    def _build_engine_env_vars(self):
        return {"SOME_ENV": "1"}

    def _engine_ctor_args(self, rank):
        return ("ctor-args", rank)

    def _build_engine_ctor_kwargs(self, rank):
        return {"extra": rank}

    def _build_engine_init_kwargs(self, rank, addr_and_ports):
        return {**addr_and_ports, "skip_dcs_registration": True}


def test_multi_engine_manager_init_passes_hook_values_to_ray(_patch_ray, monkeypatch):
    """Characterization: where each subclass hook's value ends up when the
    base class brings engines up."""
    import relax.distributed.ray.multi_engine_manager as mem

    creations: list[dict] = []

    class _CapturingActor:
        @classmethod
        def options(cls, **options):
            creations.append({"options": options})
            return cls

        @classmethod
        def remote(cls, *args, **kwargs):
            engine = _RecordingEngine(f"engine-{kwargs['rank']}")
            creations[-1].update(ctor_args=args, ctor_kwargs=kwargs, engine=engine)
            return engine

    monkeypatch.setattr(mem.ray, "remote", lambda cls: _CapturingActor)

    manager = _HookedManager()

    assert len(creations) == 2
    first, second = creations
    # Ray resources and env vars come straight from the hooks.
    assert {key: first["options"][key] for key in ("num_cpus", "num_gpus")} == {"num_cpus": 0.3, "num_gpus": 0.1}
    assert first["options"]["runtime_env"] == {"env_vars": {"SOME_ENV": "1"}}
    # Placement: the bundle index and base GPU id are looked up at the hook's gpu_index.
    strategies = [creation["options"]["scheduling_strategy"] for creation in creations]
    assert [strategy.placement_group for strategy in strategies] == ["shared-pg", "shared-pg"]
    assert [strategy.placement_group_bundle_index for strategy in strategies] == [40, 42]
    assert all(strategy.placement_group_capture_child_tasks for strategy in strategies)
    # Constructor: hook args first, then rank / worker_type / base_gpu_id plus hook kwargs.
    assert first["ctor_args"] == (("ctor-args", 0),)
    assert first["ctor_kwargs"] == {"rank": 0, "worker_type": "regular", "base_gpu_id": 4, "extra": 0}
    assert second["ctor_kwargs"] == {"rank": 1, "worker_type": "regular", "base_gpu_id": 6, "extra": 1}
    # init(): the allocated address merged through the init-kwargs hook.
    assert second["engine"].remote_kwargs["init"] == {
        "host": "h",
        "port": 9001,
        "nccl_port": 9101,
        "dist_init_addr": "h:9201",
        "skip_dcs_registration": True,
    }
    assert manager.all_engines == [first["engine"], second["engine"]]
    assert manager._engine_addr_and_ports[1]["port"] == 9001


class _TwoNodeEngineManager(_FakeManager):
    """Two slots (nodes) per logical engine, each slot on its own host."""

    def __init__(self, num_slots: int):
        MultiEngineManager.__init__(
            self,
            SimpleNamespace(),
            num_slots=num_slots,
            nodes_per_engine=2,
            engine_actor_cls=_FakeEngineActorCls,
            log_prefix="[two-node]",
        )

    _instance_owns_pg = False

    def _allocate_engine_addr_and_ports(self, *, new_engines):
        return {rank: {"host": f"node-{rank}", "port": 9000 + rank} for rank, _ in new_engines}


def _named(manager):
    from relax.engine.inference.discovery import model_snapshot_from_payload

    return model_snapshot_from_payload("judge", manager.get_inference_snapshot())


def test_multi_engine_manager_snapshot_lists_only_head_engines(_patch_ray):
    manager = _TwoNodeEngineManager(num_slots=4)

    model = _named(manager)

    assert [engine.engine_id for engine in model.engines] == ["judge/0", "judge/1"]
    assert [engine.base_url for engine in model.engines] == ["http://node-0:9000", "http://node-2:9002"]
    assert all(engine.direct_eligible for engine in model.engines)
    assert model.state.value == "ready" and model.router_url is None


def test_multi_engine_manager_snapshot_revision_bumps_after_rebuild(_patch_ray):
    manager = _FakeManager(num_slots=2)
    initial = manager.get_inference_snapshot()["topology_revision"]
    assert manager.get_inference_snapshot()["topology_revision"] == initial

    manager._retire_engines([0])
    dead = manager.get_inference_snapshot()
    assert dead["engines"][0] == {"index": 0, "base_url": None, "state": "dead"}
    assert dead["topology_revision"] > initial

    manager.recover()
    rebuilt = manager.get_inference_snapshot()
    # Rebuilt on the same host/port, yet the revision still advances.
    assert rebuilt["engines"][0] == {"index": 0, "base_url": "http://h:1", "state": "ready"}
    assert rebuilt["topology_revision"] > dead["topology_revision"]


def test_multi_engine_manager_snapshot_reports_draining_during_offload(_patch_ray, monkeypatch):
    """A snapshot taken while an offload is in flight -- the manager actors
    answer it from a separate concurrency group -- must not advertise engines
    that are being drained."""
    import relax.distributed.ray.multi_engine_manager as mem

    manager = _FakeManager(num_slots=2)
    resolve = mem.ray.get
    seen: list[list[str]] = []

    def get_and_snapshot(handle_or_list, timeout=None):
        if not isinstance(handle_or_list, list) and handle_or_list[1] == "release_memory_occupation":
            model = _named(manager)
            seen.append([engine.state.value for engine in model.engines])
            assert not any(engine.direct_eligible for engine in model.engines)
        return resolve(handle_or_list, timeout=timeout)

    monkeypatch.setattr(mem.ray, "get", get_and_snapshot)

    manager.offload()

    assert seen == [["draining", "draining"], ["draining", "draining"]]
    assert [engine.state.value for engine in _named(manager).engines] == ["sleeping", "sleeping"]


def test_multi_engine_manager_snapshot_marks_offloaded_engines_not_eligible(_patch_ray):
    manager = _FakeManager(num_slots=2)
    before = manager.get_inference_snapshot()["topology_revision"]

    manager.offload()
    model = _named(manager)

    assert [engine.state.value for engine in model.engines] == ["sleeping", "sleeping"]
    assert not any(engine.direct_eligible for engine in model.engines)
    assert model.state.value == "sleeping"
    # Sleeping changes whether a replica can serve, not where it is.
    assert manager.get_inference_snapshot()["topology_revision"] == before

    manager.onload()
    assert all(engine.direct_eligible for engine in _named(manager).engines)


class _PlannedManager(_FakeManager):
    """A manager whose two 2-GPU engines are planned in the actor pool."""

    def __init__(self, gpu_ids):
        self._pg_tuple = ("shared-pg", list(range(8)), gpu_ids)
        self._made = []
        self._instance_owns_pg = False
        MultiEngineManager.__init__(
            self,
            SimpleNamespace(
                colocate=True,
                hybrid=False,
                rollout_num_gpus=4,
                num_gpus_per_node=8,
                resource={"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]},
                _genrm_instances_resolved={"judge": {"num_gpus": 4, "num_gpus_per_engine": 2}},
            ),
            num_slots=2,
            engine_actor_cls=_FakeEngineActorCls,
            log_prefix="[planned]",
        )

    def _physical_placement(self):
        return "actor", self._pg_tuple


def test_multi_engine_manager_validates_physical_layout_before_start(_patch_ray, monkeypatch):
    from relax.distributed.ray import placement_physical
    from relax.distributed.ray.placement_planner import PlacementError

    monkeypatch.setattr(placement_physical, "bundle_nodes_of", lambda pg: dict.fromkeys(range(8), "node-a"))

    # Bundles 6 and 7 of the judge's second engine got GPU 6 and GPU 9.
    with pytest.raises(PlacementError, match="genrm/judge engine 1.*not contiguous"):
        _PlannedManager(gpu_ids=[0, 1, 2, 3, 4, 5, 6, 9])
    # Refused before anything was started: not even the engine with a valid layout exists.
    assert _patch_ray.created == []

    manager = _PlannedManager(gpu_ids=list(range(8)))
    assert len(_patch_ray.created) == 2 and all(engine is not None for engine in manager.all_engines)


def test_multi_engine_manager_without_a_planned_pool_skips_the_physical_check(_patch_ray, monkeypatch):
    from relax.distributed.ray import placement_physical

    monkeypatch.setattr(placement_physical, "bundle_nodes_of", lambda pg: pytest.fail("nothing to check"))

    assert len(_FakeManager(num_slots=2).all_engines) == 2
