# Copyright (c) 2026 Relax Authors. All Rights Reserved.


from __future__ import annotations

from types import SimpleNamespace

import pytest

from relax.distributed.ray import inference_manager
from relax.distributed.ray.inference_manager import InferenceManager


class _RemoteCall:
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


class _FakeManager(InferenceManager):
    def __init__(
        self, num_slots: int, *, owns_pg: bool = False, nodes_per_engine: int = 1, log_prefix: str = "[fake]"
    ):
        self._made: list[_FakeEngine] = []
        self._dead_at_init: set[int] = set()
        self._removed_pg = False
        self._instance_owns_pg = owns_pg
        self._inference_preserves_weights = True
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
    pass


@pytest.fixture(autouse=True)
def _patch_ray(monkeypatch):
    import relax.distributed.ray.inference_manager as inference_manager

    created: list[_FakeEngine] = []
    killed: list[_FakeEngine] = []

    def fake_remote(cls):
        if cls is not _FakeEngineActorCls:
            return cls

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

    monkeypatch.setattr(inference_manager.ray, "remote", fake_remote)
    monkeypatch.setattr(inference_manager.ray, "get", fake_get)
    monkeypatch.setattr(inference_manager.ray, "kill", fake_kill)
    monkeypatch.setattr(inference_manager.ray.exceptions, "RayActorError", RuntimeError, raising=False)

    yield SimpleNamespace(created=created, killed=killed)


def test_fanout_isolates_one_dead_engine_from_the_rest(_patch_ray):
    manager = _FakeManager(num_slots=3)
    for engine in manager.all_engines:
        engine.calls.clear()
    dead_engine = manager.all_engines[1]
    dead_engine.dead_methods = frozenset({"release_memory_occupation"})

    dead_ranks = manager._fanout("release_memory_occupation")

    assert dead_ranks == [1]
    assert manager.all_engines[0].calls == ["release_memory_occupation"]
    assert manager.all_engines[2].calls == ["release_memory_occupation"]


def test_fanout_reraises_non_dead_exceptions(_patch_ray, monkeypatch):
    import relax.distributed.ray.inference_manager as inference_manager

    manager = _FakeManager(num_slots=1)

    def raise_value_error(handle_or_list, timeout=None):
        raise ValueError("a real bug, not a dead engine")

    monkeypatch.setattr(inference_manager.ray, "get", raise_value_error)

    with pytest.raises(ValueError, match="a real bug"):
        manager._fanout("release_memory_occupation")


def test_offload_onload_are_idempotent_when_state_unchanged(_patch_ray):
    manager = _FakeManager(num_slots=2)
    for engine in manager.all_engines:
        engine.calls.clear()
    assert manager.is_onloaded()

    manager.offload()
    assert not manager.is_onloaded()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation"]

    manager.offload()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation"]

    manager.onload()
    assert manager.is_onloaded()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation", "resume_memory_occupation"]

    manager.onload()
    for engine in manager.all_engines:
        assert engine.calls == ["release_memory_occupation", "resume_memory_occupation"]


def test_retire_engines_kills_and_nulls_the_slot(_patch_ray):
    manager = _FakeManager(num_slots=2)
    dead_engine = manager.all_engines[0]

    manager._retire_engines([0])

    assert manager.all_engines[0] is None
    assert manager.all_engines[1] is not None
    assert dead_engine in _patch_ray.killed


def test_recover_rebuilds_only_the_dead_slot(_patch_ray):
    manager = _FakeManager(num_slots=2)
    original_engine_1 = manager.all_engines[1]
    manager.all_engines[0] = None

    rebuilt = manager.recover()

    assert rebuilt == {0}
    assert manager.all_engines[0] is not None
    assert manager.all_engines[0] is not original_engine_1
    assert manager.all_engines[1] is original_engine_1


def test_recover_raises_on_total_wipeout(_patch_ray, monkeypatch):
    import relax.distributed.ray.inference_manager as inference_manager

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

    monkeypatch.setattr(inference_manager.ray, "remote", lambda cls: _FailingActor)

    with pytest.raises(RuntimeError, match="could not be rebuilt"):
        manager.recover()


def test_shutdown_removes_owned_placement_group_but_not_borrowed_one(_patch_ray, monkeypatch):
    removed_pgs = []
    monkeypatch.setattr(inference_manager, "remove_placement_group", lambda pg: removed_pgs.append(pg))

    owning_manager = _FakeManager(num_slots=1, owns_pg=True)
    owning_manager.shutdown()
    assert removed_pgs == ["pg-0"]

    removed_pgs.clear()
    borrowing_manager = _FakeManager(num_slots=1, owns_pg=False)
    borrowing_manager.shutdown()
    assert removed_pgs == []


def test_inference_manager_partial_resume_does_not_skip_full_resume(_patch_ray):
    manager = _FakeManager(num_slots=1)
    engine = manager.all_engines[0]
    engine.calls.clear()

    manager.offload()
    manager.onload(tags=["weights"])
    manager.onload()

    assert engine.calls == [
        "release_memory_occupation",
        "resume_memory_occupation",
        "resume_memory_occupation",
        "continue_generation",
    ]


def test_inference_manager_shutdown_retires_all_multinode_worker_slots(_patch_ray):
    manager = _FakeManager(num_slots=4, nodes_per_engine=2)
    original_engines = list(manager.all_engines)

    manager.shutdown()

    assert set(_patch_ray.killed) == set(original_engines)
    assert manager.all_engines == [None, None, None, None]
