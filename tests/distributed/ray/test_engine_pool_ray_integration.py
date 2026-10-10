# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""``EnginePool`` behind a real manager actor on a local, CPU-only Ray cluster.

The unit tests fake Ray. This one does not: real actors on a real placement
group, a real ``ray.kill``, real ``RayActorError``. The engines are stand-ins
that only answer the calls the pool makes.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import ray
from ray.util.placement_group import placement_group, remove_placement_group

from relax.distributed.ray.multi_engine_manager import (
    SNAPSHOT_CONCURRENCY_GROUP,
    SNAPSHOT_CONCURRENCY_GROUPS,
    MultiEngineManager,
)


RELEASE_SECONDS = 3.0


@pytest.fixture(scope="module")
def ray_cluster():
    # Other test modules start Ray at import time and rely on it staying up,
    # so only stop a cluster this fixture started.
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
    yield
    if started_here:
        ray.shutdown()


def _manager_class():
    # Defined locally so Ray ships both classes by value: a test module is not
    # importable from a worker process.

    class StandInEngine:
        def __init__(self, args, *, rank, worker_type, base_gpu_id):
            self.release_seconds = args.release_seconds

        def init(self, host, port):
            return True

        def release_memory_occupation(self):
            time.sleep(self.release_seconds)

        def resume_memory_occupation(self, tags=None):
            return True

        def health_generate(self, timeout=5.0):
            return True

        def shutdown(self):
            return True

    @ray.remote(concurrency_groups=SNAPSHOT_CONCURRENCY_GROUPS)
    class Manager(MultiEngineManager):
        def __init__(self, args, pg_tuple):
            self.pg_tuple = pg_tuple
            super().__init__(args, num_slots=2, engine_actor_cls=StandInEngine, log_prefix="[integration]")

        @ray.method(concurrency_group=SNAPSHOT_CONCURRENCY_GROUP)
        def get_inference_snapshot(self) -> dict:
            return super().get_inference_snapshot()

        def kill_engine(self, rank):
            ray.kill(self.all_engines[rank])

        def _resolve_placement(self, rank):
            return self.pg_tuple, False, rank

        def _ray_resource_kwargs(self, rank):
            return {"num_cpus": 0.1}

        def _allocate_engine_addr_and_ports(self, *, new_engines):
            return {rank: {"host": "127.0.0.1", "port": 20000 + rank} for rank, _ in new_engines}

        def _build_engine_env_vars(self):
            return {}

    return Manager


@pytest.fixture
def manager(ray_cluster):
    pg = placement_group([{"CPU": 0.5}] * 2, strategy="PACK")
    ray.get(pg.ready(), timeout=60)
    handle = (
        _manager_class()
        .options(num_cpus=0.1)
        .remote(SimpleNamespace(release_seconds=RELEASE_SECONDS), (pg, [0, 1], [0, 1]))
    )
    yield handle
    try:
        ray.get(handle.shutdown.remote(), timeout=60)
    finally:
        ray.kill(handle)
        remove_placement_group(pg)


def _snapshot(manager) -> dict:
    return ray.get(manager.get_inference_snapshot.remote(), timeout=30)


def _states(manager) -> list[str]:
    return [engine["state"] for engine in _snapshot(manager)["engines"]]


def _states_once_changed(manager, initial: list[str]) -> list[str]:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        states = _states(manager)
        if states != initial:
            return states
        time.sleep(0.05)
    raise AssertionError(f"engine states never left {initial}")


def test_engine_pool_runs_a_full_lifecycle_on_real_ray(manager):
    started = _snapshot(manager)
    assert [engine["state"] for engine in started["engines"]] == ["ready", "ready"]
    assert [engine["base_url"] for engine in started["engines"]] == [
        "http://127.0.0.1:20000",
        "http://127.0.0.1:20001",
    ]
    assert ray.get(manager.health_check.remote(), timeout=30) is True

    # While the manager is busy releasing, discovery still answers -- and says
    # the engines are not available.
    offload = manager.offload.remote()
    assert _states_once_changed(manager, ["ready", "ready"]) == ["draining", "draining"]
    still_running, _ = ray.wait([offload], timeout=0)
    assert still_running == []
    ray.get(offload, timeout=60)
    assert _states(manager) == ["sleeping", "sleeping"]

    ray.get(manager.onload.remote(), timeout=60)
    assert _states(manager) == ["ready", "ready"]
    # Sleeping and waking never moved an engine, so the topology is the one it started with.
    assert _snapshot(manager)["topology_revision"] == started["topology_revision"]


def test_engine_pool_retires_and_rebuilds_a_killed_engine_on_real_ray(manager):
    before = _snapshot(manager)
    ray.get(manager.kill_engine.remote(1), timeout=30)

    # The dead actor is found at the next switch and retired; the other engine completes.
    ray.get(manager.offload.remote(), timeout=60)
    assert _states(manager) == ["sleeping", "dead"]

    # The next activation rebuilds it on the bundle it had.
    ray.get(manager.onload.remote(), timeout=120)
    after = _snapshot(manager)
    assert [engine["state"] for engine in after["engines"]] == ["ready", "ready"]
    assert after["topology_revision"] > before["topology_revision"]
    assert ray.get(manager.health_check.remote(), timeout=30) is True


def test_placement_physical_reads_bundle_nodes_from_real_ray(ray_cluster):
    from relax.distributed.ray.placement_physical import bundle_nodes_of, validate_engine_bundles
    from relax.distributed.ray.placement_planner import PlacementError

    pg = placement_group([{"CPU": 0.1}] * 2, strategy="PACK")
    ray.get(pg.ready(), timeout=60)
    try:
        nodes = bundle_nodes_of(pg)
        assert sorted(nodes) == [0, 1] and all(isinstance(node, str) and node for node in nodes.values())

        # Same node on a local cluster; whether the engine fits is then down to the GPU ids.
        validate_engine_bundles((pg, [0, 1], [0, 1]), 0, 2, label="engine")
        with pytest.raises(PlacementError, match="not contiguous"):
            validate_engine_bundles((pg, [0, 1], [0, 2]), 0, 2, label="engine")
    finally:
        remove_placement_group(pg)
