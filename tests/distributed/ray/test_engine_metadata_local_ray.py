# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Real CPU Ray checks for metadata observation and reference lifetime.

The production get_engines_info method runs in this process against real Ray
actors and ObjectRefs. Manager construction is bypassed to avoid GPU/TQ setup.
The collected test launches the six checks in a fresh subprocess, preserving
any Ray runtime other tests may already own. HTTP/Serve cancellation is not
covered.
"""

import gc
import os
import subprocess
import sys
import tempfile
import time
import uuid
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
from ray.core.generated.gcs_pb2 import ActorTableData

from relax.distributed.ray import rollout


ActorState = ActorTableData.ActorState


def wait_until(predicate, *, timeout=15.0, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.025)
    pytest.fail(f"Timed out waiting for {description}")


@pytest.fixture(scope="module")
def isolated_ray():
    if ray.is_initialized():
        pytest.fail("Refusing to reuse or stop an existing Ray runtime; metadata tests need an unconnected process")
    # Short unique paths also keep Ray's Unix socket paths below 108 bytes.
    with tempfile.TemporaryDirectory(prefix="ce9b-metadata-ray-") as runtime_dir:
        try:
            context = ray.init(
                address="local",
                namespace=f"engine-metadata-{uuid.uuid4().hex}",
                _temp_dir=runtime_dir,
                num_cpus=2,
                num_gpus=0,
                object_store_memory=80 * 1024 * 1024,
                include_dashboard=False,
                log_to_driver=False,
            )
            assert context.address_info["session_dir"].startswith(runtime_dir + os.sep)
            assert ray.cluster_resources().get("GPU", 0) == 0
            yield context
        finally:
            # Only shut down the runtime this fixture started with address=local.
            ray.shutdown()


@pytest.fixture
def engine_factory(isolated_ray):
    @ray.remote(num_cpus=0)
    class Journal:
        def __init__(self):
            self.events = []
            self.released = set()

        def record(self, event):
            self.events.append(event)

        def release(self, gate):
            self.released.add(gate)

        def is_released(self, gate):
            return gate in self.released

        def snapshot(self):
            return list(self.events)

    @ray.remote(num_cpus=1, max_restarts=0, max_task_retries=0)
    class FakeEngine:
        def __init__(self, journal, block_constructor):
            self.journal = journal
            self.getter_calls = {"url": 0, "pid_node": 0}
            self._record("constructor_enter")
            if block_constructor:
                self._wait_for_gate("constructor")
            self._record("constructor_exit")

        def _record(self, event):
            ray.get(self.journal.record.remote(event), timeout=5)

        def _wait_for_gate(self, gate):
            deadline = time.monotonic() + 60
            while not ray.get(self.journal.is_released.remote(gate), timeout=5):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"test gate {gate} was not released")
                time.sleep(0.01)

        def init(self, block=False):
            self._record("init_enter")
            if block:
                self._wait_for_gate("init")
            self.metadata = {
                "url": "http://127.0.0.1:31001",
                "pid": os.getpid(),
                "node_id": ray.get_runtime_context().get_node_id(),
            }
            self._record("init_exit")
            return dict(self.metadata)

        def long_work(self):
            self._record("work_enter")
            self._wait_for_gate("work")
            self._record("work_exit")
            return "business-work-completed"

        def get_url(self):
            self.getter_calls["url"] += 1
            self._record("get_url")
            return self.metadata["url"]

        def get_pid_and_node_id(self):
            self.getter_calls["pid_node"] += 1
            self._record("get_pid_and_node_id")
            return {key: self.metadata[key] for key in ("pid", "node_id")}

        def get_getter_counts(self):
            return dict(self.getter_calls)

    journal = Journal.remote()
    handles = []

    def create(*, block_constructor=False):
        engine = FakeEngine.remote(journal, block_constructor)
        # The lifetime test must not acquire an extra strong ActorHandle here.
        handles.append(weakref.ref(engine))
        return engine, journal

    try:
        yield create
    finally:
        for handle_ref in handles:
            handle = handle_ref()
            if handle is not None:
                ray.kill(handle, no_restart=True)
        ray.kill(journal, no_restart=True)


def make_manager(engine, init_ref=None):
    records = {} if init_ref is None else {0: rollout._EngineInitRecord(init_ref)}
    group = rollout.EngineGroup(
        args=SimpleNamespace(num_gpus_per_node=8),
        pg=None,
        all_engines=[engine],
        num_gpus_per_engine=1,
        num_new_engines=0,
        engine_init_records=records,
    )
    manager = object.__new__(rollout.RolloutManager.__ray_metadata__.modified_class)
    manager.servers = {"actor": rollout.RolloutServer(engine_groups=[group], model_name="actor")}
    manager._engine_actor_states = rollout._get_engine_actor_states()
    assert manager._engine_actor_states is not None, f"Local actor state API missing in Ray {ray.__version__}"
    return manager, group


def row(result):
    return result["models"]["actor"]["engine_groups"][0]["engines"][0]


def wait_for_event(journal, event):
    return wait_until(
        lambda: event in ray.get(journal.snapshot.remote(), timeout=5),
        description=f"FakeEngine event {event}",
    )


def query_repeatedly(manager, monkeypatch):
    real_get = ray.get
    timeouts = []

    def traced_get(ref, *, timeout=None):
        # Keep the real Ray operation: this spy only verifies the wait bound.
        timeouts.append(timeout)
        assert timeout == 0, "Management observation must not wait for init"
        return real_get(ref, timeout=timeout)

    with monkeypatch.context() as patch:
        patch.setattr(rollout.ray, "get", traced_get)
        started = time.monotonic()
        results = [manager.get_engines_info() for _ in range(5)]
        elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"Five metadata queries blocked for {elapsed:.3f}s"
    return results, timeouts


def ready_observation(manager):
    return wait_until(
        lambda: (result := manager.get_engines_info())["observation"]["complete"] and result,
        description="complete metadata observation",
    )


def check_local_ray_slow_constructor_and_init_queries_preserve_business_init(engine_factory, monkeypatch):
    engine, journal = engine_factory(block_constructor=True)
    wait_for_event(journal, "constructor_enter")
    manager, group = make_manager(engine)

    # The handle is published, but init has not even been submitted.
    results, _ = query_repeatedly(manager, monkeypatch)
    assert all(result["observation"]["complete"] is False for result in results)
    assert all("url" not in row(result) for result in results)

    init_ref = engine.init.remote(block=True)
    group.engine_init_records[0] = rollout._EngineInitRecord(init_ref)
    results, _ = query_repeatedly(manager, monkeypatch)
    assert all(result["observation"]["complete"] is False for result in results)
    assert ray.wait([init_ref], timeout=0)[0] == []

    ray.get(journal.release.remote("constructor"), timeout=5)
    wait_for_event(journal, "init_enter")
    wait_until(lambda: engine._get_local_state() == ActorState.ALIVE, description="local ALIVE state")
    results, timeouts = query_repeatedly(manager, monkeypatch)
    assert timeouts == [0] * 5
    assert all(result["observation"]["issues"][0]["reason"] == "init_pending" for result in results)
    assert ray.wait([init_ref], timeout=0)[0] == []

    # Pending zero-wait reads neither completed nor cancelled the business init.
    ray.get(journal.release.remote("init"), timeout=5)
    expected = ray.get(init_ref, timeout=10)
    result = ready_observation(manager)
    assert {key: row(result)[key] for key in expected} == expected
    assert ray.get(engine.get_getter_counts.remote(), timeout=5) == {"url": 0, "pid_node": 0}
    assert ray.get(journal.snapshot.remote(), timeout=5) == [
        "constructor_enter",
        "constructor_exit",
        "init_enter",
        "init_exit",
    ]


def check_local_ray_cached_metadata_does_not_queue_behind_long_business_method(engine_factory, monkeypatch):
    engine, journal = engine_factory()
    init_ref = engine.init.remote()
    expected = ray.get(init_ref, timeout=10)
    manager, group = make_manager(engine, init_ref)
    ready_observation(manager)
    assert isinstance(group.engine_init_records[0].value, dict)

    work_ref = engine.long_work.remote()
    wait_for_event(journal, "work_enter")
    results, timeouts = query_repeatedly(manager, monkeypatch)
    assert timeouts == [], "Successful metadata must be served from the cached dict"
    assert all(result["observation"]["complete"] is True for result in results)
    assert all(row(result)["url"] == expected["url"] for result in results)
    assert ray.wait([work_ref], timeout=0)[0] == []

    ray.get(journal.release.remote("work"), timeout=5)
    assert ray.get(work_ref, timeout=10) == "business-work-completed"
    assert ray.get(engine.get_getter_counts.remote(), timeout=5) == {"url": 0, "pid_node": 0}


@pytest.mark.parametrize("cache_before_death", [False, True])
def check_local_ray_dead_handle_is_incomplete_until_slot_is_cleared(engine_factory, cache_before_death):
    engine, _ = engine_factory()
    init_ref = engine.init.remote()
    ray.get(init_ref, timeout=10)
    manager, group = make_manager(engine, init_ref)
    if cache_before_death:
        ready_observation(manager)

    ray.kill(engine, no_restart=True)
    wait_until(lambda: engine._get_local_state() == ActorState.DEAD, description="local DEAD state propagation")
    result = manager.get_engines_info()
    assert row(result) == {"rank": 0, "status": "active", "actor_state": "DEAD"}
    assert result["observation"] == {
        "complete": False,
        "issues": [{"scope": "actor/0/0", "reason": "actor_dead_unreconciled"}],
    }

    group.all_engines[0] = None
    result = manager.get_engines_info()
    assert row(result) == {"rank": 0, "status": "dead", "actor_state": "ABSENT"}
    assert result["observation"] == {"complete": True, "issues": []}


def check_local_ray_uncached_init_ref_does_not_keep_actor_or_cpu_alive(engine_factory):
    engine, _ = engine_factory()
    init_ref = engine.init.remote()
    expected = ray.get(init_ref, timeout=10)
    manager, group = make_manager(engine, init_ref)
    record = group.engine_init_records[0]
    assert record.value is init_ref
    actor_id = engine._actor_id.hex()
    handle_ref = weakref.ref(engine)

    # Retain an uncached record after clearing its slot. The fixture also only
    # retains weak handles, and no metadata GET has consumed the init ref.
    group.all_engines[0] = None
    del engine
    gc.collect()
    assert handle_ref() is None
    wait_until(
        lambda: ray._private.state.actors(actor_id).get("State") == "DEAD",
        description="Actor GC while its completed init ObjectRef remains held",
    )
    wait_until(lambda: ray.available_resources().get("CPU", 0) == 2, description="Actor CPU resource release")
    assert record.value is init_ref
    assert ray.get(record.value, timeout=0) == expected
    assert manager.get_engines_info()["observation"] == {"complete": True, "issues": []}


def check_local_ray_successful_cache_releases_the_record_object_ref(engine_factory):
    engine, _ = engine_factory()
    init_ref = engine.init.remote()
    expected = ray.get(init_ref, timeout=10)
    ref_id = init_ref.hex().encode("ascii")
    manager, group = make_manager(engine, init_ref)
    core_worker = ray._private.worker.global_worker.core_worker
    assert core_worker.get_all_reference_counts()[ref_id]["local"] > 0
    ready_observation(manager)
    assert group.engine_init_records[0].value == expected
    assert isinstance(group.engine_init_records[0].value, dict)

    del init_ref
    gc.collect()
    # Query only this fixture's runtime. Absence or local=0 both mean that the
    # successful record and the test no longer retain a local ObjectRef.
    wait_until(
        lambda: core_worker.get_all_reference_counts().get(ref_id, {}).get("local", 0) == 0,
        description="completed init ObjectRef local-reference release after caching",
    )
    assert row(manager.get_engines_info())["url"] == expected["url"]


def test_local_ray_in_isolated_process():
    test_file = Path(__file__).resolve()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_file),
            "-o",
            "python_functions=check_local_ray_*",
            "-q",
            "--disable-warnings",
        ],
        cwd=test_file.parents[3],
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "6 passed" in completed.stdout and " skipped" not in completed.stdout, output
