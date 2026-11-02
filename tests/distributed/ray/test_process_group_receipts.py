# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep node-local partial process-group evidence across Ray wait timeouts."""

import os
import pickle
import signal
from types import SimpleNamespace

import psutil
import pytest
from ray.core.generated.gcs_pb2 import ActorTableData

from relax.distributed.ray import engine_pool
from relax.engine.inference.ownership import ResourceBlockedError
from tests.backends.sglang.test_server_incarnation import _alive, _exit_parent, _start_late_worker
from tests.distributed.ray.test_engine_pool import _Cluster, _make_pool


pytest_plugins = [
    "tests.backends.sglang.test_server_incarnation",
    "tests.distributed.ray.test_engine_pool_ray_integration",
]


@pytest.mark.parametrize("follower", ["cpu_pool"], indirect=True)
def test_node_group_evidence_survives_serialization_and_pending_ray_wait(follower, monkeypatch):
    record = follower()
    _start_late_worker(record)
    cluster = _Cluster(monkeypatch)
    pool = _make_pool(cluster, 1, teardown_timeout_s=0.5)
    pool.start()
    engine = pool.slots[0]
    engine._get_local_state = lambda: ActorTableData.DEAD
    engine.fail["shutdown"] = engine_pool.ray.exceptions.ActorDiedError()
    identity = pickle.loads(pickle.dumps(record.engine._release_evidence))
    assert identity["pgid"] == record.process.pid
    identity["node_id"] = "01" * 28
    pool.release_evidence[0] = identity
    node = {"NodeID": identity["node_id"], "Alive": True}
    refs = []

    def node_task(function):
        def run(saved):
            # Ray copies arguments and exceptions: mutations in this task
            # cannot silently update the owner's identity through aliasing.
            copied = pickle.loads(pickle.dumps(saved))
            try:
                function(copied, timeout=0 if not refs else 2)
            except Exception as error:
                outcome = pickle.loads(pickle.dumps(error))
            else:
                outcome = None
            ref = SimpleNamespace(outcome=outcome, gets=0)
            refs.append(ref)
            return ref

        return SimpleNamespace(options=lambda **kwargs: SimpleNamespace(remote=run))

    def get(ref, timeout=None):
        if any(ref is pending for pending in refs):
            ref.gets += 1
            if ref is refs[0] and ref.gets == 1:
                raise engine_pool.ray.exceptions.GetTimeoutError()
            if isinstance(ref.outcome, Exception):
                raise ref.outcome
            return ref.outcome
        return cluster.get(ref, timeout)

    monkeypatch.setattr(engine_pool.ray, "remote", lambda **kwargs: node_task)
    monkeypatch.setattr(engine_pool.ray, "nodes", lambda: [node])
    monkeypatch.setattr(engine_pool.ray, "get", get)
    killpg = os.killpg

    def delay_worker_exit(pgid, sig):
        if pgid == record.process.pid and sig == signal.SIGKILL:
            _exit_parent(record)
            return
        return killpg(pgid, sig)

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "killpg", delay_worker_exit)
        patcher.setattr(psutil.Process, "kill", lambda proc: None)
        with pytest.raises(ResourceBlockedError):
            pool.shutdown()
        assert len(refs) == 1 and pool.release_evidence[0]["pgid"] == record.process.pid
        assert pool.slots == [engine] and record.worker.is_running()
        # An already submitted proof must still settle when GCS says DEAD.
        node["Alive"] = False
        with pytest.raises(ResourceBlockedError):
            pool.shutdown()
        assert len(refs) == 1 and refs[0].gets == 2
        assert pool.release_evidence[0]["pgid"] == record.process.pid
        assert _alive(record.worker)
    node["Alive"] = True
    pool.shutdown()
    assert len(refs) == 2 and pool.slots == [None]
    assert not _alive(record.worker)


@pytest.mark.parametrize("mismatch", [None, "node_id", "server_pid", "incarnation", "processes"])
def test_shutdown_partial_evidence_is_bound_to_server_generation(monkeypatch, mismatch):
    from relax.backends.sglang.process_cleanup import ServerCleanupError

    cluster = _Cluster(monkeypatch)
    pool = _make_pool(cluster, 1)
    pool.start()
    identity = {
        "node_id": "node-a",
        "server_pid": 12345,
        "incarnation": "original",
        "processes": [{"pid": 12345, "create_time": 1.0}],
    }
    pool.release_evidence[0] = identity
    learned = pickle.loads(pickle.dumps(identity))
    if mismatch is not None:
        learned[mismatch] = {
            "node_id": "node-b",
            "server_pid": 54321,
            "incarnation": "different",
            "processes": [{"pid": 12345, "create_time": 2.0}],
        }[mismatch]
    learned.update(pgid=12345, released_process_group=[12345, 1.0, "original"])
    error = ServerCleanupError("worker exit pending", learned)
    # Exercise the same dynamic cause wrapper that ray.get returns, including
    # an exception serialization round trip, not just the plain local error.
    task_error = engine_pool.ray.exceptions.RayTaskError("shutdown", "trace", error)
    pool.slots[0].fail["shutdown"] = pickle.loads(pickle.dumps(task_error)).as_instanceof_cause()
    monkeypatch.setattr(engine_pool.ray, "nodes", lambda: [{"NodeID": "node-a", "Alive": False}])
    with pytest.raises(ResourceBlockedError):
        pool.shutdown()
    assert pool.slots[0] is not None
    if mismatch is None:
        assert identity["pgid"] == 12345
        assert identity["released_process_group"] == [12345, 1.0, "original"]
    else:
        assert "pgid" not in identity and "released_process_group" not in identity


def test_partial_group_evidence_crosses_real_ray_exception_transport(ray_cluster):
    from relax.backends.sglang.process_cleanup import ServerCleanupError

    @engine_pool.ray.remote
    def fail_with_partial_evidence():
        from relax.backends.sglang.process_cleanup import ServerCleanupError

        raise ServerCleanupError(
            "escapee still alive", {"pgid": 12345, "released_process_group": [12345, 1.0, "original"]}
        )

    with pytest.raises(ServerCleanupError, match="escapee still alive") as caught:
        engine_pool.ray.get(fail_with_partial_evidence.remote(), timeout=30)
    assert caught.value.identity == {"pgid": 12345, "released_process_group": [12345, 1.0, "original"]}


@pytest.mark.parametrize("partial", [False, True])
def test_shutdown_wait_timeout_keeps_actor_and_ref_until_evidence_arrives(monkeypatch, partial):
    from relax.backends.sglang.process_cleanup import ServerCleanupError

    cluster = _Cluster(monkeypatch)
    pool = _make_pool(cluster, 1, teardown_timeout_s=0.01)
    pool.start()
    engine = pool.slots[0]
    identity = {
        "node_id": "01" * 28,
        "server_pid": 12345,
        "incarnation": "original",
        "processes": [{"pid": 12345, "create_time": 1.0}],
    }
    pool.release_evidence[0] = identity
    engine._get_local_state = lambda: ActorTableData.DEAD if cluster.killed else ActorTableData.ALIVE
    shutdown_ref, orphan_ref = object(), object()
    submitted, gets, orphans = [], [], []

    def shutdown():
        submitted.append(shutdown_ref)
        return shutdown_ref

    engine.shutdown = SimpleNamespace(remote=shutdown)

    def node_task(function):
        def run(saved):
            assert len(gets) == 2, "killed actor/submitted orphan before its pending shutdown settled"
            assert saved["pgid"] == 12345
            orphans.append(saved)
            return orphan_ref

        return SimpleNamespace(options=lambda **kwargs: SimpleNamespace(remote=run))

    def get(ref, timeout=None):
        if ref is shutdown_ref:
            gets.append(ref)
            if len(gets) == 1:
                raise engine_pool.ray.exceptions.GetTimeoutError()
            if partial:
                raise ServerCleanupError("worker still alive", {**identity, "pgid": 12345})
            return None
        if ref is orphan_ref:
            return None
        return cluster.get(ref, timeout)

    monkeypatch.setattr(engine_pool.ray, "get", get)
    monkeypatch.setattr(engine_pool.ray, "remote", lambda **kwargs: node_task)
    monkeypatch.setattr(engine_pool.ray, "nodes", lambda: [{"NodeID": identity["node_id"], "Alive": True}])
    monkeypatch.setattr(engine_pool.ray, "kill", lambda actor, **kw: cluster.killed.append(actor))
    with pytest.raises(ResourceBlockedError):
        pool.shutdown()
    assert cluster.killed == [] and orphans == []
    assert submitted == [shutdown_ref] and gets == [shutdown_ref]
    assert pool.slots == [engine]
    pool.shutdown()
    assert submitted == [shutdown_ref] and gets == [shutdown_ref, shutdown_ref]
    assert bool(orphans) is partial
    assert pool.slots == [None]
