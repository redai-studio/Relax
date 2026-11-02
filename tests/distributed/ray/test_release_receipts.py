# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Exit receipts for one multi-node replica accumulate across retries."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
from ray.core.generated.gcs_pb2 import ActorTableData

from relax.backends.sglang.process_cleanup import capture_server_identity
from relax.distributed.ray import engine_pool
from relax.engine.inference.discovery import EngineState
from relax.engine.inference.ownership import ResourceBlockedError
from tests.distributed.ray.test_engine_pool import _Cluster, _make_pool


@pytest.fixture
def replica(monkeypatch):
    processes = []

    def create(owns_pg=False):
        cluster = _Cluster(monkeypatch)
        pool = _make_pool(cluster, 2, owns_pg=owns_pg, nodes_per_engine=2, teardown_timeout_s=0.5)
        pool.start()
        nodes = [{"NodeID": f"{slot + 1:02x}" * 28, "Alive": False} for slot in range(2)]
        proofs = []
        receipt = object()

        def start_server(slot, *, mutate=False):
            process = subprocess.Popen([sys.executable, "-c", "import threading; threading.Event().wait(60)"])
            processes.append(process)
            identity = capture_server_identity(process, node_id=nodes[slot]["NodeID"], port=18000 + slot)
            if mutate:
                pool.release_evidence[slot].update(identity)
            else:
                pool.release_evidence[slot] = identity
            pool.slots[slot]._get_local_state = lambda: ActorTableData.DEAD
            pool.slots[slot].fail["shutdown"] = engine_pool.ray.exceptions.ActorDiedError()
            return process

        servers = [start_server(slot) for slot in range(2)]

        def node_task(function):
            def run(identity):
                function(identity, timeout=0.2)
                proofs.append(identity["node_id"])
                return receipt

            return SimpleNamespace(options=lambda **kwargs: SimpleNamespace(remote=run))

        monkeypatch.setattr(engine_pool.ray, "nodes", lambda: nodes)
        monkeypatch.setattr(engine_pool.ray, "remote", lambda **kwargs: node_task)
        monkeypatch.setattr(
            engine_pool.ray, "get", lambda ref, timeout=None: None if ref is receipt else cluster.get(ref, timeout)
        )
        return SimpleNamespace(
            pool=pool, nodes=nodes, servers=servers, proofs=proofs, cluster=cluster, start_server=start_server
        )

    yield create
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("owns_pg", [False, True])
def test_multi_node_exit_receipts_survive_alternating_node_reachability(replica, owns_pg):
    case = replica(owns_pg)
    pool, nodes = case.pool, case.nodes
    original = list(pool.slots)
    nodes[0]["Alive"] = True
    with pytest.raises(ResourceBlockedError):
        pool.shutdown()
    assert case.servers[0].poll() is not None and case.servers[1].poll() is None
    assert pool.slots == original and set(pool.placements) == {0, 1}
    assert pool.state(0) is EngineState.BLOCKED
    assert case.cluster.removed_pgs == []

    nodes[0]["Alive"], nodes[1]["Alive"] = False, True
    pool.shutdown()
    assert all(process.poll() is not None for process in case.servers)
    assert pool.slots == [None, None] and pool.placements == {}
    assert original[0].called().count("shutdown") == 1
    assert case.proofs == [node["NodeID"] for node in nodes]
    assert case.cluster.removed_pgs == (["pg-0", "pg-1"] if owns_pg else [])
    nodes[1]["Alive"] = False
    pool.shutdown()
    assert len(case.proofs) == 2


@pytest.mark.parametrize("mutate_identity", [False, True])
def test_exit_receipt_cannot_be_reused_for_a_different_recorded_process(replica, mutate_identity):
    case = replica()
    pool, nodes = case.pool, case.nodes
    nodes[0]["Alive"] = True
    with pytest.raises(ResourceBlockedError):
        pool.shutdown()
    replacement = case.start_server(0, mutate=mutate_identity)
    nodes[0]["Alive"], nodes[1]["Alive"] = False, True
    with pytest.raises(ResourceBlockedError):
        pool.shutdown()
    assert replacement.poll() is None and all(pool.slots)
    assert set(pool.placements) == {0, 1}
    nodes[0]["Alive"], nodes[1]["Alive"] = True, False
    pool.shutdown()
    assert replacement.poll() is not None
    assert pool.slots == [None, None]
    assert case.proofs == [nodes[0]["NodeID"], nodes[1]["NodeID"], nodes[0]["NodeID"]]


def test_rebuilt_slots_require_new_exit_receipts(replica):
    case = replica()
    pool, nodes = case.pool, case.nodes
    for node in nodes:
        node["Alive"] = True
    pool.shutdown()
    assert len(case.proofs) == 2
    pool.start()
    assert pool.incarnations == {0: 2, 1: 2}
    new = [case.start_server(slot) for slot in range(2)]
    for node in nodes:
        node["Alive"] = False
    with pytest.raises(ResourceBlockedError):
        pool.shutdown()
    assert all(process.poll() is None for process in new)
    assert all(pool.slots) and set(pool.placements) == {0, 1}
    assert len(case.proofs) == 2
