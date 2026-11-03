# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A queued shutdown cannot prevent independent proof after wrapper death."""

import asyncio
import time

import psutil
import pytest
import ray

from relax.backends.sglang.process_cleanup import cleanup_server_identity
from relax.distributed.ray.engine_pool import EnginePool, EnginePoolSpec


pytest_plugins = ["tests.cpu.distributed.ray.test_engine_pool_ray_integration"]


def _alive(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and (process.status() != psutil.STATUS_ZOMBIE or process.num_threads() > 1)
    except psutil.NoSuchProcess:
        return False


def test_queued_shutdown_times_out_then_real_node_proof_cleans_managed_group(ray_cluster):
    server_program = """
import os, sys, threading
os.setpgid(0, 0)
os.environ["RELAX_SGLANG_SERVER_INCARNATION"] = sys.argv[1]
print("group-ready", flush=True)
assert sys.stdin.readline().strip() == "fork"
pid = os.fork()
if pid:
    print(pid, flush=True)
threading.Event().wait()
"""

    @ray.remote(num_cpus=0)
    class Barrier:
        def __init__(self):
            self.entered, self.finish = asyncio.Event(), asyncio.Event()
            self.shutdown_started = False

        async def block(self):
            self.entered.set()
            await self.finish.wait()

        async def wait_entered(self):
            await self.entered.wait()

        def record_shutdown(self):
            self.shutdown_started = True

        def observed_shutdown(self):
            return self.shutdown_started

    @ray.remote(num_cpus=0, max_restarts=0)
    class Wrapper:
        def __init__(self):
            import subprocess
            import sys
            import uuid

            from relax.backends.sglang.process_cleanup import capture_server_identity

            token = uuid.uuid4().hex
            self.server = subprocess.Popen(
                [sys.executable, "-c", server_program, token],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            self.server._relax_server_incarnation = token
            self.identity = capture_server_identity(
                self.server, node_id=ray.get_runtime_context().get_node_id(), port=1
            )
            assert self.server.stdout.readline().strip() == "group-ready"

        def get_release_evidence(self):
            return self.identity

        def fork_worker(self):
            self.server.stdin.write("fork\n")
            self.server.stdin.flush()
            return int(self.server.stdout.readline())

        def block(self, barrier):
            self.barrier = barrier
            ray.get(barrier.block.remote())

        def shutdown(self):
            ray.get(self.barrier.record_shutdown.remote())
            raise AssertionError("shutdown must remain queued behind the blocked method")

    barrier = Barrier.remote()
    wrappers = [Wrapper.remote(), Wrapper.remote()]
    identities = []
    try:
        identities = ray.get([wrapper.get_release_evidence.remote() for wrapper in wrappers], timeout=30)
        workers = ray.get([wrapper.fork_worker.remote() for wrapper in wrappers], timeout=10)
        for identity, worker in zip(identities, workers, strict=True):
            assert identity["pgid"] == identity["server_pid"]
            assert [row["pid"] for row in identity["processes"]] == [identity["server_pid"]]
            assert psutil.Process(worker).environ().get("RELAX_SGLANG_SERVER_INCARNATION") is None
        blocked = wrappers[0].block.remote(barrier)
        ray.get(barrier.wait_entered.remote(), timeout=10)
        pool = EnginePool(
            EnginePoolSpec(
                actor_class=lambda: None,
                resolve_placement=lambda slot: None,
                actor_options=lambda *args: {},
                ctor=lambda *args: ((), {}),
                init_kwargs=lambda *args: {},
                teardown_timeout_s=5,
            ),
            [wrappers[0]],
        )
        pool.release_evidence[0] = identities[0]
        started = time.monotonic()
        pool.shutdown()
        assert time.monotonic() - started < 25
        assert pool.slots == [None]
        assert not pool._pending_shutdowns and not pool._orphan_cleanups
        with pytest.raises(ray.exceptions.ActorDiedError):
            ray.get(blocked, timeout=5)
        assert ray.get(barrier.observed_shutdown.remote(), timeout=5) is False
        assert not _alive(identities[0]["server_pid"]) and not _alive(workers[0])
        assert _alive(identities[1]["server_pid"]) and _alive(workers[1])
    finally:
        for wrapper in wrappers:
            ray.kill(wrapper, no_restart=True)
        for identity in identities:
            cleanup_server_identity(identity, timeout=5)
        ray.kill(barrier, no_restart=True)
