# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU crash regression: actor death does not prove backend-process cleanup."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _live_process(identity: tuple[int, float]) -> Any:
    import psutil

    try:
        process = psutil.Process(identity[0])
        if process.create_time() == identity[1] and process.status() != psutil.STATUS_ZOMBIE:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def _kill_owned_process(identity: tuple[int, float]) -> None:
    import psutil

    process = _live_process(identity)
    if process is not None:
        try:
            process.kill()  # psutil rechecks the PID's creation time before signalling.
        except psutil.NoSuchProcess:
            pass


def _run_crash_scenario(identity_file: Path) -> None:
    import psutil
    import ray
    from ray.util.placement_group import placement_group, placement_group_table
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from relax.distributed.ray.inference_manager import InferenceManager, InferenceRecoveryRequired

    @ray.remote(num_cpus=1, max_restarts=0)
    class BackendOwner:
        def start(self, path: str) -> tuple[tuple[int, float], tuple[int, float]]:
            worker = psutil.Process()
            # Bounded lifetime also protects against interruption before identity persistence.
            self.child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(90)"])
            child = psutil.Process(self.child.pid)
            identities = ((worker.pid, worker.create_time()), (child.pid, child.create_time()))
            Path(path).write_text(json.dumps(identities))
            return identities

        def shutdown(self) -> bool:
            self.child.terminate()
            self.child.wait(timeout=5)
            return True

    class CountedShutdown:
        def __init__(self, method: Any) -> None:
            self.method = method
            self.calls = 0

        def remote(self) -> Any:
            self.calls += 1
            return self.method.remote()

    try:
        ray.init(
            address="local",
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            log_to_driver=False,
            object_store_memory=80 * 1024 * 1024,
            # Match the production Ray 2.55.1 defaults even on newer local Ray versions.
            _system_config={
                "kill_child_processes_on_worker_exit": True,
                "kill_child_processes_on_worker_exit_with_raylet_subreaper": False,
                "process_group_cleanup_enabled": False,
            },
        )
        pg = placement_group([{"CPU": 1}])
        ray.get(pg.ready(), timeout=20)
        actor = BackendOwner.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote()
        worker_identity, child_identity = ray.get(actor.start.remote(str(identity_file)), timeout=20)
        shutdown = CountedShutdown(actor.shutdown)
        actor.shutdown = shutdown
        handle = actor
        manager = InferenceManager(SimpleNamespace(), num_slots=1, engine_actor_cls=object, skip_init=True)
        manager.all_engines[0] = handle
        placement = ((pg, [0], [0]), True)
        manager._engine_placements[0] = placement
        manager._inference_observation.initialized("__default__/replica-0", [handle], weights_ready=True)

        worker = _live_process(worker_identity)
        assert worker is not None
        assert _live_process(child_identity).ppid() == worker.pid
        worker.send_signal(signal.SIGKILL)

        for operation in (lambda: manager._retire_engines([0]), manager.recover, manager.recover):
            with pytest.raises(InferenceRecoveryRequired, match="unconfirmed"):
                operation()
            # These are real Ray RPCs and a real owned placement group, not lifecycle mocks.
            assert shutdown.calls == 1
            assert manager.all_engines[0] is handle
            assert manager._cleanup_pending == {0}
            assert not manager._shutdown_confirmed
            assert manager._engine_placements == {0: placement}
            assert placement_group_table(pg)["state"] == "CREATED"
            assert _live_process(child_identity) is not None
            entry = manager._inference_observation.entries["__default__/replica-0"]
            assert entry["state"] == "FAILED"
            assert not entry["admission"]
    finally:
        if identity_file.exists():
            for identity in reversed(json.loads(identity_file.read_text())):
                _kill_owned_process(tuple(identity))
        ray.shutdown()


@pytest.mark.skipif(sys.platform != "linux", reason="Requires Linux SIGKILL and Ray process-cleanup flags")
def test_terminal_actor_death_keeps_live_backend_and_owned_pg_fenced(tmp_path: Path) -> None:
    pytest.importorskip("ray")
    psutil = pytest.importorskip("psutil")
    identity_file = tmp_path / "owned-processes.json"
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    # A separate interpreter cannot disconnect or reconfigure a caller's initialized runtime.
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), str(identity_file)],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = process.communicate(timeout=75)
        assert process.returncode == 0, output
    except subprocess.TimeoutExpired:
        # Snapshot only this test's descendants, including Ray processes in other groups.
        descendants = psutil.Process(process.pid).children(recursive=True)
        identities = []
        for child in descendants:
            try:
                identities.append((child.pid, child.create_time()))
            except psutil.NoSuchProcess:
                pass
        for identity in reversed(identities):
            _kill_owned_process(identity)
        process.kill()
        output, _ = process.communicate(timeout=5)
        pytest.fail(f"Isolated Ray crash regression exceeded 75 seconds:\n{output}")
    finally:
        # The backend may have been reparented after the actor's crash.
        if identity_file.exists():
            for identity in reversed(json.loads(identity_file.read_text())):
                _kill_owned_process(tuple(identity))


if __name__ == "__main__":
    _run_crash_scenario(Path(sys.argv[1]))
