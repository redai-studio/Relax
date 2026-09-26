# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Subprocess proof that a timed-out one-shot attach worker cannot mutate
later."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _running(pid: int, created: float) -> bool:
    import psutil

    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - created) < 1e-3 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _write_tq_stub(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "transfer_queue.py").write_text(
        """\
import os
import time
from pathlib import Path
import psutil

MUTATED = False

def init(conf=None):
    process = psutil.Process()
    Path(os.environ["RELAX_TEST_TQ_STARTED_MARKER"]).write_text(
        f"{process.pid},{process.create_time()}", encoding="utf-8"
    )
    time.sleep(float(os.environ["RELAX_TEST_TQ_LATE_MUTATION_DELAY"]))
    global MUTATED
    MUTATED = True
    Path(os.environ["RELAX_TEST_TQ_LATE_MARKER"]).write_text("dirty", encoding="utf-8")
""",
        encoding="utf-8",
    )


def main(probe_dir: Path) -> None:
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.pop("RAY_ADDRESS", None)
    probe_dir.mkdir(parents=True, exist_ok=True)
    stub_dir = probe_dir / "stub"
    started = probe_dir / "started"
    mutated = probe_dir / "mutated"
    delay = 2.0
    _write_tq_stub(stub_dir)
    pythonpath = os.pathsep.join(filter(None, (str(stub_dir), os.environ.get("PYTHONPATH", ""))))

    import ray

    ray.init(
        address="local",
        num_cpus=1,
        include_dashboard=False,
        logging_level="ERROR",
        runtime_env={
            "env_vars": {
                "PYTHONPATH": pythonpath,
                "RELAX_TQ_ATTACH_TIMEOUT_SECONDS": "0.3",
                "RELAX_TEST_TQ_STARTED_MARKER": str(started),
                "RELAX_TEST_TQ_LATE_MARKER": str(mutated),
                "RELAX_TEST_TQ_LATE_MUTATION_DELAY": str(delay),
            }
        },
        _temp_dir=str(probe_dir / "ray"),
    )
    try:
        from relax.utils.tq import lifecycle

        conf = {"backend": {"storage_backend": "SimpleStorage"}, "controller": {}}

        @ray.remote(num_cpus=0)
        class Controller:
            def get_config(self) -> dict:
                return conf

        controller = Controller.options(
            name=lifecycle.CONTROLLER_NAME, namespace=lifecycle.CONTROLLER_NAMESPACE
        ).remote()
        assert ray.get(controller.get_config.remote()) == conf
        failures = lifecycle.verify_cluster_attach(conf, timeout=0.3)
        assert len(failures) == 1 and failures[0].endswith("handshake task failed (RayTaskError)")
        pid_text, created_text = started.read_text(encoding="utf-8").split(",")
        identity = int(pid_text), float(created_text)

        time.sleep(delay + 0.2)
        assert not mutated.exists(), "timed-out tq.init mutated state after task failure"
        deadline = time.monotonic() + 10
        while _running(*identity) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(*identity), "one-shot handshake worker did not exit"

        @ray.remote(num_cpus=0, max_retries=0)
        def clean_successor() -> tuple[int, float, bool]:
            import psutil
            import transfer_queue

            process = psutil.Process()
            return process.pid, process.create_time(), transfer_queue.MUTATED

        successor_pid, successor_created, successor_mutated = ray.get(clean_successor.remote())
        assert (successor_pid, successor_created) != identity
        assert successor_mutated is False and not mutated.exists()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
