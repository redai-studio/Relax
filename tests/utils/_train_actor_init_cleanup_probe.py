# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Subprocess proof that failed train-actor init cannot mutate after
cleanup."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path


def main(directory: Path) -> None:
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.pop("RAY_ADDRESS", None)
    import ray

    from relax.distributed.ray.actor_group import RayTrainGroup

    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "late-mutation"

    @ray.remote(max_restarts=0)
    class InitActor:
        def __init__(self, fail: bool) -> None:
            self.fail = fail

        def init(self, _args, _role, **_kwargs) -> str:
            if not self.fail:
                return "ready"
            threading.Thread(target=lambda: (time.sleep(2), marker.write_text("dirty")), daemon=True).start()
            raise RuntimeError("expected initialization failure")

        def termination_probe(self) -> None:
            threading.Event().wait()

    ray.init(
        address="local", num_cpus=2, include_dashboard=False, logging_level="ERROR", _temp_dir=str(directory / "ray")
    )
    try:
        group = object.__new__(RayTrainGroup)
        group._actor_handlers = [InitActor.remote(True), InitActor.remote(False)]
        with_error = False
        try:
            group.init_and_wait(object(), "actor")
        except ray.exceptions.RayTaskError:
            with_error = True
        assert with_error and group._actor_handlers == []
        time.sleep(2.2)
        assert not marker.exists()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
