# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""A model-local SGLang Router owned by the task's engine pool."""

from copy import copy
from typing import Any

import ray
import requests

from relax.utils.http_utils import _wrap_ipv6


class ModelRouter:
    def __init__(self) -> None:
        self.processes: list[Any] = []
        self.endpoint: tuple[str, int] | None = None
        self.ready = False

    def start(self, args: Any) -> None:
        from relax.distributed.ray import policy_engine_pool

        config = copy(args)
        before = tuple(policy_engine_pool._LAUNCHED_ROUTER_PROCESSES)
        try:
            host, port = policy_engine_pool._start_router(config, force_new=True)
            if not host or not port:
                raise RuntimeError("Static model router did not start")
            self.endpoint = (host, port)
        finally:
            self.processes.extend(
                process for process in policy_engine_pool._LAUNCHED_ROUTER_PROCESSES if process not in before
            )

    @property
    def url(self) -> str | None:
        if self.endpoint is None:
            return None
        return f"http://{_wrap_ipv6(self.endpoint[0])}:{self.endpoint[1]}"

    def activate(self, engines: list[Any]) -> None:
        self.ready = False
        if self.endpoint is None or not any(engine is not None for engine in engines):
            raise RuntimeError("Static router has no live replica")
        requests.get(f"{self.url}/health", timeout=5).raise_for_status()
        for engine in engines:
            if (
                engine is not None
                and ray.get(engine.configure_model_router.remote(*self.endpoint), timeout=60) is not True
            ):
                raise RuntimeError("Static replica router registration is unconfirmed")
        self.ready = True

    def withdraw(self, engines: list[Any]) -> None:
        self.ready = False
        for engine in engines:
            if (
                engine is not None
                and ray.get(engine.unregister_from_router.remote(wait_for_removal=True, timeout=30), timeout=35)
                is not True
            ):
                raise RuntimeError("Static replica router withdrawal is unconfirmed")

    def close(self) -> None:
        self.ready = False
        for process in tuple(self.processes):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            if process.is_alive():
                raise RuntimeError("Static router process exit is unconfirmed")
            self.processes.remove(process)
