# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Common pool ownership, publication and lifecycle for all inference roles.

Backend pools supply startup, endpoint and weight-sync strategies. Business
workloads keep their existing facades; only this owner publishes readiness.
"""

import copy
import threading
from collections.abc import Callable
from typing import Any


class InferenceManager:
    def __init__(self, role: str, pools: dict[str, Any] | None = None) -> None:
        self.role = role
        self.pools = pools if pools is not None else {}
        self.state = "ready"
        self.phase = "inference"
        self.topology_revision = 0
        self._published: dict | None = None
        self._lock = threading.RLock()
        self._loaded_tags: set[str] = set()

    def transition(self, operation: str, action: Callable[[], Any], tags: list[str] | None = None) -> Any:
        """Serialize transitions; failures close admission until recovery.

        A partial onload never publishes READY. Repeated full transitions and
        already loaded tags are no-ops. Backend actions must complete before
        the terminal state can be published.
        """
        with self._lock:
            if self.state == "dead" and operation != "shutdown":
                raise RuntimeError("Inference pool has been shut down")
            if operation == "activate":
                if self.state == "ready" or (tags and set(tags) <= self._loaded_tags):
                    return None
                pending, terminal = "onloading", "ready"
            elif operation == "deactivate":
                if self.state == "sleeping":
                    return None
                pending, terminal = "draining", "sleeping"
            elif operation == "drain":
                if self.state in {"sleeping", "draining"}:
                    return None
                pending = terminal = "draining"
            elif operation == "shutdown":
                if self.state == "dead":
                    return None
                pending, terminal = "draining", "dead"
            else:
                raise ValueError(f"Unknown lifecycle operation: {operation}")
            self.state = pending
            self.topology_revision += 1
            try:
                result = action()
            except BaseException:
                self.state = "failed"
                self._loaded_tags.clear()
                self.topology_revision += 1
                raise
            if operation == "activate" and tags:
                self._loaded_tags.update(tags)
                if not {"weights", "kv_cache", "cuda_graph"} <= self._loaded_tags:
                    terminal = "onloading"
            elif operation in {"deactivate", "shutdown"}:
                self._loaded_tags.clear()
            self.state = terminal
            self.topology_revision += 1
            return result

    def snapshot(self, models: dict[str, dict], *, default_model: str | None = None) -> dict:
        # Do not hold the transition lock here: discovery must remain available
        # during long GPU RPCs and report DRAINING/ONLOADING immediately.
        models = copy.deepcopy(models)
        state = self.state
        for info in models.values():
            info["state"] = state if state != "ready" else info.get("state", "ready")
            for engine in info["engines"]:
                if info["state"] != "ready":
                    engine["state"] = info["state"]
                    engine["direct_eligible"] = False
        content = {"role": self.role, "phase": self.phase, "models": models}
        if content != self._published:
            self._published = copy.deepcopy(content)
            self.topology_revision += 1
        return {
            **content,
            "topology_revision": self.topology_revision,
            "default_model": default_model,
            "routes": {name: name for name in models},
        }
