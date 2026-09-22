# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Agent-side thin client for the cluster-shared WebShop env server.

Mirrors the ``reset()/step()/close()`` surface the single-process
``env_webshop.WebshopEnv`` used to expose, but each call is an HTTP round-trip
to the shared server instead of in-process work.

One agent process drives exactly one session, so a synchronous client is fine:
there is no in-process concurrency to overlap.
"""

from __future__ import annotations

from typing import Any

import httpx


class WebshopClient:
    """HTTP client bound to one ``instance_id`` on the shared server."""

    def __init__(self, base_url: str, instance_id: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.instance_id = instance_id
        self._http = httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0))

    def _post(self, route: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._http.post(f"{self.base_url}{route}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def reset(self, goal_idx: int) -> tuple[str, list[str]]:
        """Start the session on ``goal_idx``; returns ``(raw_obs,
        available_actions)``."""
        data = self._post("/reset", {"instance_id": self.instance_id, "goal_idx": int(goal_idx)})
        return data["obs"], data["available_actions"]

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Apply ``action``; returns ``(raw_obs, reward, done, info)``.

        ``info`` carries ``won`` / ``task_score`` / ``available_actions``.
        """
        data = self._post("/step", {"instance_id": self.instance_id, "action": action})
        info = {
            "won": bool(data["won"]),
            "task_score": float(data["task_score"]),
            "available_actions": data["available_actions"],
        }
        return data["obs"], float(data["reward"]), bool(data["done"]), info

    def close(self) -> None:
        """Free the server-side session (best-effort) and the HTTP
        connection."""
        try:
            self._post("/close", {"instance_id": self.instance_id})
        except Exception:  # noqa: BLE001 - close is best-effort (server may be gone)
            pass
        finally:
            self._http.close()
