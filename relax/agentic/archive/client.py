# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Fail-open synchronous producer: bounded RPCs, no threads or import-time
I/O."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable

from relax.utils.logging_utils import get_logger

from .schema import encode


logger = get_logger(__name__)


class Recorder:
    def __init__(
        self,
        endpoint: str,
        metadata: dict,
        token: str = "",
        timeout: float = 5.0,
        *,
        transport: Callable[[str, dict], dict] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token, self.timeout = token, timeout
        self.transport = transport
        self.trajectory_id: str | None = None
        self.reason: str | None = None
        result = self._send("begin", {"metadata": metadata})
        if result:
            self.trajectory_id = result["trajectory_id"]

    def _rpc(self, operation: str, payload: dict) -> dict:
        if self.transport is not None:
            return self.transport(operation, payload)
        request = urllib.request.Request(
            self.endpoint + "/" + operation,
            data=encode(payload),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if not result.get("ok"):
            raise ValueError(result.get("reason", "archive request failed"))
        return result

    def _send(self, operation: str, payload: dict) -> dict | None:
        try:
            return self._rpc(operation, payload)
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
            logger.warning("CUA archive abandoned candidate: %s", self.reason)
            identity, self.trajectory_id = self.trajectory_id, None
            if identity:
                try:
                    self._rpc("abort", {"trajectory_id": identity})
                except Exception:
                    pass  # Leftover staging remains charged until exclusive recovery.
            return None

    def event(self, kind: str, **data: Any) -> None:
        if self.trajectory_id:
            self._send("event", {"trajectory_id": self.trajectory_id, "event": {"kind": kind, "data": data}})

    def finish(self, outcome: dict) -> dict | None:
        if not self.trajectory_id:
            return None
        result = self._send("finish", {"trajectory_id": self.trajectory_id, "outcome": outcome})
        self.trajectory_id = None
        if result and not result.get("retained"):
            self.reason = result.get("reason")
            logger.info("CUA archive candidate not retained: %s", self.reason)
        return result

    def abort(self) -> None:
        if self.trajectory_id:
            self._send("abort", {"trajectory_id": self.trajectory_id})
            self.trajectory_id = None
