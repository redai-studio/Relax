# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Drain admitted HTTP requests, including streaming response bodies."""

import asyncio
import json
import time


class InstallationFence:
    """Admission and worker commands share one installation identity."""

    def __init__(self) -> None:
        self.identity = None
        self.phase = "NEW"

    def claim(self, identity: dict) -> None:
        if type(identity.get("epoch")) is not int or identity["epoch"] < 0 or not identity.get("workers"):
            raise ValueError("Installation requires an epoch and fixed worker incarnations")
        if self.identity is not None:
            if identity["epoch"] < self.identity["epoch"]:
                raise ValueError("Stale admission epoch")
            if identity["epoch"] == self.identity["epoch"]:
                self.require(identity)
                return
        self.identity, self.phase = identity, "PREPARED"

    def require(self, identity: dict) -> None:
        if self.identity is None or identity != self.identity:
            raise ValueError("Admission installation identity changed")

    def check_workers(self, request: dict) -> None:
        if self.identity is None:
            raise ValueError("Claim installation before worker commands")
        for rank, instance in self.identity["workers"].items():
            worker = request["workers"][rank]
            if worker["instance"] != instance:
                raise ValueError("Worker incarnation differs from admission identity")
            if request["phase"] == "begin":
                candidate = request
            else:
                candidate = worker
            if any(candidate[key] != self.identity[key] for key in ("epoch", "installation_id", "root")):
                raise ValueError("Worker command differs from admission installation")


class AdmissionGate:
    def __init__(self) -> None:
        self.opened = False
        self.active = 0
        self.condition = asyncio.Condition()
        self.closed_at = time.monotonic()
        self.failure = None

    async def enter(self) -> bool:
        async with self.condition:
            if not self.opened:
                return False
            self.active += 1
            return True

    async def leave(self) -> None:
        async with self.condition:
            self.active -= 1
            self.condition.notify_all()

    async def close(self, timeout: float) -> None:
        async with self.condition:
            if self.opened:
                self.closed_at = time.monotonic()
                self.failure = None
            self.opened = False
            await asyncio.wait_for(self.condition.wait_for(lambda: self.active == 0), timeout)

    async def open(self) -> None:
        async with self.condition:
            self.opened = True
            self.failure = None

    def unavailable(self) -> dict:
        elapsed = time.monotonic() - self.closed_at
        return {
            "error": "Weight installation needs recovery" if self.failure else "Weight installation is in progress",
            "status": "WAITING_ACTION" if self.failure else "INSTALLING",
            "reason": self.failure,
            "closed_seconds": elapsed,
            "action": "inspect_consumer_status_and_recover_checkpoint" if self.failure else "retry_after_activation",
        }


class AdmissionMiddleware:
    def __init__(self, app, gate: AdmissionGate, fence=None, control=None) -> None:
        self.app, self.gate = app, gate
        self.fence, self.control = fence, control

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        mutation = path.startswith(("/update_weight", "/init_weights", "/destroy_weights")) or path in (
            "/release_memory_occupation",
            "/resume_memory_occupation",
            "/pause_generation",
            "/continue_generation",
            "/flush_cache",
            "/load_lora_adapter",
            "/unload_lora_adapter",
        )
        if scope["type"] == "http" and mutation and self.fence is not None:
            try:
                # The lock spans the existing endpoint, including device completion.
                await asyncio.wait_for(self.control.acquire(), 120)
            except TimeoutError:
                return await self._reject(send, "Control queue deadline exceeded")
            try:
                native_control = path in (
                    "/pause_generation",
                    "/continue_generation",
                    "/load_lora_adapter",
                    "/unload_lora_adapter",
                )
                if self.fence.identity is not None:
                    if native_control:
                        if not self.gate.opened:
                            raise ValueError(
                                "Installation is in progress; retry this engine operation after activation"
                            )
                    else:
                        headers = dict(scope.get("headers", []))
                        identity = json.loads(headers.get(b"x-relax-installation", b"null"))
                        self.fence.require(identity)
                        if self.gate.opened or path not in (
                            "/release_memory_occupation",
                            "/resume_memory_occupation",
                            "/flush_cache",
                        ):
                            raise ValueError("Use the checkpoint installation adapter for this weight operation")
            except (ValueError, KeyError, TypeError) as error:
                self.control.release()
                return await self._reject(send, str(error))
            try:
                return await self.app(scope, receive, send)
            finally:
                self.control.release()
        inference = path in ("/generate", "/encode", "/classify", "/score", "/health_generate") or path.startswith(
            ("/v1/", "/v2/", "/api/generate", "/api/chat", "/api/embed")
        )
        if scope["type"] != "http" or not inference:
            return await self.app(scope, receive, send)
        if not await self.gate.enter():
            body = json.dumps(self.gate.unavailable()).encode()
            headers = [(b"content-type", b"application/json")]
            if not self.gate.failure:
                headers.append((b"retry-after", b"1"))
            await send(
                {
                    "type": "http.response.start",
                    "status": 503,
                    "headers": headers,
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        try:
            await self.app(scope, receive, send)
        finally:
            await self.gate.leave()

    @staticmethod
    async def _reject(send, reason):
        await send({"type": "http.response.start", "status": 409, "headers": [(b"content-type", b"application/json")]})
        await send(
            {
                "type": "http.response.body",
                "body": json.dumps(
                    {
                        "error": reason,
                        "action": "use_current_installation_adapter",
                    }
                ).encode(),
            }
        )
