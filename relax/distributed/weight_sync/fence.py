# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Serialize engine commands and reject stale coordinators before mutation."""

import threading
import uuid
from typing import Callable

from .store import canonical, digest


class WorkerFence:
    def __init__(self) -> None:
        self.instance = uuid.uuid4().hex
        self.epoch = -1
        self.installation_id: str | None = None
        self.root: str | None = None
        self._responses: dict[str, tuple[str, object]] = {}
        self._commands: dict[str, str] = {}
        self.phase = "NEW"
        self._lock = threading.RLock()

    def begin(self, instance: str, epoch: int, installation_id: str, root: str) -> dict:
        with self._lock:
            if instance != self.instance or type(epoch) is not int or epoch < self.epoch:
                raise ValueError("Stale worker instance or coordinator epoch")
            if epoch == self.epoch:
                if (installation_id, root) != (self.installation_id, self.root):
                    raise ValueError("Coordinator epoch was reused for another installation")
            else:
                self.epoch, self.installation_id, self.root = epoch, installation_id, root
                self._responses.clear()
                self._commands.clear()
                self.phase = "PREPARED"
            return self.identity()

    def identity(self) -> dict:
        return {
            "instance": self.instance,
            "epoch": self.epoch,
            "installation_id": self.installation_id,
            "root": self.root,
        }

    def validate(self, identity: dict, command_id: str, phase: str, payload: dict | None = None) -> str:
        if identity != self.identity():
            raise ValueError("Command does not belong to the current worker installation")
        signature = digest(canonical([phase, payload]))
        if command_id in self._commands and self._commands[command_id] != signature:
            raise ValueError("Command ID was reused with different content")
        if command_id not in self._responses:
            required = {"verify": {"load", "verify"}, "flush": {"verify", "flush"}}
            if phase in required and self.phase not in required[phase]:
                raise ValueError(f"Worker phase {self.phase} cannot execute {phase}")
        self._commands[command_id] = signature
        return signature

    def execute(
        self,
        identity: dict,
        command_id: str,
        phase: str,
        operation: Callable[[], object],
        *,
        payload: dict | None = None,
    ) -> dict:
        with self._lock:
            signature = self.validate(identity, command_id, phase, payload)
            previous = self._responses.get(command_id)
            if previous is not None:
                return {"identity": self.identity(), "phase": phase, "command_id": command_id, "result": previous[1]}
            result = operation()
            self._responses[command_id] = (signature, result)
            if phase in ("load", "verify", "flush"):
                self.phase = phase
            return {"identity": self.identity(), "phase": phase, "command_id": command_id, "result": result}
