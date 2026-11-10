# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from typing import Any


class NativeWeightLease:
    """Keep native allocation mutations excluded until an entire update
    commits.

    The backend installs the actual offload/update guards. Clearing one NIXL
    registration is insufficient: the adapter must agree across its workers
    before calling commit. Failures never unlock outstanding native memory.
    """

    def __init__(self) -> None:
        self.active: tuple[str, int, str] | None = None
        self.pending: set[Any] = set()
        self.failed = False

    def begin(self, session: str, version: int, delivery_hash: str) -> tuple[str, int, str]:
        if self.failed or self.active is not None or self.pending:
            raise RuntimeError("Native weight lease is failed or active")
        if (
            not isinstance(session, str)
            or not session
            or type(version) is not int
            or version < 0
            or not isinstance(delivery_hash, str)
            or not delivery_hash
        ):
            raise ValueError("Invalid native weight lease identity")
        self.active = (session, version, delivery_hash)
        return self.active

    def add(self, target: Any, version: int) -> None:
        if self.failed or self.active is None or self.active[1] != version or target in self.pending:
            raise RuntimeError("Native target has no matching allocation lease")
        self.pending.add(target)

    def complete(self, target: Any) -> None:
        if self.failed or self.active is None or target not in self.pending:
            raise RuntimeError("Native target has no outstanding allocation lease")
        self.pending.remove(target)

    def ready(self, identity: tuple[str, int, str]) -> None:
        if self.failed or self.active != identity or self.active is None or self.pending:
            raise RuntimeError("Native weight lease cannot commit before all targets finish")

    def commit(self, identity: tuple[str, int, str]) -> None:
        self.ready(identity)
        self.active = None

    def fail(self) -> None:
        self.failed = True
