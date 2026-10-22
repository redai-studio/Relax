# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Direct Agent writes sharing one filesystem lock, budget and task quotas."""

from pathlib import Path

from .store import ArchiveStore


class DirectoryTransport:
    """Own the store only during one operation; never hold a lock over a model
    call.

    All nodes must mount the same directory and honor cross-node flock. A local
    directory on each node gives separate budgets and is not a shared archive.
    """

    def __init__(
        self,
        root: str | Path,
        max_bytes: int,
        per_task_limit: int = 5,
        task_limits: dict[str, int] | None = None,
        lock_timeout: float = 30.0,
    ) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.per_task_limit = per_task_limit
        self.task_limits = task_limits or {}
        self.lock_timeout = lock_timeout

    def __call__(self, operation: str, payload: dict) -> dict:
        store = ArchiveStore(
            self.root,
            self.max_bytes,
            self.per_task_limit,
            task_limits=self.task_limits,
            allow_shared_filesystem=True,
            lock_timeout=self.lock_timeout,
            new_asset_layout="task",
        )
        try:
            identity = payload.get("trajectory_id")
            if operation == "begin":
                metadata = payload["metadata"]
                if self.task_limits.get(metadata["task_id"], self.per_task_limit) == 0:
                    raise ValueError("task_quota: task disabled")
                return {"trajectory_id": store.begin(metadata)}
            if operation == "event":
                store.append(identity, payload["event"])
            elif operation == "finish":
                return store.finish(identity, payload["outcome"])
            elif operation == "abort":
                store.abort(identity)
            else:
                raise ValueError(f"unknown archive operation: {operation}")
            return {"ok": True}
        finally:
            store.close()
