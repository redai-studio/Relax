# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Client-side view of an inference role's topology.

``InferenceClient`` caches a role's snapshot and resolves requests to an
address with the same routing rules a gateway uses. Where the snapshot comes
from is injected (a manager handle, an HTTP GET, a test fake), so the class has
no transport of its own.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Hashable, Mapping

from relax.engine.inference.discovery import RoleSnapshot
from relax.engine.inference.routing import RouteTarget, RoutingState, select_target
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

SnapshotSource = Callable[[], "RoleSnapshot | Mapping[str, Any]"]


class InferenceClient:
    def __init__(
        self,
        fetch_snapshot: SnapshotSource,
        *,
        max_age_s: float = 0.0,
        refresh_cooldown_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            fetch_snapshot: Returns the role's current snapshot (object or dict).
            max_age_s: Re-fetch a snapshot older than this, which is how a
                ``topology_revision`` change is noticed. ``0`` re-fetches only
                after ``report_failure``.
            refresh_cooldown_s: Minimum spacing between re-fetches triggered by
                ``report_failure``, so a burst of failing requests against one
                dead replica costs a single fetch. Also spaces out retries
                while no snapshot could be fetched at all.
        """
        self._fetch_snapshot = fetch_snapshot
        self._max_age_s = max_age_s
        self._refresh_cooldown_s = refresh_cooldown_s
        self._clock = clock
        self._state = RoutingState()
        self._snapshot: RoleSnapshot | None = None
        self._fetched_at = 0.0
        self._attempted = False
        self._stale = True
        # Concurrent callers that find the snapshot stale share one fetch.
        self._refresh_lock = threading.Lock()

    def snapshot(self) -> RoleSnapshot:
        if self.needs_refresh():
            with self._refresh_lock:
                if self.needs_refresh():
                    self._refresh()
        if self._snapshot is None:
            raise RuntimeError("No inference topology snapshot is available.")
        return self._snapshot

    @property
    def last_snapshot(self) -> RoleSnapshot | None:
        """The snapshot last fetched, without fetching; ``None`` before the
        first successful fetch."""
        return self._snapshot

    def resolve(
        self, *, model: str | None = None, route_key: str | None = None, affinity_key: Hashable | None = None
    ) -> RouteTarget:
        return select_target(self.snapshot(), self._state, model=model, route_key=route_key, affinity_key=affinity_key)

    def report_failure(self) -> None:
        """Tell the client a request to a resolved address could not connect.

        The next ``snapshot``/``resolve`` re-fetches, unless one was fetched
        within the cooldown.
        """
        if self._snapshot is not None and self._clock() - self._fetched_at < self._refresh_cooldown_s:
            return
        self._stale = True

    def needs_refresh(self) -> bool:
        """Whether the next ``snapshot`` call would fetch."""
        if self._snapshot is None:
            return not self._attempted or self._clock() - self._fetched_at >= self._refresh_cooldown_s
        if self._stale:
            return True
        return self._max_age_s > 0 and self._clock() - self._fetched_at >= self._max_age_s

    def _refresh(self) -> None:
        try:
            fetched = self._fetch_snapshot()
        except Exception as exc:
            if self._snapshot is None:
                self._attempted = True
                self._fetched_at = self._clock()
                raise
            # Keep routing on the last known topology rather than failing every
            # request while discovery is briefly unreachable.
            logger.warning(
                f"Inference topology refresh failed; keeping revision {self._snapshot.topology_revision}: {exc}"
            )
            self._fetched_at = self._clock()
            self._stale = False
            return

        snapshot = fetched if isinstance(fetched, RoleSnapshot) else RoleSnapshot.from_dict(fetched)
        if self._snapshot is not None and snapshot.topology_revision != self._snapshot.topology_revision:
            logger.info(
                f"Inference topology of role {snapshot.role!r} changed: "
                f"revision {self._snapshot.topology_revision} -> {snapshot.topology_revision}"
            )
        self._snapshot = snapshot
        self._fetched_at = self._clock()
        self._stale = False
