# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Client-side view of an inference role's topology.

``InferenceClient`` caches diagnostic topology. Resolving an address does not
authorize sending a request. ``InferenceRequestClient`` obtains a fresh permit
from the owning manager for each HTTP attempt.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import replace
from typing import Any, Awaitable, Callable, Hashable, Mapping, TypeVar

from relax.engine.inference.admission import RequestPermit
from relax.engine.inference.discovery import RoleSnapshot
from relax.engine.inference.routing import (
    ModelUnavailableError,
    RouteTarget,
    RoutingError,
    RoutingState,
    select_model,
    select_target,
)
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


T = TypeVar("T")


async def _settle(task: asyncio.Task[T]) -> T:
    """Finish a control RPC even if its caller is cancelled again."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
    return task.result()


async def _manager_call(manager: Any, method: str, *args: Any, **kwargs: Any) -> Any:
    call = getattr(manager, method)
    if hasattr(call, "remote"):
        return await call.remote(*args, **kwargs)
    # Also used by in-process callers and transport-independent tests.
    return await asyncio.to_thread(call, *args, **kwargs)


class InferenceRequest:
    """One permit, owned until the HTTP request (including its stream)
    closes."""

    def __init__(self, client: InferenceRequestClient, selection: dict[str, Any]) -> None:
        self._client = client
        self._selection = selection
        self._manager = None
        self._permit: RequestPermit | None = None
        self._release_task: asyncio.Task[None] | None = None
        self.target: RouteTarget | None = None
        self.served_model_name: str | None = None

    async def __aenter__(self) -> InferenceRequest:
        from relax.engine.inference.holder import current_holder

        resources = await self._client.resources()
        snapshot = RoleSnapshot.from_dict(resources["snapshot"])
        selection = dict(self._selection)
        selection["holder"] = current_holder()
        selected = select_model(snapshot, selection.pop("model", None), selection.pop("route_key", None))
        self._manager = resources["managers"][selected.name]
        pending = asyncio.create_task(
            _manager_call(self._manager, "acquire_inference", uuid.uuid4().hex, model=selected.name, **selection)
        )
        try:
            self._permit = await asyncio.shield(pending)
        except asyncio.CancelledError as cancelled:
            # Cancellation does not retract an already queued Ray method. Wait
            # for its outcome before returning a permit that arrived late.
            try:
                self._permit = await _settle(pending)
            except Exception:
                raise cancelled from None
            finally:
                await self.aclose()
            raise
        except RoutingError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(f"Admission for {self._client.role!r} is unavailable: {exc}") from exc
        self.target = replace(self._permit.target, model=selected.name)
        self.served_model_name = selected.served_model_name
        return self

    async def aclose(self) -> None:
        if self._permit is None:
            return
        if self._release_task is None:
            self._release_task = asyncio.create_task(_manager_call(self._manager, "release_inference", self._permit))
        try:
            await asyncio.shield(self._release_task)
        except asyncio.CancelledError:
            await _settle(self._release_task)
            raise

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()


class InferenceRequestClient:
    """Lookaside control client. Request bodies never pass through this client.

    Bootstrap once through the role's Serve facade, then acquire/release on the
    owning manager's independent concurrency group. Only identities and manager
    handles are cached; READY and replica addresses are never permits.
    """

    def __init__(self, role: str, resources: Callable[[], Awaitable[dict[str, Any]]] | None = None) -> None:
        self.role = role
        self._resources_source = resources
        self._resources: dict[str, Any] | None = None
        self._bootstraps: dict[asyncio.AbstractEventLoop, asyncio.Task[dict[str, Any]]] = {}

    async def _bootstrap(self) -> dict[str, Any]:
        if self._resources_source is not None:
            resources = await self._resources_source()
        else:
            from ray import serve

            resources = await serve.get_app_handle(self.role).get_inference_resources.remote()
        self._resources = resources
        return resources

    async def resources(self) -> dict[str, Any]:
        if self._resources is not None:
            return self._resources
        loop = asyncio.get_running_loop()
        pending = self._bootstraps.get(loop)
        if pending is None:
            pending = self._bootstraps[loop] = asyncio.create_task(self._bootstrap())

            def finished(task: asyncio.Task[dict[str, Any]]) -> None:
                self._bootstraps.pop(loop, None)
                # Bootstrap may outlive every cancelled waiter. Consume a
                # failure then too; the next call will retry discovery.
                if not task.cancelled():
                    task.exception()

            pending.add_done_callback(finished)
        try:
            return await asyncio.shield(pending)
        except Exception as exc:
            raise ModelUnavailableError(f"Admission for {self.role!r} is unavailable: {exc}") from exc

    def request(self, **selection: Any) -> InferenceRequest:
        return InferenceRequest(self, selection)


# Handles are independent of asyncio loops; request/cleanup tasks are not cached.
_REQUEST_CLIENTS: dict[str, InferenceRequestClient] = {}


def get_request_client(role: str) -> InferenceRequestClient:
    client = _REQUEST_CLIENTS.get(role)
    if client is None:
        client = _REQUEST_CLIENTS[role] = InferenceRequestClient(role)
    return client
