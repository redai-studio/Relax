# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Request execution shared by every inference entry point.

One request runs as: resolve its model and target from the Manager snapshot ->
take a Manager permit -> send -> release (abort what the engine may still run,
then return the permit exactly once). The Gateway and the GenRM backend
``/generate`` both run their requests through :class:`InferenceRequestExecutor`;
only their payload adaptation differs.

Retry contract: a send that fails before a connection carried the request
(connect error or timeout, pool timeout) re-resolves the target and is retried,
since no engine saw it. Once the request was sent and its outcome is unknown
(read error or timeout, reset, a 5xx reply) it is never replayed: engines have
no idempotent request protocol, so it is aborted and the failure reported.
"""

import asyncio
import inspect
import itertools
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import ray

from relax.engine.inference.routing import RoutingError, resolve_model, select_target
from relax.engine.inference.types import Role, RoleSnapshot
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)

_ABORT_TIMEOUT_S = 10.0
_RETRY_BACKOFF_S = 0.3

# Failures raised before any byte of the request reached the upstream.
_NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


@dataclass
class AdmittedRequest:
    """One request holding a Manager permit.

    ``rid`` is the backend ID namespace of this one execution; see
    :func:`assign_backend_rids`. ``dispatched`` means an engine may have seen
    the request; ``finished`` means the upstream reported its outcome. A
    dispatched request that never finished is aborted on release, by
    ``engine_rids`` when set (its logical IDs), else by ``rid``.
    """

    model_id: str
    target: str
    via_router: bool
    permit: str
    rid: str
    engine_rids: tuple[str, ...] | None = None
    # The caller's ID for logs; never sent to an engine.
    trace_id: str | None = None
    dispatched: bool = False
    finished: bool = False
    released: bool = False


# SGLang's scheduler aborts every request whose rid starts with the aborted
# one, so each execution owns a namespace no other rid can share or prefix: a
# plain random rid cannot start with this marker.
_BACKEND_RID_PREFIX = "relax-inf-"


def new_backend_rid() -> str:
    return f"{_BACKEND_RID_PREFIX}{uuid.uuid4().hex}"


def _batch_size(payload: Mapping[str, Any]) -> int | None:
    """Prompts of a batched native or completions payload, else ``None``.

    Follows SGLang's ``GenerateReqInput``: the first input present decides, in
    the order text (``prompt`` for completions), ``input_ids``,
    ``input_embeds``.
    """
    for key in ("text", "prompt"):
        value = payload.get(key)
        if value is not None:
            return len(value) if isinstance(value, list) and value and not isinstance(value[0], int) else None
    input_ids = payload.get("input_ids")
    if input_ids is not None:
        return len(input_ids) if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list) else None
    embeds = payload.get("input_embeds")
    # One prompt is a list of vectors; a batch nests one more level. SGLang's
    # HTTP layer coerces integer entries to floats first, so only depth counts.
    if isinstance(embeds, list) and embeds and isinstance(embeds[0], list) and embeds[0]:
        return len(embeds) if isinstance(embeds[0][0], list) else None
    return None


def assign_backend_rids(request: AdmittedRequest, payload: dict[str, Any]) -> None:
    """Replace ``payload["rid"]`` with this execution's logical IDs.

    A batch gets an explicit list, which SGLang keeps as is instead of
    normalizing a string rid; a single prompt keeps the bare namespace. Abort
    names these logical IDs, which also covers ``n > 1``: the TokenizerManager
    aborts a logical ID's parallel-sampling children.
    """
    size = _batch_size(payload)
    rids = (request.rid,) if size is None else tuple(f"{request.rid}_{index}" for index in range(size))
    payload["rid"] = rids[0] if size is None else list(rids)
    request.engine_rids = rids


class InferenceRequestExecutor:
    """Runs admitted requests of one role against the Manager's targets.

    The Manager behind ``manager_handle`` is the only state source; the
    executor keeps none besides a round-robin cursor.
    """

    def __init__(self, role: str | Role, manager_handle: Any, client: httpx.AsyncClient) -> None:
        self.role = Role(role)
        self.manager_handle = manager_handle
        self.client = client
        # Rotates across direct-eligible replicas; a Router target ignores it.
        self._cursor = itertools.count()

    async def manager_call(self, method: str, *args: Any) -> Any:
        value = getattr(self.manager_handle, method).remote(*args)
        if inspect.isawaitable(value):
            return await value
        return await asyncio.to_thread(ray.get, value)

    async def snapshot(self) -> RoleSnapshot:
        return await self.manager_call("snapshot", self.role)

    async def resolve(self, *, model: str | None = None, route_key: str | None = None) -> tuple[str, str, bool]:
        """Return ``(target, model_id, via_router)``, or raise
        :class:`RoutingError`."""
        snapshot = await self.snapshot()
        selected = resolve_model(snapshot, model=model, route_key=route_key)
        target = select_target(selected, cursor=next(self._cursor)).base_url.rstrip("/")
        return target, selected.model_id, target == (selected.router_url or "").rstrip("/")

    async def admit(self, trace_id: str, *, model: str | None = None, route_key: str | None = None) -> AdmittedRequest:
        """Resolve a target and take a permit for it.

        Raises :class:`RoutingError` when no target resolves or the Manager
        refuses the admission (503, e.g. the model is draining).
        """
        target, model_id, via_router = await self.resolve(model=model, route_key=route_key)
        call = asyncio.ensure_future(self.manager_call("admit_request", self.role, model_id, trace_id))
        try:
            permit = await asyncio.shield(call)
        except asyncio.CancelledError:
            # The Manager may already hold the permit while its reply is in
            # flight; no caller cleanup sees it yet, so return it here.
            await self._return_unclaimed_permit(call, trace_id)
            raise
        except Exception as exc:
            logger.warning("Inference request %s admission failed: %s", trace_id, exc)
            raise RoutingError(
                "unavailable", "Inference request is not admitted", status_code=503, model_id=model_id
            ) from exc
        return AdmittedRequest(
            model_id=model_id,
            target=target,
            via_router=via_router,
            permit=permit,
            rid=new_backend_rid(),
            trace_id=trace_id,
        )

    async def _return_unclaimed_permit(self, call: asyncio.Future, trace_id: str) -> None:
        """Wait for an admission its caller abandoned, then return its permit.

        Further cancellations are absorbed: the caller re-raises its own once
        the permit is back, so a drain never waits on a leaked admission.
        """
        while True:
            try:
                permit = await asyncio.shield(call)
                break
            except asyncio.CancelledError:
                if call.cancelled():
                    return  # Cancelled before the Manager was called.
            except Exception:
                return  # Refused: no permit was recorded.
        complete = asyncio.ensure_future(self.manager_call("complete_request", permit))
        while True:
            try:
                await asyncio.shield(complete)
                return
            except asyncio.CancelledError:
                if complete.cancelled():
                    return
            except Exception as exc:
                logger.warning("Failed to return abandoned permit %s of request %s: %s", permit, trace_id, exc)
                return

    async def send(self, request: AdmittedRequest, build: Callable[[str], httpx.Request]) -> httpx.Response:
        """Send ``build(target)`` and return the unread, streamed response.

        Only failures before the request was sent are retried, each against a
        freshly resolved target of the same model; see the module docstring.
        """
        attempts = max(1, self._connect_attempts())
        for attempt in range(1, attempts + 1):
            # Set first: a cancellation mid-send may already have reached an engine.
            request.dispatched = True
            try:
                return await self.client.send(build(request.target), stream=True)
            except _NOT_SENT_ERRORS as exc:
                request.dispatched = False
                if attempt >= attempts:
                    raise
                logger.warning(
                    "Inference request %s could not reach %s (attempt %d/%d): %s",
                    request.rid,
                    request.target,
                    attempt,
                    attempts,
                    exc,
                )
                await asyncio.sleep(_RETRY_BACKOFF_S * attempt)
                # The endpoint may be gone and recovered elsewhere; a model that
                # stopped admitting meanwhile raises RoutingError here.
                request.target, _, request.via_router = await self.resolve(model=request.model_id)
        raise AssertionError("unreachable")

    async def read(self, request: AdmittedRequest, response: httpx.Response) -> bytes:
        """Read and close a non-streamed response.

        Only a reply below 500 reports the outcome; a 5xx may come from a
        Router or engine that still runs the request.
        """
        try:
            content = await response.aread()
        finally:
            await response.aclose()
        request.finished = response.status_code < 500
        return content

    async def post(self, request: AdmittedRequest, path: str, payload: dict[str, Any]) -> httpx.Response:
        """Send one JSON POST and return its read response."""
        response = await self.send(
            request, lambda target: self.client.build_request("POST", f"{target}/{path.lstrip('/')}", json=payload)
        )
        await self.read(request, response)
        return response

    async def release(self, request: AdmittedRequest) -> None:
        """Abort what may still run, then return the permit, exactly once."""
        if request.released:
            return
        request.released = True
        try:
            if request.dispatched and not request.finished:
                await self.abort(request)
        finally:
            try:
                await self.manager_call("complete_request", request.permit)
            except Exception as exc:
                logger.warning(
                    "Failed to complete inference request %s (permit %s): %s", request.rid, request.permit, exc
                )

    async def abort(self, request: AdmittedRequest) -> None:
        """Send an abort for the request to the engines that may run it.

        The router does not proxy aborts, so it is broadcast to its workers,
        the same way the rollout and Agentic paths do it; a direct replica is
        aborted itself. An ACK only proves the scheduler received it, never
        that the request left the running batch.
        """
        rids = (request.rid,) if request.engine_rids is None else request.engine_rids
        if not rids:
            return
        if request.via_router:
            try:
                urls = await self._worker_base_urls(request.target)
            except Exception as exc:
                logger.warning("Could not resolve engines behind %s to abort %s: %s", request.target, request.rid, exc)
                return
        else:
            urls = [request.target]
        # SGLang matches an abort against one exact rid.
        targets = [(url, rid) for url in urls for rid in rids]
        results = await asyncio.gather(
            *(
                self.client.post(f"{url}/abort_request", json={"rid": rid}, timeout=_ABORT_TIMEOUT_S)
                for url, rid in targets
            ),
            return_exceptions=True,
        )
        for (url, rid), result in zip(targets, results, strict=False):
            if isinstance(result, BaseException):
                logger.warning("Abort of %s at %s failed: %s", rid, url, result)

    async def _worker_base_urls(self, target: str) -> list[str]:
        from relax.utils.http_utils import router_worker_base_urls

        try:
            response = await self.client.get(f"{target}/workers", timeout=_ABORT_TIMEOUT_S)
            response.raise_for_status()
            urls = [worker["url"] for worker in response.json()["workers"] if worker.get("url")]
        except Exception:
            response = await self.client.get(f"{target}/list_workers", timeout=_ABORT_TIMEOUT_S)
            response.raise_for_status()
            urls = list(response.json()["urls"])
        return router_worker_base_urls(urls)

    def _connect_attempts(self) -> int:
        return Envs.GENRM_ENGINE_RETRY_ATTEMPTS if self.role == Role.GENRM else 1


__all__ = ["AdmittedRequest", "InferenceRequestExecutor", "assign_backend_rids", "new_backend_rid"]
