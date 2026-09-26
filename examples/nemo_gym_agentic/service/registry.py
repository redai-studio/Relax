# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""In-memory run registry for the Phase-2 shared Gateway spike."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..app.protocol import PROTOCOL_VERSION, TrialRequest, TrialStatus
from .config import EnvironmentSpec, GatewayConfigError, GatewaySettings, validate_callback_url
from .run_adapter import AdapterResult, CleanupResult, RunAdapter, RunHandle, TrialRunContext


logger = logging.getLogger("uvicorn.error")


class TrialNotFound(KeyError):
    pass


class TrialConflict(RuntimeError):
    pass


class AdmissionRejected(RuntimeError):
    pass


class CallbackUnavailable(RuntimeError):
    pass


class InternalTrialState(str, Enum):
    ACTIVE = "active"
    CANCELLING = "cancelling"
    FINALIZING = "finalizing"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class CallbackTarget:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    api_key_header: str
    api_key_prefix: str = field(repr=False)
    headers: dict[str, str] = field(repr=False)
    generation: dict[str, Any]
    remaining_s: float


@dataclass
class TrialRecord:
    request_id: str
    payload_fingerprint: str
    spec: EnvironmentSpec
    rollout_id: str
    created_at: float
    deadline_at: float
    lease_expires_at: float
    request: TrialRequest | None = field(repr=False)
    public_status: TrialStatus = TrialStatus.QUEUED
    internal_state: InternalTrialState = InternalTrialState.ACTIVE
    reward: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_ref: str | None = None
    error: dict[str, Any] | None = None
    handle: RunHandle | None = field(default=None, repr=False)
    supervisor_task: asyncio.Task[None] | None = field(default=None, repr=False)
    cleanup_task: asyncio.Task[None] | None = field(default=None, repr=False)
    callback_tasks: set[asyncio.Task[Any]] = field(default_factory=set, repr=False)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.internal_state is InternalTrialState.TERMINAL


class GatewayRegistry:
    """Owns admission, trial races, leases, deadlines, and callback
    credentials.

    This implementation intentionally uses process memory. It is suitable for
    the Phase-2 spike only and must run with one uvicorn worker.
    """

    def __init__(
        self,
        *,
        settings: GatewaySettings,
        adapter: RunAdapter,
        clock: Any = time.monotonic,
    ) -> None:
        self.settings = settings
        self._adapter = adapter
        self._clock = clock
        self._fingerprint_key = secrets.token_bytes(32)
        self._records: dict[str, TrialRecord] = {}
        self._callbacks: dict[str, TrialRecord] = {}
        self._active_by_environment: dict[tuple[str, str], int] = {}
        self._semaphores = {
            key: asyncio.Semaphore(spec.max_concurrency) for key, spec in settings.environments.items()
        }
        self._lock = asyncio.Lock()
        self._janitor_task: asyncio.Task[None] | None = None
        self._janitor_healthy = True
        self._closing = False
        self.service_epoch = secrets.token_hex(16)

    async def start(self) -> None:
        if self._janitor_task is None:
            self._janitor_task = asyncio.create_task(self._janitor_loop())

    async def close(self) -> None:
        self._closing = True
        records = list(self._records.values())
        for record in records:
            await self._begin_cancellation(record, cause="service_shutdown", status=TrialStatus.ABORTED)
        cleanup_tasks = [record.cleanup_task for record in records if record.cleanup_task is not None]
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        if self._janitor_task is not None:
            self._janitor_task.cancel()
            await asyncio.gather(self._janitor_task, return_exceptions=True)
            self._janitor_task = None
        await self._adapter.close()

    async def ready(self) -> bool:
        return (
            not self._closing
            and self._janitor_task is not None
            and not self._janitor_task.done()
            and self._janitor_healthy
            and await self._adapter.ready()
        )

    async def create(self, payload: Any) -> dict[str, Any]:
        request = TrialRequest.from_payload(payload)
        spec = self._resolve_spec(request)
        validate_callback_url(
            request.model_endpoint.base_url,
            self.settings.callback_allowed_hosts,
            allowed_networks=self.settings.callback_allowed_networks,
            require_tls=self.settings.callback_proxy is not None,
        )
        fingerprint = self._payload_fingerprint(payload)
        now = self._clock()
        key = (spec.environment, spec.config)

        async with self._lock:
            existing = self._records.get(request.request_id)
            if existing is not None:
                if not hmac.compare_digest(existing.payload_fingerprint, fingerprint):
                    raise TrialConflict("request_id already exists with a different payload")
                return self._snapshot(existing)

            capacity = spec.max_concurrency + spec.queue_capacity
            active = self._active_by_environment.get(key, 0)
            if active >= capacity:
                raise AdmissionRejected("environment queue is full")

            rollout_id = f"{secrets.token_hex(16)}-0"
            record = TrialRecord(
                request_id=request.request_id,
                payload_fingerprint=fingerprint,
                spec=spec,
                rollout_id=rollout_id,
                created_at=now,
                deadline_at=now + request.deadline_s,
                lease_expires_at=now + request.lease_s,
                request=request,
            )
            self._records[request.request_id] = record
            self._callbacks[rollout_id] = record
            self._active_by_environment[key] = active + 1
            record.supervisor_task = asyncio.create_task(self._supervise(record))
            return self._snapshot(record)

    async def get(self, request_id: str) -> dict[str, Any]:
        return self._snapshot(await self._get_record(request_id))

    async def renew(self, request_id: str) -> None:
        record = await self._get_record(request_id)
        async with record.lock:
            if record.internal_state is not InternalTrialState.ACTIVE:
                return
            if record.request is None:
                return
            record.lease_expires_at = self._clock() + record.request.lease_s

    async def abort(self, request_id: str) -> dict[str, Any]:
        record = await self._get_record(request_id)
        await self._begin_cancellation(record, cause="aborted", status=TrialStatus.ABORTED)
        return self._snapshot(record)

    @asynccontextmanager
    async def callback_target(self, rollout_id: str) -> AsyncIterator[CallbackTarget]:
        record = self._callbacks.get(rollout_id)
        if record is None:
            raise CallbackUnavailable("callback capability is unknown or no longer active")
        callback_task = asyncio.current_task()
        if callback_task is None:
            raise CallbackUnavailable("callback task is unavailable")
        target: CallbackTarget | None = None
        expired: tuple[str, TrialStatus] | None = None
        async with record.lock:
            if record.internal_state is not InternalTrialState.ACTIVE or record.request is None:
                raise CallbackUnavailable("callback capability is no longer active")
            now = self._clock()
            if now >= record.deadline_at:
                expired = ("deadline_exceeded", TrialStatus.TRUNCATED)
            elif now >= record.lease_expires_at:
                expired = ("lease_expired", TrialStatus.ABORTED)
            else:
                endpoint = record.request.model_endpoint
                target = CallbackTarget(
                    base_url=endpoint.base_url,
                    api_key=endpoint.api_key,
                    model=endpoint.model,
                    api_key_header=endpoint.api_key_header,
                    api_key_prefix=endpoint.api_key_prefix,
                    headers=copy.deepcopy(endpoint.headers),
                    generation=copy.deepcopy(record.request.generation),
                    remaining_s=record.deadline_at - now,
                )
                record.callback_tasks.add(callback_task)
        if expired is not None:
            await self._begin_cancellation(record, cause=expired[0], status=expired[1])
            raise CallbackUnavailable("callback capability has expired")
        assert target is not None
        try:
            yield target
        finally:
            async with record.lock:
                record.callback_tasks.discard(callback_task)

    def stats(self) -> dict[str, int | bool]:
        terminal = sum(record.is_terminal for record in self._records.values())
        return {
            "trials": len(self._records),
            "active_trials": len(self._records) - terminal,
            "terminal_trials": terminal,
            "janitor_healthy": self._janitor_healthy,
        }

    async def _get_record(self, request_id: str) -> TrialRecord:
        record = self._records.get(request_id)
        if record is None:
            raise TrialNotFound(request_id)
        return record

    def _resolve_spec(self, request: TrialRequest) -> EnvironmentSpec:
        spec = self.settings.environments.get((request.environment, request.config))
        if spec is None:
            raise GatewayConfigError("environment/config is not registered")
        if request.interrupt_policy != spec.interrupt_policy:
            raise GatewayConfigError("interrupt_policy does not match the registered environment capability")
        if request.deadline_s > spec.max_deadline_s:
            raise GatewayConfigError("deadline_s exceeds the registered environment maximum")
        return spec

    async def _supervise(self, record: TrialRecord) -> None:
        semaphore = self._semaphores[(record.spec.environment, record.spec.config)]
        acquired = False
        try:
            remaining = max(0.0, record.deadline_at - self._clock())
            await asyncio.wait_for(semaphore.acquire(), timeout=remaining)
            acquired = True
            async with record.lock:
                if record.internal_state is not InternalTrialState.ACTIVE or record.request is None:
                    return
                record.public_status = TrialStatus.RUNNING
                context = TrialRunContext(
                    request=record.request,
                    spec=record.spec,
                    rollout_id=record.rollout_id,
                )
                # RunAdapter.start must only allocate a tracked handle. Keeping
                # it under the trial lock prevents abort from observing a
                # remotely-started run before its handle is registered.
                handle = await self._adapter.start(context)
                record.handle = handle

            remaining = max(0.0, record.deadline_at - self._clock())
            result = await asyncio.wait_for(handle.wait(), timeout=remaining)
            await self._finalize_from_adapter(record, result)
        except TimeoutError:
            await self._begin_cancellation(
                record,
                cause="deadline_exceeded",
                status=TrialStatus.TRUNCATED,
            )
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception(
                "NeMo Gym trial failed request_id=%s rollout_id=%s environment=%s config=%s",
                record.request_id,
                record.rollout_id,
                record.spec.environment,
                record.spec.config,
            )
            await self._begin_cancellation(
                record,
                cause="agent_error",
                status=TrialStatus.FAILED,
            )
        finally:
            if acquired:
                semaphore.release()

    async def _finalize_from_adapter(self, record: TrialRecord, result: AdapterResult) -> None:
        async with record.lock:
            if record.internal_state is not InternalTrialState.ACTIVE:
                return
            record.internal_state = InternalTrialState.FINALIZING
            record.public_status = result.status
            record.reward = copy.deepcopy(result.reward)
            record.metrics = copy.deepcopy(result.metrics)
            record.artifact_ref = result.artifact_ref
            record.error = copy.deepcopy(result.error)
            self._mark_terminal(record)

    async def _begin_cancellation(
        self,
        record: TrialRecord,
        *,
        cause: str,
        status: TrialStatus,
    ) -> None:
        async with record.lock:
            if record.internal_state in {
                InternalTrialState.CANCELLING,
                InternalTrialState.FINALIZING,
                InternalTrialState.TERMINAL,
            }:
                return
            record.internal_state = InternalTrialState.CANCELLING
            self._callbacks.pop(record.rollout_id, None)
            record.cleanup_task = asyncio.create_task(self._cleanup(record, cause=cause, status=status))

    async def _cleanup(self, record: TrialRecord, *, cause: str, status: TrialStatus) -> None:
        supervisor = record.supervisor_task
        if supervisor is not None and supervisor is not asyncio.current_task():
            supervisor.cancel()
            await asyncio.gather(supervisor, return_exceptions=True)

        cleanup_result = CleanupResult(confirmed=True)
        callback_tasks = [
            task for task in record.callback_tasks if task is not asyncio.current_task() and not task.done()
        ]
        for task in callback_tasks:
            task.cancel()
        if callback_tasks:
            await asyncio.gather(*callback_tasks, return_exceptions=True)

        if record.handle is not None:
            try:
                cleanup_result = await asyncio.wait_for(
                    record.handle.abort(),
                    timeout=self.settings.cleanup_grace_s,
                )
                if not cleanup_result.confirmed:
                    cleanup_result = await asyncio.wait_for(
                        record.handle.force_cleanup(),
                        timeout=self.settings.cleanup_grace_s,
                    )
                if not cleanup_result.confirmed:
                    cleanup_result = await asyncio.wait_for(
                        record.handle.probe_cleanup(),
                        timeout=self.settings.cleanup_grace_s,
                    )
            except (Exception, TimeoutError):
                cleanup_result = CleanupResult(confirmed=False, error_code="cleanup_failed")

        async with record.lock:
            if record.internal_state is not InternalTrialState.CANCELLING:
                return
            record.internal_state = InternalTrialState.FINALIZING
            if cleanup_result.confirmed:
                record.public_status = status
                record.error = {"code": cause}
            else:
                record.public_status = TrialStatus.FAILED
                record.error = {
                    "code": cleanup_result.error_code or "cleanup_unverified",
                    "type": "CleanupError",
                    "cause": cause,
                }
            self._mark_terminal(record)

    def _mark_terminal(self, record: TrialRecord) -> None:
        record.internal_state = InternalTrialState.TERMINAL
        record.request = None
        record.handle = None
        self._callbacks.pop(record.rollout_id, None)
        key = (record.spec.environment, record.spec.config)
        self._active_by_environment[key] = max(0, self._active_by_environment.get(key, 1) - 1)

    async def _janitor_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.lease_scan_interval_s)
            now = self._clock()
            for record in list(self._records.values()):
                try:
                    if record.internal_state is not InternalTrialState.ACTIVE:
                        continue
                    if now >= record.deadline_at:
                        await self._begin_cancellation(
                            record,
                            cause="deadline_exceeded",
                            status=TrialStatus.TRUNCATED,
                        )
                    elif now >= record.lease_expires_at:
                        await self._begin_cancellation(
                            record,
                            cause="lease_expired",
                            status=TrialStatus.ABORTED,
                        )
                except Exception:
                    # Keep the janitor alive so one malformed adapter/record
                    # cannot disable lease handling for every other trial.
                    self._janitor_healthy = False
                    continue

    def _payload_fingerprint(self, payload: Any) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hmac.new(self._fingerprint_key, canonical, hashlib.sha256).hexdigest()

    @staticmethod
    def _snapshot(record: TrialRecord) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": record.request_id,
            "status": record.public_status.value,
            "reward": copy.deepcopy(record.reward),
            "metrics": copy.deepcopy(record.metrics),
            "artifact_ref": record.artifact_ref,
            "error": copy.deepcopy(record.error),
        }
