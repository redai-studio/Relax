# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Runtime-owned global admission queue for agentic backend attempts."""

from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx
import ray

from relax.agentic.session.admission import AdmissionAction, AdmissionReason, BudgetState, GrantResult, WorkerSnapshot
from relax.utils.http_utils import router_worker_base_urls


_RECONCILE_INTERVAL_S = 2.0
_LEASE_TTL_S = 600.0
_METRICS_TIMEOUT_S = 5.0
_MAX_CONCURRENCY = 131072
_KV_GAUGE_NAMES = (
    "sglang:max_total_num_tokens",
    "sglang:num_used_tokens",
    "sglang:token_usage",
    "sglang:full_token_usage",
)
_PROM_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(.+)$")


@dataclass
class _Waiter:
    ticket_id: str
    tokens: int
    enqueued_at: float
    result: asyncio.Future[dict[str, Any]]


def _parse_engine_kv_gauges(text: str) -> dict[str, float]:
    gauges: dict[str, float] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE_RE.match(line)
        if match is None or match.group(1) not in _KV_GAUGE_NAMES:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        name = match.group(1)
        gauges[name] = max(gauges.get(name, value), value)
    return gauges


def _decision(
    action: AdmissionAction,
    reason: AdmissionReason,
    *,
    ticket_id: str,
    tokens: int,
    lease_id: str | None = None,
    owner_epoch: int = -1,
    wait_s: float = 0.0,
) -> dict[str, Any]:
    return {
        "action": action.value,
        "reason": reason.value,
        "ticket_id": ticket_id,
        "reservation_tokens": tokens,
        "lease_id": lease_id,
        "owner_epoch": owner_epoch,
        "wait_s": wait_s,
    }


@ray.remote(max_concurrency=_MAX_CONCURRENCY)
class AdmissionCoordinator:
    """Single-writer token ledger and FIFO waiter queue shared by all
    Shards."""

    def __init__(self, args: Any) -> None:
        self._router_ip = args.sglang_router_ip
        self._router_port = args.sglang_router_port
        self._max_wait_s = float(args.agentic_admission_max_wait_s)
        self._state = BudgetState(
            headroom=float(args.agentic_admission_headroom),
            pressure_threshold=float(args.agentic_admission_pressure_threshold),
            lease_ttl_s=_LEASE_TTL_S,
            staleness_s=max(3 * _RECONCILE_INTERVAL_S, 5.0),
        )
        self._waiters: dict[str, _Waiter] = {}
        self._waiter_order: deque[str] = deque()
        self._cancelled_tickets: dict[str, float] = {}
        self._counters: dict[str, float] = {}
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(_METRICS_TIMEOUT_S))
        self._poll_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._reconcile_once()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="agentic-admission-reconcile")

    async def acquire(self, request: dict[str, Any]) -> dict[str, Any]:
        ticket_id = str(request["ticket_id"])
        tokens = max(0, int(request["reservation_tokens"]))
        if self._cancelled_tickets.pop(ticket_id, None) is not None:
            return _decision(AdmissionAction.BYPASS, AdmissionReason.CANCELLED, ticket_id=ticket_id, tokens=tokens)
        if bool(request.get("protected")):
            self._increment("bypass")
            self._increment("bypass_protected")
            return _decision(AdmissionAction.BYPASS, AdmissionReason.PROTECTED, ticket_id=ticket_id, tokens=tokens)

        lease = self._state.lease_for_ticket(ticket_id)
        if lease is not None:
            return _decision(
                AdmissionAction.ADMIT,
                AdmissionReason.CAPACITY_AVAILABLE,
                ticket_id=ticket_id,
                tokens=lease.tokens,
                lease_id=lease.lease_id,
                owner_epoch=lease.owner_epoch,
            )
        waiter = self._waiters.get(ticket_id)
        if waiter is not None and waiter.tokens != tokens:
            raise RuntimeError(f"admission ticket {ticket_id!r} was retried with a different reservation")
        if waiter is None:
            if not self._waiters:
                immediate = self._state.reserve(ticket_id=ticket_id, tokens=tokens, now=time.monotonic())
                if immediate.granted or immediate.reason is AdmissionReason.DEGRADED:
                    return self._finish_grant(ticket_id, immediate)
            waiter = _Waiter(ticket_id, tokens, time.monotonic(), asyncio.get_running_loop().create_future())
            self._waiters[ticket_id] = waiter
            self._waiter_order.append(ticket_id)
            self._increment("wait")
            self._drain_waiters()
        return await waiter.result

    async def cancel(self, ticket_id: str) -> None:
        waiter = self._waiters.pop(ticket_id, None)
        lease = self._state.lease_for_ticket(ticket_id)
        if waiter is not None:
            self._increment("cancelled")
            if not waiter.result.done():
                waiter.result.set_result(
                    _decision(
                        AdmissionAction.BYPASS,
                        AdmissionReason.CANCELLED,
                        ticket_id=ticket_id,
                        tokens=waiter.tokens,
                        wait_s=time.monotonic() - waiter.enqueued_at,
                    )
                )
        elif lease is None:
            self._cancelled_tickets[ticket_id] = time.monotonic()
        self._state.release_ticket(ticket_id)
        self._drain_waiters()

    async def release(self, lease_id: str) -> None:
        self._state.release(lease_id)
        self._drain_waiters()

    async def metrics(self, reset: bool = False) -> dict[str, float]:
        snapshot = self._state.snapshot(now=time.monotonic(), reset_usage_window=reset)
        counters = self._counters.copy()
        wait_granted = counters.pop("wait_granted", 0)
        wait_seconds_sum = counters.pop("wait_seconds_sum", 0.0)
        lease_expired = counters.pop("lease_expired", 0)
        metrics = {f"admission/{key}": value for key, value in counters.items()}
        # Preserve the legacy decision-event rates after DEFER became a FIFO
        # wait. These are rates over admission decisions, not unique tickets:
        # one ticket may contribute a wait decision and a later admit/bypass.
        admit = counters.get("admit", 0.0)
        defer = counters.get("wait", 0.0)
        bypass = counters.get("bypass", 0.0)
        decision_total = admit + defer + bypass
        if decision_total > 0:
            metrics["admission/defer_rate"] = defer / decision_total
            metrics["admission/degraded_rate"] = counters.get("bypass_degraded", 0.0) / decision_total
        metrics["admission/waiting"] = float(len(self._waiters))
        if wait_granted:
            metrics["admission/wait_seconds_mean"] = wait_seconds_sum / wait_granted
        ceiling = float(snapshot["capacity"])
        reserved = float(snapshot["reserved_tokens"])
        metrics.update(
            {
                "budget/ceiling": ceiling,
                "budget/reserved": reserved,
                "budget/available_tokens": float(snapshot["available_tokens"]),
                "budget/lease_count": float(snapshot["lease_count"]),
                "budget/kv_token_usage_mean": float(snapshot["kv_usage_mean"]),
                "budget/kv_token_usage_max": float(snapshot["kv_usage_max"]),
                "budget/epoch": float(snapshot["epoch"]),
                "budget/degraded": float(snapshot["degraded"]),
            }
        )
        if ceiling > 0:
            metrics["budget/reserved_utilization"] = reserved / ceiling
        if lease_expired:
            metrics["budget/lease_expired"] = lease_expired
        if reset:
            self._counters.clear()
        return metrics

    async def shutdown(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            await asyncio.gather(self._poll_task, return_exceptions=True)
            self._poll_task = None
        for waiter in self._waiters.values():
            if not waiter.result.done():
                waiter.result.set_result(
                    _decision(
                        AdmissionAction.BYPASS,
                        AdmissionReason.DEGRADED,
                        ticket_id=waiter.ticket_id,
                        tokens=waiter.tokens,
                    )
                )
        self._waiters.clear()
        self._waiter_order.clear()
        self._cancelled_tickets.clear()
        await self._client.aclose()

    def _finish_grant(self, ticket_id: str, grant: GrantResult, *, wait_s: float = 0.0) -> dict[str, Any]:
        if grant.granted:
            assert grant.lease_id is not None
            self._increment("admit")
            if wait_s:
                self._increment("wait_granted")
                self._increment("wait_seconds_sum", wait_s)
            return _decision(
                AdmissionAction.ADMIT,
                grant.reason,
                ticket_id=ticket_id,
                tokens=grant.reservation_tokens,
                lease_id=grant.lease_id,
                owner_epoch=grant.owner_epoch,
                wait_s=wait_s,
            )
        assert grant.reason is AdmissionReason.DEGRADED
        self._increment("bypass")
        self._increment("bypass_degraded")
        return _decision(
            AdmissionAction.BYPASS,
            grant.reason,
            ticket_id=ticket_id,
            tokens=grant.reservation_tokens,
            wait_s=wait_s,
        )

    def _drain_waiters(self) -> None:
        now = time.monotonic()
        while self._waiter_order:
            ticket_id = self._waiter_order[0]
            waiter = self._waiters.get(ticket_id)
            if waiter is None:
                self._waiter_order.popleft()
                continue
            waited = now - waiter.enqueued_at
            if waited >= self._max_wait_s:
                self._waiter_order.popleft()
                self._waiters.pop(ticket_id)
                self._increment("bypass")
                self._increment("bypass_aged")
                waiter.result.set_result(
                    _decision(
                        AdmissionAction.BYPASS,
                        AdmissionReason.AGED,
                        ticket_id=ticket_id,
                        tokens=waiter.tokens,
                        wait_s=waited,
                    )
                )
                continue
            grant = self._state.reserve(ticket_id=ticket_id, tokens=waiter.tokens, now=now)
            if not grant.granted and grant.reason is not AdmissionReason.DEGRADED:
                break
            self._waiter_order.popleft()
            self._waiters.pop(ticket_id)
            waiter.result.set_result(self._finish_grant(ticket_id, grant, wait_s=waited))

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(_RECONCILE_INTERVAL_S)
            await self._reconcile_once()

    async def _reconcile_once(self) -> None:
        try:
            snapshots = await self._fetch_worker_snapshots()
        except Exception:
            snapshots = []
        now = time.monotonic()
        self._state.reconcile(snapshots, now=now)
        expired = set(self._state.expire_ttl(now))
        if expired:
            self._increment("lease_expired", len(expired))
        self._cancelled_tickets = {
            ticket_id: cancelled_at
            for ticket_id, cancelled_at in self._cancelled_tickets.items()
            if now - cancelled_at <= _LEASE_TTL_S
        }
        self._drain_waiters()

    async def _fetch_worker_snapshots(self) -> list[WorkerSnapshot]:
        base_url = f"http://{self._router_ip}:{self._router_port}"
        raw_urls: list[str]
        try:
            response = await self._client.get(f"{base_url}/workers")
            response.raise_for_status()
            raw_urls = [
                worker["url"] for worker in response.json()["workers"] if worker["url"] and worker["is_healthy"]
            ]
        except Exception:
            response = await self._client.get(f"{base_url}/list_workers")
            response.raise_for_status()
            raw_urls = list(response.json()["urls"])
        urls = router_worker_base_urls(raw_urls)
        responses = await asyncio.gather(*(self._client.get(f"{url}/metrics") for url in urls))
        snapshots = []
        for url, response in zip(urls, responses, strict=True):
            response.raise_for_status()
            gauges = _parse_engine_kv_gauges(response.text)
            capacity = int(gauges.get("sglang:max_total_num_tokens", 0))
            used = int(gauges.get("sglang:num_used_tokens", 0))
            usage = gauges.get("sglang:token_usage", gauges.get("sglang:full_token_usage", 0.0))
            if usage == 0.0 and capacity:
                usage = used / capacity
            snapshots.append(WorkerSnapshot(url, capacity, usage, used))
        return snapshots

    def _increment(self, key: str, value: float = 1.0) -> None:
        self._counters[key] = self._counters.get(key, 0.0) + value


class RayAdmissionClient:
    """Cancellation-safe client for one runtime-owned Coordinator."""

    def __init__(self, handle: Any) -> None:
        self.handle = handle

    async def acquire(self, request: dict[str, Any]) -> dict[str, Any]:
        ticket_id = str(request["ticket_id"])
        acquire_ref = self.handle.acquire.remote(request)
        try:
            return await asyncio.shield(acquire_ref)
        except asyncio.CancelledError:
            cleanup = asyncio.gather(self.handle.cancel.remote(ticket_id), acquire_ref, return_exceptions=True)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            raise
        except Exception:
            await asyncio.gather(self.handle.cancel.remote(ticket_id), return_exceptions=True)
            return _decision(
                AdmissionAction.BYPASS,
                AdmissionReason.DEGRADED,
                ticket_id=ticket_id,
                tokens=int(request["reservation_tokens"]),
            )

    async def release(self, lease_id: str) -> None:
        try:
            await asyncio.shield(self.handle.release.remote(lease_id))
        except Exception:
            return
