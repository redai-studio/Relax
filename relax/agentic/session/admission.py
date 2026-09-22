# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Pure execution-token accounting for agentic program admission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AdmissionAction(str, Enum):
    ADMIT = "admit"
    BYPASS = "bypass"


class AdmissionReason(str, Enum):
    CAPACITY_AVAILABLE = "capacity_available"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    PRESSURE_GUARD = "pressure_guard"
    PROTECTED = "protected"
    AGED = "aged"
    DEGRADED = "degraded"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class WorkerSnapshot:
    engine_id: str
    max_total_num_tokens: int
    token_usage: float
    num_used_tokens: int = 0
    healthy: bool = True


@dataclass(frozen=True)
class Lease:
    lease_id: str
    ticket_id: str
    tokens: int
    owner_epoch: int
    created_at: float


@dataclass(frozen=True)
class GrantResult:
    granted: bool
    reason: AdmissionReason
    lease_id: str | None
    owner_epoch: int
    reservation_tokens: int


def compute_reservation_tokens(*, prompt_tokens: int, remaining_completion_tokens: int) -> int:
    """Reserve the expanded training prefix plus the remaining decode
    budget."""

    return max(0, int(prompt_tokens)) + max(0, int(remaining_completion_tokens))


class BudgetState:
    """Deterministic single-writer ledger used by the admission coordinator."""

    def __init__(
        self,
        *,
        headroom: float = 0.90,
        pressure_threshold: float = 0.92,
        lease_ttl_s: float = 600.0,
        staleness_s: float = 6.0,
    ) -> None:
        self.headroom = headroom
        self.pressure_threshold = pressure_threshold
        self.lease_ttl_s = lease_ttl_s
        self.staleness_s = staleness_s
        self.epoch = 0
        self.ceiling = 0
        self.reserved = 0
        self.avg_usage = 0.0
        self.max_usage = 0.0
        self.peak_usage = 0.0
        self._usage_sum = 0.0
        self._usage_count = 0
        self._leases: dict[str, Lease] = {}
        self._lease_by_ticket: dict[str, str] = {}
        self._worker_signature: tuple[str, ...] = ()
        self._last_snapshot_at: float | None = None

    def reconcile(self, snapshots: list[WorkerSnapshot], *, now: float) -> None:
        healthy = [snapshot for snapshot in snapshots if snapshot.healthy and snapshot.max_total_num_tokens > 0]
        if not healthy:
            self.invalidate()
            return
        signature = tuple(sorted(snapshot.engine_id for snapshot in healthy))
        if signature != self._worker_signature:
            self.epoch += 1
            self._worker_signature = signature
        usages = [min(1.0, max(0.0, float(snapshot.token_usage))) for snapshot in healthy]
        self.avg_usage = sum(usages) / len(usages)
        self.max_usage = max(usages)
        self.peak_usage = max(self.peak_usage, self.max_usage)
        self._usage_sum += self.avg_usage
        self._usage_count += 1
        self.ceiling = int(sum(snapshot.max_total_num_tokens for snapshot in healthy) * self.headroom)
        self._last_snapshot_at = now

    def invalidate(self) -> None:
        self.ceiling = 0
        self._last_snapshot_at = None

    def is_stale(self, now: float) -> bool:
        return self.ceiling <= 0 or self._last_snapshot_at is None or now - self._last_snapshot_at > self.staleness_s

    def reserve(self, *, ticket_id: str, tokens: int, now: float) -> GrantResult:
        tokens = max(0, int(tokens))
        existing_lease_id = self._lease_by_ticket.get(ticket_id)
        if existing_lease_id is not None:
            lease = self._leases[existing_lease_id]
            return GrantResult(
                True, AdmissionReason.CAPACITY_AVAILABLE, lease.lease_id, lease.owner_epoch, lease.tokens
            )
        if self.is_stale(now):
            return GrantResult(False, AdmissionReason.DEGRADED, None, self.epoch, tokens)
        if self.max_usage >= self.pressure_threshold:
            return GrantResult(False, AdmissionReason.PRESSURE_GUARD, None, self.epoch, tokens)
        if self.reserved + tokens > self.ceiling:
            return GrantResult(False, AdmissionReason.CAPACITY_EXHAUSTED, None, self.epoch, tokens)
        lease_id = f"{self.epoch}:{ticket_id}"
        lease = Lease(lease_id, ticket_id, tokens, self.epoch, now)
        self._leases[lease_id] = lease
        self._lease_by_ticket[ticket_id] = lease_id
        self.reserved += tokens
        return GrantResult(True, AdmissionReason.CAPACITY_AVAILABLE, lease_id, self.epoch, tokens)

    def release(self, lease_id: str) -> None:
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            return
        self._lease_by_ticket.pop(lease.ticket_id, None)
        self.reserved = max(0, self.reserved - lease.tokens)

    def release_ticket(self, ticket_id: str) -> None:
        lease_id = self._lease_by_ticket.get(ticket_id)
        if lease_id is not None:
            self.release(lease_id)

    def lease_for_ticket(self, ticket_id: str) -> Lease | None:
        lease_id = self._lease_by_ticket.get(ticket_id)
        return self._leases.get(lease_id) if lease_id is not None else None

    def expire_ttl(self, now: float) -> tuple[str, ...]:
        expired = tuple(
            lease_id for lease_id, lease in self._leases.items() if now - lease.created_at > self.lease_ttl_s
        )
        for lease_id in expired:
            self.release(lease_id)
        return expired

    def snapshot(self, *, now: float, reset_usage_window: bool = False) -> dict[str, float | int | bool]:
        window_mean = self._usage_sum / self._usage_count if self._usage_count else self.avg_usage
        snapshot: dict[str, float | int | bool] = {
            "epoch": self.epoch,
            "capacity": self.ceiling,
            "reserved_tokens": self.reserved,
            "available_tokens": max(0, self.ceiling - self.reserved),
            "lease_count": len(self._leases),
            "kv_usage_mean": window_mean,
            "kv_usage_max": self.peak_usage,
            "degraded": self.is_stale(now),
        }
        if reset_usage_window:
            self.peak_usage = self.max_usage
            self._usage_sum = 0.0
            self._usage_count = 0
        return snapshot
