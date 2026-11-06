# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Host-local Session credits, independent of logical Group residency."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(eq=False)
class SessionReservation:
    group_id: str
    node_id: str
    count: int
    session_ids: frozenset[str] = frozenset()
    confirmed: set[str] = field(default_factory=set)
    submitted: bool = False
    allocated_count: int = field(init=False)
    allocated_ids: set[str] = field(default_factory=set)
    session_order: tuple[str, ...] = ()
    closing: bool = False

    def __post_init__(self) -> None:
        self.allocated_count = self.count

    @property
    def outstanding(self) -> int:
        return self.allocated_count - len(self.confirmed & self.allocated_ids)

    @property
    def pending_ids(self) -> tuple[str, ...]:
        return tuple(sid for sid in self.session_order if sid not in self.allocated_ids and sid not in self.confirmed)


class SessionCapacity:
    """One Runtime's atomic reservations; ambiguous owners keep their
    credits."""

    def __init__(self, node_ids: set[str], per_node: int, *, incremental: bool = False) -> None:
        if type(per_node) is not int or per_node < 1 or not node_ids or not all(node_ids):
            raise ValueError("Session capacity needs distinct node identities and a positive per-node limit")
        self.per_node = per_node
        self.used = dict.fromkeys(node_ids, 0)
        self.peak = dict.fromkeys(node_ids, 0)
        self.global_peak = 0
        self.confirmed_count = 0
        self.reservations: dict[str, SessionReservation] = {}
        if type(incremental) is not bool:
            raise ValueError("Incremental Session startup must be explicitly boolean")
        self.incremental = incremental
        self.partial_by_node: dict[str, SessionReservation] = {}

    def available(self, node_id: str) -> int:
        return self.per_node - self.used[node_id]

    def can_reserve(self, node_id: str, count: int) -> bool:
        if self.incremental:
            return node_id not in self.partial_by_node and self.available(node_id) > 0
        return self.available(node_id) >= count

    def _debit(self, node_id: str, count: int) -> None:
        self.used[node_id] += count
        self.peak[node_id] = max(self.peak[node_id], self.used[node_id])
        self.global_peak = max(self.global_peak, sum(self.used.values()))

    def reserve(self, group_id: str, node_id: str, count: int) -> SessionReservation:
        if (
            group_id in self.reservations
            or type(count) is not int
            or not 0 < count <= self.per_node
            or not self.can_reserve(node_id, count)
        ):
            raise ValueError("Group reservation must be new and fit atomically on one host")
        reservation = SessionReservation(group_id, node_id, count)
        reservation.allocated_count = min(count, self.available(node_id))
        self.reservations[group_id] = reservation
        self._debit(node_id, reservation.allocated_count)
        if reservation.allocated_count < count:
            self.partial_by_node[node_id] = reservation
        return reservation

    def bind(self, reservation: SessionReservation, session_ids: tuple[str, ...]) -> None:
        self._owned(reservation)
        if reservation.session_ids or reservation.submitted or len(session_ids) != reservation.count:
            raise ValueError("Session identities must bind exactly once before submission")
        if len(set(session_ids)) != len(session_ids) or not all(session_ids):
            raise ValueError("Session identities must be nonempty and distinct")
        reservation.session_ids = frozenset(session_ids)
        reservation.session_order = session_ids
        reservation.allocated_ids.update(session_ids[: reservation.allocated_count])

    def grant(self, reservation: SessionReservation) -> tuple[str, ...]:
        """Debit exact identities before asking their owner to start
        processes."""
        self._owned(reservation)
        if not self.incremental or not reservation.submitted or reservation.closing:
            raise ValueError("Only a submitted open incremental Group can receive credits")
        if self.partial_by_node.get(reservation.node_id) is not reservation:
            raise ValueError("Only the host's partial Group can receive additional credits")
        ids = reservation.pending_ids[: self.available(reservation.node_id)]
        reservation.allocated_ids.update(ids)
        reservation.allocated_count += len(ids)
        self._debit(reservation.node_id, len(ids))
        if not reservation.pending_ids:
            self.partial_by_node.pop(reservation.node_id)
        return ids

    def close(self, reservation: SessionReservation) -> None:
        self._owned(reservation)
        reservation.closing = True

    def confirm(self, reservation: SessionReservation, session_ids: tuple[str, ...]) -> None:
        self._owned(reservation)
        if not set(session_ids) <= reservation.session_ids:
            raise ValueError("Resource acknowledgement names a Session outside its reservation")
        fresh = (set(session_ids) - reservation.confirmed) & reservation.allocated_ids
        # Never-started members can acknowledge drop, but own no debit to refund.
        reservation.confirmed.update(session_ids)
        self.used[reservation.node_id] -= len(fresh)
        self.confirmed_count += len(fresh)
        if reservation.session_ids and not reservation.pending_ids:
            if self.partial_by_node.get(reservation.node_id) is reservation:
                self.partial_by_node.pop(reservation.node_id)

    def retire(self, reservation: SessionReservation) -> None:
        self._owned(reservation)
        if not reservation.submitted:
            self.used[reservation.node_id] -= reservation.outstanding
        elif reservation.outstanding:
            # Retain the exact owner and debit: deleting a logical Group is not
            # evidence that its external processes have stopped.
            raise RuntimeError(f"Unconfirmed task resources for {reservation.group_id}: {reservation.outstanding}")
        del self.reservations[reservation.group_id]
        if self.partial_by_node.get(reservation.node_id) is reservation:
            self.partial_by_node.pop(reservation.node_id)

    def _owned(self, reservation: SessionReservation) -> None:
        if self.reservations.get(reservation.group_id) is not reservation:
            raise ValueError("Stale Session reservation identity")

    def snapshot(self) -> dict:
        return {
            "per_node_limit": self.per_node,
            "global_limit": self.per_node * len(self.used),
            "incremental_session_start": self.incremental,
            "used_by_node": dict(self.used),
            "peak_by_node": dict(self.peak),
            "global_peak": self.global_peak,
            "confirmed_releases": self.confirmed_count,
            "outstanding": sum(self.used.values()),
        }
