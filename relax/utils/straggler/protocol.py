# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Versioned wire protocol and transport-robustness helpers.

The profiler ships envelopes between ranks over a same-host TCP socket, so the
collector sees a stream that can contain three kinds of noise it must not let
into the detector:

* malformed packets (a truncated line, a field of the wrong type);
* re-delivered packets (a reconnect resends what the sender already queued);
* out-of-order packets (a slow connection delivers an older sample after a
  newer one for the same rank).

Everything here is deliberately allocation-light, bounded and fail-open. None
of these helpers may raise into the training path: a protocol error is
*counted* and the packet is dropped, never propagated.

The schema version is explicit because the envelope is produced by
``observer.py`` and consumed by ``collector.py``, which are allowed to evolve
independently. A packet that does not declare the version it was written with
cannot be interpreted safely, so a decoded wire dict without
``schema_version`` is rejected. A locally constructed envelope object is
trusted instead: reading it through :func:`getattr` keeps this module working
both before and after the observer grows a ``schema_version`` (or any other)
attribute.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple


#: Version of the packet layout produced by this module. Bumped from the
#: implicit, unversioned JSON the observer used to emit.
SCHEMA_VERSION = 2

#: Stable identifier a reader can log to tell which protocol wrote a packet.
PROTOCOL_NAME = "relax.straggler.timing"

#: Values accepted for the optional ``measurement_kind`` field. ``device``
#: means a CUDA event pair backed the interval, ``host_only`` means only the
#: host timestamps exist, ``unknown`` means the sender could not say.
MEASUREMENT_KINDS: Tuple[str, ...] = ("device", "host_only", "unknown")

#: Optional per-interval workload counters. They are not required: when a
#: deployment cannot report them the detector keeps comparing raw durations
#: with ``work_tolerance``. When present, equal-work windows can be compared
#: against each other instead of trusting that every rank did the same amount
#: of work.
WORKLOAD_FIELDS: Tuple[str, ...] = ("tokens", "sequences", "microbatches")

#: Hard cap on retained dedup keys. At ~1 envelope per stage per step, 8192
#: covers thousands of windows while keeping the tracker a few hundred
#: kilobytes; the cap matters because a silent or hostile peer must not be
#: able to grow the collector's memory.
DEDUP_MAX_ENTRIES = 8192

#: A key is only remembered for two minutes. A resend after that window is a
#: fresh sample, not a duplicate: a rank that was disconnected long enough for
#: its reconnect loop to retry an old packet has already moved on.
DEDUP_TTL_S = 120.0

#: Hard cap on the length of every free-text field that reaches the collector's
#: bounded structures. ``DEDUP_MAX_ENTRIES`` bounds the *number* of retained
#: keys, but each key embeds ``run_id``/``topology_epoch`` and each sample
#: embeds ``name``; without a per-field cap a single hostile line can be just
#: under ``MAX_LINE_BYTES`` and multiply the 8192-entry cap into gigabytes.
#: Real run ids, cohort keys and Megatron timer names are far below this.
MAX_TEXT_CHARS = 4096

#: Free-text fields checked against :data:`MAX_TEXT_CHARS`.
_TEXT_FIELDS: Tuple[str, ...] = ("run_id", "topology_epoch", "name", "cohort", "label")

#: Rank and sequence fallbacks, matching ``TimingEnvelope.from_dict`` so a
#: malformed packet keys the same way however it entered the process.
_DEFAULT_RANK = -1
_DEFAULT_SEQ = 0


def _read(payload: Any, name: str, default: Any) -> Any:
    """Read ``name`` from a mapping or an object, never raising.

    Args:
        payload: A decoded dict or an envelope-like object.
        name: Field name.
        default: Value returned when the field is absent or unreadable.

    Returns:
        The field value, or ``default``.
    """
    try:
        if isinstance(payload, dict):
            value = payload.get(name, default)
            return default if value is None else value
        value = getattr(payload, name, default)
        return default if value is None else value
    except Exception:
        return default


def _as_int(value: Any, default: int) -> int:
    """Coerce ``value`` to ``int``, returning ``default`` when impossible."""
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def dedup_key(payload: Any) -> Tuple[Any, Any, int, int]:
    """Return the idempotency key of one payload.

    The key is ``(run_id, topology_epoch, global_rank, sample_seq)``. Fields are
    read defensively so the same code works for a decoded wire dict and for a
    locally built envelope, and so it keeps working before
    ``topology_epoch``/``global_rank``/``sample_seq`` exist on the observer's
    dataclass:

    * ``topology_epoch`` defaults to ``""`` (a topology change must invalidate
      comparisons, and an empty epoch matches the identity default);
    * ``global_rank`` falls back to ``rank``;
    * ``sample_seq`` falls back to ``seq``.

    Args:
        payload: A mapping or an envelope-like object.

    Returns:
        A hashable four-tuple; never raises.
    """
    run_id = _read(payload, "run_id", "")
    topology_epoch = _read(payload, "topology_epoch", "")
    global_rank = _read(payload, "global_rank", None)
    if global_rank is None:
        global_rank = _read(payload, "rank", _DEFAULT_RANK)
    sample_seq = _read(payload, "sample_seq", None)
    if sample_seq is None:
        sample_seq = _read(payload, "seq", _DEFAULT_SEQ)
    return (run_id, topology_epoch, _as_int(global_rank, _DEFAULT_RANK), _as_int(sample_seq, _DEFAULT_SEQ))


def _validate(payload: Any) -> Tuple[bool, str]:
    """Implementation of :func:`validate`; may raise."""
    if isinstance(payload, dict):
        # A wire dict must declare its layout: without a version the reader
        # cannot know which fields to trust.
        if payload.get("schema_version") is None:
            return False, "missing_schema_version"
    elif getattr(payload, "schema_version", SCHEMA_VERSION) is None:
        # A locally constructed envelope predating the versioned wire format is
        # trusted and assumed to speak the current schema; dropping every local
        # observation would disable the profiler entirely.
        return False, "missing_schema_version"

    rank = _read(payload, "rank", None)
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        return False, "invalid_rank"

    name = _read(payload, "name", None)
    if not isinstance(name, str) or not name:
        return False, "invalid_name"

    kind = _read(payload, "measurement_kind", "")
    if kind not in ("", None) and kind not in MEASUREMENT_KINDS:
        return False, "invalid_measurement_kind"

    for field in _TEXT_FIELDS:
        value = _read(payload, field, None)
        if isinstance(value, str) and len(value) > MAX_TEXT_CHARS:
            return False, "text_field_too_long"
    return True, ""


def validate(payload: Any) -> Tuple[bool, str]:
    """Check one payload against the wire schema.

    The checks are the ones the detector depends on: a schema version is
    declared, ``rank`` is a non-negative ``int``, ``name`` is a non-empty
    string, and ``measurement_kind`` (when populated) is a known value. The
    optional workload counters are intentionally *not* validated: they are
    advisory and a bad counter must not discard a real timing sample.

    Args:
        payload: A mapping or an envelope-like object.

    Returns:
        ``(ok, reason)``. ``reason`` is an empty string when ``ok`` is
        ``True`` and a short machine-readable code otherwise. Never raises.
    """
    try:
        return _validate(payload)
    except Exception:
        return False, "validate_error"


class BoundedDedup:
    """Hard-capped, TTL-bounded idempotency tracker for arriving packets.

    Two independent questions are answered per packet:

    * has this exact key already been accepted recently (a reconnect resend)?
    * is this key older than the newest sequence already seen for the same
      ``(run_id, topology_epoch, global_rank)`` (out-of-order arrival)?

    Both structures are bounded, so a peer that floods the collector with
    distinct keys evicts old entries (counted) instead of growing memory.

    The tracker is **not** safe to call from several threads on its own: the
    expiry/eviction walks are read-modify-write over ``OrderedDict``s, and an
    unguarded concurrent call raises ``RuntimeError: OrderedDict mutated during
    iteration`` (which :meth:`check` used to swallow, returning ``"new"``
    without recording the key and therefore losing idempotency). ``check`` and
    ``stats`` are serialised by an internal lock so the class is self-contained
    for direct callers.
    """

    def __init__(
        self,
        max_entries: int = DEDUP_MAX_ENTRIES,
        ttl_s: float = DEDUP_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._ttl_s = max(0.0, float(ttl_s))
        self._clock = clock
        self._seen: "OrderedDict[Tuple[Any, Any, Any, int], float]" = OrderedDict()
        self._newest: "OrderedDict[Tuple[Any, Any, Any], int]" = OrderedDict()
        self._counters: Dict[str, int] = {"new": 0, "duplicate": 0, "late": 0, "expired": 0, "evicted": 0}
        self._lock = threading.Lock()

    def check(self, key: Tuple[Any, Any, Any, int]) -> str:
        """Classify one key as ``"new"``, ``"duplicate"`` or ``"late"``.

        A key already accepted within the TTL is a duplicate. Otherwise, if its
        sequence is behind the newest sequence seen for the same rank, it is
        late (accepted nowhere near the detector). Anything else is new and
        becomes the newest sequence for its rank.

        Args:
            key: A ``(run_id, topology_epoch, global_rank, sample_seq)`` tuple,
                normally from :func:`dedup_key`.

        Returns:
            One of ``"new"``, ``"duplicate"`` or ``"late"``. Never raises; an
            internal failure degrades to ``"new"`` so evidence is kept rather
            than silently suppressed.
        """
        with self._lock:
            try:
                return self._check(key)
            except Exception:
                return "new"

    def _check(self, key: Any) -> str:
        """Implementation of :meth:`check`; may raise."""
        normalised = self._normalise(key)
        now = float(self._clock())
        self._expire(now)

        seen_at = self._seen.get(normalised)
        if seen_at is not None and now - seen_at <= self._ttl_s:
            self._counters["duplicate"] += 1
            return "duplicate"

        rank_key = normalised[:3]
        seq = normalised[3]
        newest = self._newest.get(rank_key)
        if newest is not None and seq < newest:
            self._remember(normalised, now)
            self._counters["late"] += 1
            return "late"

        self._remember(normalised, now)
        self._note_newest(rank_key, seq)
        self._counters["new"] += 1
        return "new"

    @staticmethod
    def _normalise(key: Any) -> Tuple[Any, Any, Any, int]:
        """Coerce an arbitrary key into the tracker's four-tuple shape."""
        if isinstance(key, (tuple, list)) and len(key) == 4:
            return (key[0], key[1], key[2], _as_int(key[3], _DEFAULT_SEQ))
        return (key, "", _DEFAULT_RANK, _DEFAULT_SEQ)

    def _expire(self, now: float) -> None:
        """Drop every entry older than the TTL, counting the expiries."""
        while self._seen:
            key, seen_at = next(iter(self._seen.items()))
            if now - seen_at <= self._ttl_s:
                break
            self._seen.popitem(last=False)
            self._counters["expired"] += 1

    def _remember(self, key: Tuple[Any, Any, Any, int], now: float) -> None:
        """Record a key, evicting the oldest entry when the cap is reached."""
        if key in self._seen:
            self._seen[key] = now
            self._seen.move_to_end(key)
            return
        while len(self._seen) >= self._max_entries:
            self._seen.popitem(last=False)
            self._counters["evicted"] += 1
        self._seen[key] = now

    def _note_newest(self, rank_key: Tuple[Any, Any, Any], seq: int) -> None:
        """Advance the newest sequence of one rank, bounded like the keys."""
        if rank_key in self._newest:
            if seq > self._newest[rank_key]:
                self._newest[rank_key] = seq
            self._newest.move_to_end(rank_key)
            return
        while len(self._newest) >= self._max_entries:
            self._newest.popitem(last=False)
            self._counters["evicted"] += 1
        self._newest[rank_key] = seq

    def stats(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot of occupancy and decisions."""
        with self._lock:
            return {
                "max_entries": self._max_entries,
                "ttl_s": self._ttl_s,
                "entries": len(self._seen),
                "newest_ranks": len(self._newest),
                **self._counters,
            }


__all__ = [
    "DEDUP_MAX_ENTRIES",
    "DEDUP_TTL_S",
    "MAX_TEXT_CHARS",
    "MEASUREMENT_KINDS",
    "PROTOCOL_NAME",
    "SCHEMA_VERSION",
    "WORKLOAD_FIELDS",
    "BoundedDedup",
    "dedup_key",
    "validate",
]
