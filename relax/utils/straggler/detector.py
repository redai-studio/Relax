# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Pure (CPU-only, torch-free) analysis of one gathered straggler window.

Input is the ``[world, NUM_FIELDS]`` table produced by every rank's
``WindowStats`` plus static per-rank metadata. Output is a flat metrics dict,
a list of alerts and a per-rank row table. Keeping this free of torch and of
process-group state makes it unit-testable with synthetic tables.

Peer groups are pipeline stages: ranks on different PP stages legitimately
have different compute times, so a rank is only ever compared with ranks that
run the same layers. Within a group every rank is compared with the
leave-one-out median of its peers, which stays meaningful for groups as small
as two ranks.

Two independent signals are checked:

* **self compute** (``fwd + bwd + optim`` compute-stream time) and forward
  alone: a rank whose step or forward takes longer than its stage peers ->
  ``slow_device`` / ``data_imbalance`` / ``cpu_bound``. With
  ``--overlap-grad-reduce`` the reduce-scatter runs during backward, so every
  peer's backward also waits for a slow rank and the self totals even out;
  forward has no DP collective.
* **late arrival** at the DP grad-sync collective: every peer's
  ``bwd + dp_grad_sync`` ends when the collective completes (the peers wait
  in ``dp_grad_sync``, or in ``bwd`` with an overlapped grad reduce), so a
  rank with a much *shorter* sum than its peers is the one everybody waited
  for. Its GPU kernels may be perfectly normal; the lost time sits between
  kernels on the host (data fetch, GIL / GC, blocking I/O) -> ``late_arrival``.

A rank that is neither but waits on its PP neighbours much longer than the
unflagged, on-time ranks of its stage is ``upstream_wait``; flagged and late
ranks wait *less* (their neighbours wait for them), so they are left out of
that reference. A rank that is late only because of such a PP wait is
``upstream_wait`` too.

Segment times come from events on the compute stream, so they include host
launch gaps inside the bracket; "GPU time" below is shorthand for that.

A rank whose window cannot support a verdict is reported as **uncertain**
with a cause instead of silently passing: its timing is incomplete (dropped or
still-unread event pairs, caught profiler errors), it has no stage peer to
compare with, it looks slow but its stage has no token counts to rule out
data imbalance, or it waits on PP but every other rank of its stage is
flagged, late or uncertain. An uncertain window never counts as evidence
either way: it breaks candidate streaks and does not count towards clearing
an alert, which needs ``recover_windows`` consecutive clean windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Sequence

import numpy as np

from relax.utils.straggler.stats import (
    FIELD_INDEX,
    GPU_SEGMENTS,
    SELF_SEGMENTS,
    WAIT_SEGMENTS,
)


# Order is part of the metric contract (``straggler/flagged/reason`` logs the code).
REASONS: tuple[str, ...] = ("none", "slow_device", "data_imbalance", "upstream_wait", "cpu_bound", "late_arrival")
REASON_CODE: dict[str, int] = {reason: code for code, reason in enumerate(REASONS)}
# Checked in this order; ``straggler/uncertain/cause`` logs the code.
UNCERTAIN_CAUSES: tuple[str, ...] = (
    "none",
    "dropped_events",
    "unread_events",
    "profiler_errors",
    "no_peers",
    "missing_tokens",
    "no_wait_reference",
)
UNCERTAIN_CODE: dict[str, int] = {cause: code for code, cause in enumerate(UNCERTAIN_CAUSES)}

_EPS = 1e-9


@dataclass(frozen=True)
class RankMeta:
    """Static identity of one training rank."""

    rank: int
    dp: int
    tp: int
    pp: int
    cp: int = 0
    ep: int = 0
    host: str = ""
    device: int = -1

    @property
    def tag(self) -> str:
        return f"rank{self.rank}_dp{self.dp}_tp{self.tp}_pp{self.pp}"


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds for ``analyze_window``; the first four are environment
    variables in production (see ``runtime``)."""

    z_threshold: float = 3.0
    rel_threshold: float = 0.10
    persist_windows: int = 3
    # Consecutive clean (neither candidate nor uncertain) windows before an alert clears.
    recover_windows: int = 2
    # A slow rank whose GC pauses are at least this fraction of its compute is ``cpu_bound``.
    gc_ratio_threshold: float = 0.05
    # Extra PP waiting / DP lateness must be at least this fraction of the stage's
    # median compute to count.
    wait_abs_frac: float = 0.05


@dataclass
class DetectorState:
    """Carry per-rank candidate streaks and active reasons across windows."""

    consecutive: dict[int, int] = field(default_factory=dict)
    active: dict[int, str] = field(default_factory=dict)
    # ``upstream_wait`` streaks are kept apart from ``consecutive`` so windows
    # spent as a victim never count towards flagging the same rank as a culprit.
    waiting: dict[int, int] = field(default_factory=dict)
    # Consecutive clean windows of ranks that currently hold an alert.
    clean: dict[int, int] = field(default_factory=dict)
    # Uncertain cause per rank in the previous window.
    uncertain: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Alert:
    """Verdict for one rank; ``new`` is True when its reason changed since the
    previous window (only those are logged as WARNING)."""

    rank: int
    reason: str
    message: str
    new: bool


@dataclass(frozen=True)
class Uncertain:
    """A rank whose window supports no verdict; ``new`` when the cause
    changed."""

    rank: int
    cause: str
    message: str
    new: bool


@dataclass(frozen=True)
class Recovery:
    """A rank whose alert cleared after ``recover_windows`` clean windows."""

    rank: int
    reason: str
    message: str


@dataclass
class WindowReport:
    """Scalars, alerts and per-rank rows produced by one ``analyze_window``."""

    metrics: dict[str, float]
    alerts: list[Alert]
    rows: list[dict[str, float | int | str]]
    # Wall-clock start of the window; filled in by the collector for the timeline.
    window_start_wall: float = 0.0
    # Free-text status for the log, e.g. why the profiler switched itself off.
    note: str = ""
    # Rollouts covered by the window; the report may be emitted a rollout or so later.
    first_rollout: int = -1
    last_rollout: int = -1
    # Wall-clock times for the report-latency measurement: the window closed
    # (before the gather), the analysis finished (alerts are logged right then),
    # and the training thread handed the scalars to the metrics call.
    closed_wall: float = 0.0
    analyzed_wall: float = 0.0
    emitted_wall: float = 0.0
    uncertain: list[Uncertain] = field(default_factory=list)
    recovered: list[Recovery] = field(default_factory=list)


def _leave_one_out_median(values: np.ndarray) -> np.ndarray:
    """``out[i]`` is the median of ``values`` without element ``i``.

    One sort: removing the element at sorted position ``p`` shifts the peers'
    middle index by one when ``p`` lies at or below it.
    """
    n = values.shape[0]
    if n < 2:
        return values.copy()
    order = np.argsort(values, kind="stable")
    s = values[order]
    p = np.arange(n)
    m = n - 1  # number of peers
    if m % 2 == 1:
        mid = m // 2
        med_sorted = s[np.where(p <= mid, mid + 1, mid)]
    else:
        lo, hi = m // 2 - 1, m // 2
        med_sorted = 0.5 * (s[np.where(p <= lo, lo + 1, lo)] + s[np.where(p <= hi, hi + 1, hi)])
    out = np.empty_like(values)
    out[order] = med_sorted
    return out


def _leave_one_out_mad(values: np.ndarray, med: np.ndarray) -> np.ndarray:
    """``out[i]`` is the median of ``|values[j] - med[i]|`` over ``j != i``.

    ``med`` comes from ``_leave_one_out_median``, which takes at most three
    distinct values (the removed element only shifts the middle index), so this
    is one leave-one-out median of the deviations per distinct centre: O(n log
    n) instead of an ``n x n`` deviation matrix.
    """
    out = np.empty_like(values)
    for centre in np.unique(med):
        rows = med == centre
        out[rows] = _leave_one_out_median(np.abs(values - centre))[rows]
    return out


_REL_CAP = 100.0


def _relative_and_z(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative excess over, robust z-score against, and median of the leave-
    one-out peers.

    z uses the MAD of the peers scaled to a normal sigma (1.4826). When the
    peers are (near) identical the MAD collapses and z explodes; the relative
    threshold in the caller is what keeps that from flagging noise. When the
    peers are (near) zero the relative excess is capped at ``_REL_CAP`` and z
    is computed against 1% of the value itself so a lone non-zero rank still
    registers; callers add an absolute-significance guard where that matters.
    """
    n = values.shape[0]
    if n < 2:
        return np.zeros(n), np.zeros(n), values.copy()
    med = _leave_one_out_median(values)
    rel, z = _excess(values, med, _leave_one_out_mad(values, med))
    return rel, z, med


def _excess(values: np.ndarray, med: np.ndarray, mad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative excess and robust z of ``values`` against per-element peer
    ``med`` / ``mad``, with the fallbacks described in ``_relative_and_z``."""
    rel = np.zeros(values.shape[0])
    positive = med > _EPS
    rel[positive] = np.minimum(values[positive] / med[positive] - 1.0, _REL_CAP)
    rel[~positive & (values > _EPS)] = _REL_CAP

    scale = 1.4826 * mad
    fallback = np.maximum(0.01 * np.maximum(np.abs(med), np.abs(values)), _EPS)
    scale = np.where(scale < _EPS, fallback, scale)
    return rel, (values - med) / scale


def _group_indices(meta: Sequence[RankMeta], key: Callable[[RankMeta], Hashable]) -> list[np.ndarray]:
    groups: dict[Hashable, list[int]] = {}
    for index, rank_meta in enumerate(meta):
        groups.setdefault(key(rank_meta), []).append(index)
    return [np.asarray(members) for members in groups.values()]


@dataclass
class _Window:
    """One gathered window, per step, plus its raw window-level counters."""

    x: np.ndarray
    dropped: np.ndarray
    unread: np.ndarray
    errors: np.ndarray
    health: np.ndarray
    self_ms: np.ndarray
    wait_ms: np.ndarray
    tokens: np.ndarray
    per_ktok: np.ndarray

    def col(self, name: str) -> np.ndarray:
        return self.x[:, FIELD_INDEX[name]]


def _window(table: Sequence[Sequence[float]]) -> _Window:
    x = np.asarray(table, dtype=np.float64)
    raw = {name: x[:, FIELD_INDEX[name]].copy() for name in ("dropped", "unread", "errors", "health")}
    # Everything else per step, so numbers stay comparable across intervals.
    x = x / np.maximum(x[:, FIELD_INDEX["num_steps"]], 1.0)[:, None]
    self_ms = sum(x[:, FIELD_INDEX[name]] for name in SELF_SEGMENTS)
    tokens = x[:, FIELD_INDEX["tokens"]]
    return _Window(
        x=x,
        **raw,
        self_ms=self_ms,
        wait_ms=sum(x[:, FIELD_INDEX[name]] for name in WAIT_SEGMENTS),
        tokens=tokens,
        per_ktok=_per_ktok(self_ms, tokens),
    )


def _per_ktok(ms: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    return ms / np.maximum(tokens, 1.0) * 1e3


@dataclass
class _Peers:
    """Each rank against the leave-one-out peers of its PP stage."""

    rel_self: np.ndarray
    z_self: np.ndarray
    rel_fwd: np.ndarray
    z_fwd: np.ndarray
    rel_tok: np.ndarray
    rel_ktok: np.ndarray
    rel_fwd_ktok: np.ndarray
    rel_wait: np.ndarray
    segment_rel: dict[str, np.ndarray]
    stage_self: np.ndarray  # median self time of the rank's stage
    stage_medians: list[float]
    group_size: np.ndarray
    group_has_tokens: np.ndarray


def _stage_peers(w: _Window, meta: Sequence[RankMeta]) -> _Peers:
    world = len(meta)
    zeros = ("rel_self", "z_self", "rel_fwd", "z_fwd", "rel_tok", "rel_ktok", "rel_fwd_ktok", "rel_wait")
    p = _Peers(
        **{name: np.zeros(world) for name in zeros},
        segment_rel={name: np.zeros(world) for name in GPU_SEGMENTS},
        stage_self=np.zeros(world),
        stage_medians=[],
        group_size=np.zeros(world),
        group_has_tokens=np.zeros(world, dtype=bool),
    )
    fwd = w.col("fwd")
    fwd_per_ktok = _per_ktok(fwd, w.tokens)
    for idx in _group_indices(meta, lambda m: m.pp):
        median_self = float(np.median(w.self_ms[idx]))
        p.stage_medians.append(median_self)
        p.stage_self[idx] = median_self
        p.group_size[idx] = idx.shape[0]
        p.group_has_tokens[idx] = bool(np.all(w.tokens[idx] > 0))
        p.rel_self[idx], p.z_self[idx], _ = _relative_and_z(w.self_ms[idx])
        p.rel_fwd[idx], p.z_fwd[idx], _ = _relative_and_z(fwd[idx])
        p.rel_tok[idx], _, _ = _relative_and_z(w.tokens[idx])
        p.rel_ktok[idx], _, _ = _relative_and_z(w.per_ktok[idx])
        p.rel_fwd_ktok[idx], _, _ = _relative_and_z(fwd_per_ktok[idx])
        p.rel_wait[idx], _, _ = _relative_and_z(w.wait_ms[idx])
        for name in GPU_SEGMENTS:
            p.segment_rel[name][idx], _, _ = _relative_and_z(w.col(name)[idx])
    return p


@dataclass
class _Late:
    """Lateness at the DP grad-sync collective."""

    ms: np.ndarray  # how much earlier than this rank its peers arrived
    rel: np.ndarray  # ms as a fraction of the peers' idle time
    z: np.ndarray
    peer_idle: np.ndarray  # how long the peers waited: ms plus this rank's own grad-sync bracket


def _late_arrival(w: _Window, meta: Sequence[RankMeta]) -> _Late:
    # Megatron reduce-scatters dense grads over the DP x CP group (and EP ranks
    # share the dense parameters), so a rank's peers are those with the same (pp, tp).
    sync_ms = w.col("dp_grad_sync")
    bwd_sync_ms = w.col("bwd") + sync_ms
    world = len(meta)
    late = _Late(ms=np.zeros(world), rel=np.zeros(world), z=np.zeros(world), peer_idle=sync_ms.copy())
    for idx in _group_indices(meta, lambda m: (m.pp, m.tp)):
        # Without a grad-sync bracket on every member there is no collective to place.
        if idx.shape[0] < 2 or not np.all(sync_ms[idx] > _EPS):
            continue
        _, z, peer_bwd_sync = _relative_and_z(bwd_sync_ms[idx])
        late.ms[idx] = np.maximum(peer_bwd_sync - bwd_sync_ms[idx], 0.0)
        late.peer_idle[idx] = late.ms[idx] + sync_ms[idx]
        late.rel[idx] = late.ms[idx] / np.maximum(late.peer_idle[idx], _EPS)
        late.z[idx] = np.maximum(-z, 0.0)
    return late


@dataclass
class _Wait:
    """Each rank's ``pp_recv`` against reference ranks of its stage."""

    rel: np.ndarray
    z: np.ndarray
    peer: np.ndarray  # the reference's median pp_recv
    has_reference: np.ndarray


def _pp_wait(pp_recv: np.ndarray, meta: Sequence[RankMeta], reference: np.ndarray) -> _Wait:
    """Compare each rank with the ``reference`` ranks of its stage other than
    itself."""
    world = len(meta)
    wait = _Wait(
        rel=np.zeros(world), z=np.zeros(world), peer=pp_recv.copy(), has_reference=np.zeros(world, dtype=bool)
    )
    for idx in _group_indices(meta, lambda m: m.pp):
        members, others = idx[reference[idx]], idx[~reference[idx]]
        if members.size >= 2:
            wait.rel[members], wait.z[members], wait.peer[members] = _relative_and_z(pp_recv[members])
            wait.has_reference[members] = True
        if members.size and others.size:
            med = np.full(others.size, np.median(pp_recv[members]))
            mad = np.full(others.size, np.median(np.abs(pp_recv[members] - med[0])))
            wait.rel[others], wait.z[others] = _excess(pp_recv[others], med, mad)
            wait.peer[others] = med
            wait.has_reference[others] = True
    return wait


def _with_fallback(wait: _Wait, fallback: _Wait) -> _Wait:
    """``wait``, with ``fallback`` for the ranks it has no reference for."""
    keep = wait.has_reference
    return _Wait(
        rel=np.where(keep, wait.rel, fallback.rel),
        z=np.where(keep, wait.z, fallback.z),
        peer=np.where(keep, wait.peer, fallback.peer),
        has_reference=keep | fallback.has_reference,
    )


def _wait_signal(pp_recv: np.ndarray, wait: _Wait, peers: _Peers, config: DetectorConfig) -> np.ndarray:
    return (
        (pp_recv > _EPS)
        & (wait.rel >= config.rel_threshold)
        & (wait.z >= config.z_threshold)
        & (pp_recv - wait.peer >= config.wait_abs_frac * peers.stage_self)
    )


def _uncertain_causes(w: _Window, peers: _Peers, self_candidates: np.ndarray) -> list[str]:
    """First applicable cause per rank; ``""`` when the window supports a
    verdict."""
    causes = []
    for i in range(w.x.shape[0]):
        if w.dropped[i] > 0:
            causes.append("dropped_events")
        elif w.unread[i] > 0:
            causes.append("unread_events")
        elif w.errors[i] > 0:
            causes.append("profiler_errors")
        elif peers.group_size[i] < 2:
            causes.append("no_peers")
        elif self_candidates[i] and not peers.group_has_tokens[i]:
            causes.append("missing_tokens")
        else:
            causes.append("")
    return causes


def _uncertain_message(rank_meta: RankMeta, cause: str, w: _Window, peers: _Peers, i: int) -> str:
    detail = {
        "dropped_events": f"{w.dropped[i]:.0f} event pairs were dropped (pending queue full), so its segment totals "
        "are incomplete",
        "unread_events": f"{w.unread[i]:.0f} event pairs were still in flight when the window closed; their time "
        "lands in the next window",
        "profiler_errors": f"the profiler caught {w.errors[i]:.0f} errors on this rank during the window",
        "no_peers": "its pipeline stage has no other rank to compare with",
        "missing_tokens": f"it looks slow (self {1 + peers.rel_self[i]:.2f}x, fwd {1 + peers.rel_fwd[i]:.2f}x its "
        "stage peers) but the stage has no token counts, so data imbalance cannot be ruled out",
        "no_wait_reference": f"it waits {w.col('pp_recv')[i]:.1f} ms/step on PP peers, but every other rank of its "
        "stage is flagged, late or uncertain, so there is nothing to compare that wait with",
    }[cause]
    return f"rank {rank_meta.rank} ({rank_meta.tag}) is uncertain: {detail}"


def _update_streaks(
    state: DetectorState,
    meta: Sequence[RankMeta],
    certain: np.ndarray,
    candidates: np.ndarray,
    waits: np.ndarray,
) -> None:
    for i, rank_meta in enumerate(meta):
        rank = rank_meta.rank
        if not certain[i]:
            # No evidence either way: a streak cannot span this window, and it
            # is not a clean window for recovery.
            for streaks in (state.consecutive, state.waiting, state.clean):
                streaks.pop(rank, None)
            continue
        for streaks, hit in ((state.consecutive, candidates[i]), (state.waiting, waits[i])):
            if hit:
                streaks[rank] = streaks.get(rank, 0) + 1
            else:
                streaks.pop(rank, None)
        if rank in state.active:
            if candidates[i] or waits[i]:
                state.clean.pop(rank, None)
            else:
                state.clean[rank] = state.clean.get(rank, 0) + 1


def _culprit(
    i: int,
    rank_meta: RankMeta,
    windows: int,
    w: _Window,
    peers: _Peers,
    late: _Late,
    self_candidate: bool,
    by_fwd: bool,
    config: DetectorConfig,
) -> tuple[str, str]:
    """Reason and message for a rank that qualified ``windows`` times in a row;
    ``by_fwd`` when only its forward, not its self total, stands out."""
    who = f"rank {rank_meta.rank} ({rank_meta.tag}, host={rank_meta.host}, gpu={rank_meta.device})"
    gc_ms = w.col("gc")[i]
    fwd = w.col("fwd")[i]
    if self_candidate:
        if by_fwd:
            lead = (
                f"fwd {fwd:.1f} ms/step = {1 + peers.rel_fwd[i]:.2f}x peers "
                f"(z={peers.z_fwd[i]:.1f}, self {1 + peers.rel_self[i]:.2f}x)"
            )
            ktok, rel_ktok = "fwd ms/ktok", peers.rel_fwd_ktok[i]
        else:
            lead = f"self {w.self_ms[i]:.1f} ms/step = {1 + peers.rel_self[i]:.2f}x peers (z={peers.z_self[i]:.1f})"
            ktok, rel_ktok = "ms/ktok", peers.rel_ktok[i]
        if peers.rel_tok[i] >= config.rel_threshold and rel_ktok < config.rel_threshold:
            reason = "data_imbalance"
        elif w.self_ms[i] > _EPS and gc_ms / w.self_ms[i] >= config.gc_ratio_threshold:
            reason = "cpu_bound"
        else:
            reason = "slow_device"
        return reason, (
            f"{who} {lead} for {windows} windows; tokens {1 + peers.rel_tok[i]:.2f}x, "
            f"{ktok} {1 + rel_ktok:.2f}x, gc {gc_ms:.1f} ms -> {reason}"
        )
    idle_frac = late.peer_idle[i] / peers.stage_self[i] if peers.stage_self[i] > _EPS else 0.0
    cpu_over_gpu = w.col("cpu_fwd")[i] / fwd if fwd > _EPS else 0.0
    return "late_arrival", (
        f"{who} reaches the DP grad-sync {late.ms[i]:.1f} ms/step after its DP peers "
        f"(peers idle {late.peer_idle[i]:.1f} ms/step = {idle_frac:.0%} of their compute, z={late.z[i]:.1f}) "
        f"for {windows} windows; its own GPU forward is {1 + peers.rel_fwd[i]:.2f}x peers, gc {gc_ms:.1f} ms, "
        f"cpu/gpu fwd {cpu_over_gpu:.2f}x -> late_arrival (host-side stall)"
    )


def _row(
    i: int, rank_meta: RankMeta, w: _Window, late: _Late, reason: str, cause: str
) -> dict[str, float | int | str]:
    return {
        "rank": rank_meta.rank,
        "tag": rank_meta.tag,
        "host": rank_meta.host,
        "device": rank_meta.device,
        "self_ms": float(w.self_ms[i]),
        "wait_ms": float(w.wait_ms[i]),
        "late_ms": float(late.ms[i]),
        "tokens": float(w.tokens[i]),
        "ms_per_ktok": float(w.per_ktok[i]),
        "gc_ms": float(w.col("gc")[i]),
        "reason": reason,
        "uncertain": cause,
        **{name: float(w.col(name)[i]) for name in GPU_SEGMENTS},
    }


def _emit(
    metrics: dict[str, float], prefix: str, values: np.ndarray, rel: np.ndarray, meta: Sequence[RankMeta]
) -> None:
    """``max_ms`` is the global maximum; ``max_rank`` / ``spread`` describe the
    rank with the largest excess over *its stage peers* (so a slower PP stage
    does not always win)."""
    if not np.any(values > _EPS):
        return
    worst = int(np.argmax(rel))
    metrics[f"straggler/{prefix}/median_ms"] = float(np.median(values))
    metrics[f"straggler/{prefix}/max_ms"] = float(np.max(values))
    metrics[f"straggler/{prefix}/max_rank"] = float(meta[worst].rank)
    metrics[f"straggler/{prefix}/spread"] = float(max(rel[worst], 0.0))


def _metrics(
    w: _Window,
    peers: _Peers,
    late: _Late,
    meta: Sequence[RankMeta],
    active: dict[int, str],
    uncertain: list[Uncertain],
    recovered: list[Recovery],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name in GPU_SEGMENTS:
        _emit(metrics, name, w.col(name), peers.segment_rel[name], meta)
    _emit(metrics, "self", w.self_ms, peers.rel_self, meta)
    _emit(metrics, "wait", w.wait_ms, peers.rel_wait, meta)

    latest = int(np.argmax(late.ms))
    is_late = late.ms[latest] > _EPS
    metrics["straggler/late/max_ms"] = float(late.ms[latest])
    metrics["straggler/late/max_rank"] = float(meta[latest].rank) if is_late else -1.0
    metrics["straggler/late/peer_idle_ms"] = float(late.peer_idle[latest]) if is_late else 0.0

    if np.all(w.tokens > 0):
        _emit(metrics, "self_per_ktok", w.per_ktok, peers.rel_ktok, meta)
        metrics["straggler/tokens/median"] = float(np.median(w.tokens))
        metrics["straggler/tokens/max"] = float(np.max(w.tokens))
        metrics["straggler/tokens/spread"] = float(max(np.max(peers.rel_tok), 0.0))
    gc_ms = w.col("gc")
    metrics["straggler/gc/median_ms"] = float(np.median(gc_ms))
    metrics["straggler/gc/max_ms"] = float(np.max(gc_ms))

    flagged = sorted((rank, reason) for rank, reason in active.items() if reason != "upstream_wait")
    metrics["straggler/flagged/count"] = float(len(flagged))
    metrics["straggler/flagged/rank"] = float(flagged[0][0]) if flagged else -1.0
    metrics["straggler/flagged/reason"] = float(REASON_CODE[flagged[0][1]]) if flagged else 0.0
    metrics["straggler/waiting/count"] = float(sum(reason == "upstream_wait" for reason in active.values()))
    metrics["straggler/uncertain/count"] = float(len(uncertain))
    metrics["straggler/uncertain/rank"] = float(uncertain[0].rank) if uncertain else -1.0
    metrics["straggler/uncertain/cause"] = float(UNCERTAIN_CODE[uncertain[0].cause]) if uncertain else 0.0
    metrics["straggler/recovered/count"] = float(len(recovered))

    stage_values = [value for value in peers.stage_medians if value > _EPS]
    metrics["straggler/pp_stage_imbalance"] = (
        float(max(stage_values) / min(stage_values)) if len(stage_values) > 1 else 1.0
    )
    metrics["straggler/self_overhead_ms"] = float(np.mean(w.col("overhead")))
    metrics["straggler/dropped_events"] = float(np.sum(w.dropped))
    metrics["straggler/unread_events"] = float(np.sum(w.unread))
    metrics["straggler/health/state"] = float(np.max(w.health))
    metrics["straggler/health/errors"] = float(np.sum(w.errors))
    metrics["straggler/health/degraded_ranks"] = float(np.sum(w.health > 0))
    return metrics


def analyze_window(
    table: Sequence[Sequence[float]],
    meta: Sequence[RankMeta],
    config: DetectorConfig,
    state: DetectorState,
) -> WindowReport:
    """Analyze one gathered window; mutates ``state`` for persistence."""
    w = _window(table)
    if w.x.shape[0] != len(meta):
        raise ValueError(
            f"straggler table has {w.x.shape[0]} rows but {len(meta)} rank metas; the value and metadata gathers "
            "must run over the same process group so row i describes rank i."
        )
    peers = _stage_peers(w, meta)
    late = _late_arrival(w, meta)

    self_signal = (peers.rel_self >= config.rel_threshold) & (peers.z_self >= config.z_threshold)
    fwd_signal = (peers.rel_fwd >= config.rel_threshold) & (peers.z_fwd >= config.z_threshold)
    self_candidates = self_signal | fwd_signal
    late_candidates = (
        (late.ms >= config.wait_abs_frac * peers.stage_self)
        & (late.rel >= config.rel_threshold)
        & (late.z >= config.z_threshold)
    )
    causes = _uncertain_causes(w, peers, self_candidates)
    certain = np.asarray([not cause for cause in causes], dtype=bool)
    self_candidates &= certain
    pp_recv = w.col("pp_recv")
    unflagged = certain & ~self_candidates
    wait = _pp_wait(pp_recv, meta, unflagged & ~late_candidates)
    # A rank that is late because it waits on PP is not a culprit; ranks that had
    # no reference without it are compared with it too.
    late_candidates &= ~_wait_signal(pp_recv, wait, peers, config)
    wait = _with_fallback(wait, _pp_wait(pp_recv, meta, unflagged & ~late_candidates))
    candidates = self_candidates | (late_candidates & certain)
    for i in np.flatnonzero(certain & ~candidates & (pp_recv > _EPS) & ~wait.has_reference):
        causes[i] = "no_wait_reference"
        certain[i] = False
    waits = certain & ~candidates & _wait_signal(pp_recv, wait, peers, config)
    _update_streaks(state, meta, certain, candidates, waits)

    alerts: list[Alert] = []
    uncertain: list[Uncertain] = []
    recovered: list[Recovery] = []
    rows: list[dict[str, float | int | str]] = []
    active: dict[int, str] = {}
    unsure_causes: dict[int, str] = {}
    for i, rank_meta in enumerate(meta):
        rank, cause = rank_meta.rank, causes[i]
        reason, message = "none", ""
        if not cause and state.consecutive.get(rank, 0) >= config.persist_windows:
            by_fwd = bool(fwd_signal[i] and not self_signal[i])
            reason, message = _culprit(
                i, rank_meta, state.consecutive[rank], w, peers, late, bool(self_candidates[i]), by_fwd, config
            )
        elif not cause and state.waiting.get(rank, 0) >= config.persist_windows:
            reason = "upstream_wait"
            message = (
                f"rank {rank} ({rank_meta.tag}) waits on PP peers {pp_recv[i]:.1f} ms/step = "
                f"{1 + wait.rel[i]:.2f}x its stage for {state.waiting[rank]} windows; its own compute is normal"
            )
        previous = state.active.get(rank)
        if reason == "none" and previous is not None:
            clean = state.clean.get(rank, 0)
            if cause or clean < config.recover_windows:
                reason = previous
                message = f"rank {rank} ({rank_meta.tag}) still {previous}: {clean}/{config.recover_windows} " + (
                    f"clean windows, this window uncertain ({cause})" if cause else "clean windows"
                )
            else:
                state.clean.pop(rank, None)
                recovered.append(
                    Recovery(
                        rank,
                        previous,
                        f"rank {rank} ({rank_meta.tag}) recovered from {previous} after {clean} clean windows",
                    )
                )
        if reason != "none":
            active[rank] = reason
            alerts.append(Alert(rank, reason, message, new=previous != reason))
        if cause:
            unsure_causes[rank] = cause
            uncertain.append(
                Uncertain(
                    rank,
                    cause,
                    _uncertain_message(rank_meta, cause, w, peers, i),
                    new=state.uncertain.get(rank) != cause,
                )
            )
        rows.append(_row(i, rank_meta, w, late, reason, cause))
    state.active = active
    state.uncertain = unsure_causes

    return WindowReport(
        metrics=_metrics(w, peers, late, meta, active, uncertain, recovered),
        alerts=alerts,
        rows=rows,
        uncertain=uncertain,
        recovered=recovered,
    )
