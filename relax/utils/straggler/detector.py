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

* **self compute** (``fwd + bwd + optim`` compute-stream time): a rank whose
  own step takes longer than its stage peers -> ``slow_device`` /
  ``data_imbalance`` / ``cpu_bound``.
* **late arrival** at the DP grad-sync collective: every peer's
  ``dp_grad_sync`` bracket ends when the collective completes, so a rank that
  shows a much *shorter* bracket than its peers is the one everybody waited
  for. Its GPU kernels may be perfectly normal; the lost time sits between
  kernels on the host (data fetch, GIL / GC, blocking I/O) -> ``late_arrival``.

Segment times come from events on the compute stream, so they include host
launch gaps inside the bracket; "GPU time" below is shorthand for that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from relax.utils.straggler.stats import (
    FIELD_INDEX,
    GPU_SEGMENTS,
    LATE_ARRIVAL_SEGMENT,
    SELF_SEGMENTS,
    WAIT_SEGMENTS,
)


# Order is part of the metric contract (``straggler/flagged/reason`` logs the code).
REASONS: tuple[str, ...] = ("none", "slow_device", "data_imbalance", "upstream_wait", "cpu_bound", "late_arrival")
REASON_CODE: dict[str, int] = {reason: code for code, reason in enumerate(REASONS)}

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
    """Thresholds for ``analyze_window``; the first three come from the
    ``RELAX_STRAGGLER_*`` environment variables in production."""

    z_threshold: float = 3.0
    rel_threshold: float = 0.10
    persist_windows: int = 3
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


@dataclass(frozen=True)
class Alert:
    """Verdict for one rank; ``new`` is True when its reason changed since the
    previous window (only those are logged as WARNING)."""

    rank: int
    reason: str
    message: str
    new: bool


@dataclass
class WindowReport:
    """Scalars, alerts and per-rank rows produced by one ``analyze_window``."""

    metrics: dict[str, float]
    alerts: list[Alert]
    rows: list[dict[str, float | int | str]]
    # Wall-clock start of the window; filled in by the collector for the timeline.
    window_start_wall: float = 0.0


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
    mad = _leave_one_out_mad(values, med)

    rel = np.zeros(n)
    positive = med > _EPS
    rel[positive] = np.minimum(values[positive] / med[positive] - 1.0, _REL_CAP)
    rel[~positive & (values > _EPS)] = _REL_CAP

    scale = 1.4826 * mad
    fallback = np.maximum(0.01 * np.maximum(np.abs(med), np.abs(values)), _EPS)
    scale = np.where(scale < _EPS, fallback, scale)
    return rel, (values - med) / scale, med


def _column(table: np.ndarray, name: str) -> np.ndarray:
    return table[:, FIELD_INDEX[name]]


def analyze_window(
    table: Sequence[Sequence[float]],
    meta: Sequence[RankMeta],
    config: DetectorConfig,
    state: DetectorState,
) -> WindowReport:
    """Analyze one gathered window.

    Mutates ``state`` for persistence.
    """
    x = np.asarray(table, dtype=np.float64)
    world = x.shape[0]
    if world != len(meta):
        raise ValueError(
            f"straggler table has {world} rows but {len(meta)} rank metas; the value and metadata gathers must "
            "run over the same process group so row i describes rank i."
        )

    # Everything below is per step so numbers stay comparable across intervals.
    steps = np.maximum(_column(x, "num_steps"), 1.0)[:, None]
    x = x / steps

    self_ms = sum(_column(x, name) for name in SELF_SEGMENTS)
    wait_ms = sum(_column(x, name) for name in WAIT_SEGMENTS)
    pp_recv = _column(x, "pp_recv")
    tokens = _column(x, "tokens")
    has_tokens = bool(np.all(tokens > 0))
    per_ktok = self_ms / np.maximum(tokens, 1.0) * 1e3

    rel_self = np.zeros(world)
    z_self = np.zeros(world)
    rel_tok = np.zeros(world)
    rel_ktok = np.zeros(world)
    rel_recv = np.zeros(world)
    z_recv = np.zeros(world)
    peer_recv = np.zeros(world)
    rel_wait = np.zeros(world)
    stage_self_of = np.zeros(world)
    segment_rel = {name: np.zeros(world) for name in GPU_SEGMENTS}
    stage_median_self: dict[int, float] = {}

    groups: dict[int, list[int]] = {}
    for index, rank_meta in enumerate(meta):
        groups.setdefault(rank_meta.pp, []).append(index)

    for pp, members in groups.items():
        idx = np.asarray(members)
        stage_median_self[pp] = float(np.median(self_ms[idx]))
        stage_self_of[idx] = stage_median_self[pp]
        rel_self[idx], z_self[idx], _ = _relative_and_z(self_ms[idx])
        rel_tok[idx], _, _ = _relative_and_z(tokens[idx])
        rel_ktok[idx], _, _ = _relative_and_z(per_ktok[idx])
        rel_recv[idx], z_recv[idx], peer_recv[idx] = _relative_and_z(pp_recv[idx])
        rel_wait[idx], _, _ = _relative_and_z(wait_ms[idx])
        for name in GPU_SEGMENTS:
            segment_rel[name][idx], _, _ = _relative_and_z(_column(x, name)[idx])

    # Late arrival: within a grad-sync group the bracket of the last rank to
    # arrive is just the transfer; everyone else's bracket also contains the wait
    # for it. ``late_ms[i]`` = how much earlier than rank i its peers arrived.
    # Megatron reduce-scatters dense grads over the DP x CP group (and EP ranks
    # share the dense parameters), so peers are the ranks with the same (pp, tp).
    sync_ms = _column(x, LATE_ARRIVAL_SEGMENT)
    late_ms = np.zeros(world)
    rel_late = np.zeros(world)
    z_late = np.zeros(world)
    peer_sync = sync_ms.copy()
    dp_groups: dict[tuple[int, int], list[int]] = {}
    for index, rank_meta in enumerate(meta):
        dp_groups.setdefault((rank_meta.pp, rank_meta.tp), []).append(index)
    for members in dp_groups.values():
        idx = np.asarray(members)
        if idx.shape[0] < 2:
            continue
        rel_sync, z_sync, peer_sync[idx] = _relative_and_z(sync_ms[idx])
        late_ms[idx] = np.maximum(peer_sync[idx] - sync_ms[idx], 0.0)
        rel_late[idx] = np.maximum(-rel_sync, 0.0)
        z_late[idx] = np.maximum(-z_sync, 0.0)
    late_candidates = (
        (late_ms >= config.wait_abs_frac * stage_self_of)
        & (rel_late >= config.rel_threshold)
        & (z_late >= config.z_threshold)
    )

    self_candidates = (rel_self >= config.rel_threshold) & (z_self >= config.z_threshold)
    candidates = self_candidates | late_candidates
    wait_candidates = (
        ~candidates
        & (pp_recv > _EPS)
        & (rel_recv >= config.rel_threshold)
        & (z_recv >= config.z_threshold)
        & (pp_recv - peer_recv >= config.wait_abs_frac * stage_self_of)
    )
    for index, rank_meta in enumerate(meta):
        for streaks, hit in ((state.consecutive, candidates[index]), (state.waiting, wait_candidates[index])):
            if hit:
                streaks[rank_meta.rank] = streaks.get(rank_meta.rank, 0) + 1
            else:
                streaks.pop(rank_meta.rank, None)

    alerts: list[Alert] = []
    rows: list[dict[str, float | int | str]] = []
    new_active: dict[int, str] = {}
    fwd = _column(x, "fwd")
    cpu_fwd = _column(x, "cpu_fwd")
    gc_ms = _column(x, "gc")

    for index, rank_meta in enumerate(meta):
        reason, message = "none", ""
        if state.consecutive.get(rank_meta.rank, 0) >= config.persist_windows:
            who = f"rank {rank_meta.rank} ({rank_meta.tag}, host={rank_meta.host}, gpu={rank_meta.device})"
            if self_candidates[index]:
                if has_tokens and rel_tok[index] >= config.rel_threshold and rel_ktok[index] < config.rel_threshold:
                    reason = "data_imbalance"
                elif self_ms[index] > _EPS and gc_ms[index] / self_ms[index] >= config.gc_ratio_threshold:
                    reason = "cpu_bound"
                else:
                    reason = "slow_device"
                message = (
                    f"{who} self {self_ms[index]:.1f} ms/step = {1 + rel_self[index]:.2f}x peers "
                    f"(z={z_self[index]:.1f}) for {state.consecutive[rank_meta.rank]} windows; "
                    f"tokens {1 + rel_tok[index]:.2f}x, ms/ktok {1 + rel_ktok[index]:.2f}x, "
                    f"gc {gc_ms[index]:.1f} ms -> {reason}"
                )
            else:
                reason = "late_arrival"
                idle_frac = peer_sync[index] / stage_self_of[index] if stage_self_of[index] > _EPS else 0.0
                message = (
                    f"{who} reaches the DP grad-sync {late_ms[index]:.1f} ms/step after its DP peers "
                    f"(peers idle {peer_sync[index]:.1f} ms/step = {idle_frac:.0%} of their compute, "
                    f"z={z_late[index]:.1f}) for {state.consecutive[rank_meta.rank]} windows; its own GPU compute "
                    f"is {1 + rel_self[index]:.2f}x peers, gc {gc_ms[index]:.1f} ms, cpu/gpu fwd "
                    f"{cpu_fwd[index] / fwd[index] if fwd[index] > _EPS else 0.0:.2f}x -> {reason} (host-side stall)"
                )
        elif state.waiting.get(rank_meta.rank, 0) >= config.persist_windows:
            reason = "upstream_wait"
            message = (
                f"rank {rank_meta.rank} ({rank_meta.tag}) waits on PP peers {pp_recv[index]:.1f} ms/step = "
                f"{1 + rel_recv[index]:.2f}x its stage for {state.waiting[rank_meta.rank]} windows; "
                "its own compute is normal"
            )
        if reason != "none":
            new_active[rank_meta.rank] = reason
            alerts.append(Alert(rank_meta.rank, reason, message, new=state.active.get(rank_meta.rank) != reason))
        rows.append(
            {
                "rank": rank_meta.rank,
                "tag": rank_meta.tag,
                "host": rank_meta.host,
                "device": rank_meta.device,
                "self_ms": float(self_ms[index]),
                "wait_ms": float(wait_ms[index]),
                "late_ms": float(late_ms[index]),
                "tokens": float(tokens[index]),
                "ms_per_ktok": float(per_ktok[index]),
                "gc_ms": float(gc_ms[index]),
                "reason": reason,
                **{name: float(_column(x, name)[index]) for name in GPU_SEGMENTS},
            }
        )
    state.active = new_active

    metrics: dict[str, float] = {}

    def _emit(prefix: str, values: np.ndarray, rel: np.ndarray) -> None:
        """``max_ms`` is the global maximum; ``max_rank`` / ``spread`` describe
        the rank with the largest excess over *its stage peers* (so a slower PP
        stage does not always win)."""
        if not np.any(values > _EPS):
            return
        worst = int(np.argmax(rel))
        metrics[f"straggler/{prefix}/median_ms"] = float(np.median(values))
        metrics[f"straggler/{prefix}/max_ms"] = float(np.max(values))
        metrics[f"straggler/{prefix}/max_rank"] = float(meta[worst].rank)
        metrics[f"straggler/{prefix}/spread"] = float(max(rel[worst], 0.0))

    for name in GPU_SEGMENTS:
        _emit(name, _column(x, name), segment_rel[name])
    _emit("self", self_ms, rel_self)
    _emit("wait", wait_ms, rel_wait)
    latest = int(np.argmax(late_ms))
    metrics["straggler/late/max_ms"] = float(late_ms[latest])
    metrics["straggler/late/max_rank"] = float(meta[latest].rank) if late_ms[latest] > _EPS else -1.0
    metrics["straggler/late/peer_idle_ms"] = float(peer_sync[latest]) if late_ms[latest] > _EPS else 0.0
    if has_tokens:
        _emit("self_per_ktok", per_ktok, rel_ktok)
        metrics["straggler/tokens/median"] = float(np.median(tokens))
        metrics["straggler/tokens/max"] = float(np.max(tokens))
        metrics["straggler/tokens/spread"] = float(max(np.max(rel_tok), 0.0))
    metrics["straggler/gc/median_ms"] = float(np.median(gc_ms))
    metrics["straggler/gc/max_ms"] = float(np.max(gc_ms))

    flagged = sorted((rank, reason) for rank, reason in new_active.items() if reason != "upstream_wait")
    metrics["straggler/flagged/count"] = float(len(flagged))
    metrics["straggler/flagged/rank"] = float(flagged[0][0]) if flagged else -1.0
    metrics["straggler/flagged/reason"] = float(REASON_CODE[flagged[0][1]]) if flagged else 0.0
    metrics["straggler/waiting/count"] = float(sum(reason == "upstream_wait" for reason in new_active.values()))
    stage_values = [value for value in stage_median_self.values() if value > _EPS]
    metrics["straggler/pp_stage_imbalance"] = (
        float(max(stage_values) / min(stage_values)) if len(stage_values) > 1 else 1.0
    )
    metrics["straggler/self_overhead_ms"] = float(np.mean(_column(x, "overhead")))
    metrics["straggler/dropped_events"] = float(np.sum(_column(x, "dropped") * steps[:, 0]))

    return WindowReport(metrics=metrics, alerts=alerts, rows=rows)
