# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Compare ranks inside one PP stage and pick a single diagnostic direction."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Mapping

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.stages import STAGES


COMPUTE_STAGES = ("fwd", "bwd", "optim")


@dataclass
class RankWindow:
    rank: int
    pp: int
    stages_ms: dict[str, float]
    tokens: float | None = None
    gc_ms: float = 0.0

    @property
    def compute_ms(self) -> float:
        return sum(self.stages_ms.get(name, 0.0) for name in COMPUTE_STAGES)


@dataclass
class Diagnostic:
    rank: int
    pp: int
    reason: str
    detail: str


@dataclass
class JudgeState:
    streak: dict[tuple[int, int], tuple[str, int]] = field(default_factory=dict)


def _high(value: float, peer_median: float, cfg: StragglerConfig) -> bool:
    if peer_median <= 0:
        return False
    return value >= peer_median * (1.0 + cfg.relative_threshold) or (value - peer_median) >= cfg.absolute_ms_threshold


def _low(value: float, peer_median: float, cfg: StragglerConfig) -> bool:
    if peer_median <= 0:
        return False
    return value <= peer_median * (1.0 - cfg.relative_threshold) and (peer_median - value) >= cfg.absolute_ms_threshold


def _peer_median(rows: list[RankWindow], rank: int, pick) -> float | None:
    peers = [pick(row) for row in rows if row.rank != rank]
    if not peers:
        return None
    return float(median(peers))


def _candidate(row: RankWindow, rows: list[RankWindow], cfg: StragglerConfig) -> str | None:
    compute_ref = _peer_median(rows, row.rank, lambda item: item.compute_ms)
    if compute_ref is None:
        return None
    compute = row.compute_ms
    if _high(compute, compute_ref, cfg):
        token_ref = _peer_median(rows, row.rank, lambda item: item.tokens or 0.0)
        if row.tokens is not None and token_ref is not None and token_ref > 0 and _high(row.tokens, token_ref, cfg):
            per_tok = compute / max(row.tokens, 1.0)
            peer_per = [
                item.compute_ms / max(item.tokens, 1.0) for item in rows if item.rank != row.rank and item.tokens
            ]
            if peer_per and per_tok <= float(median(peer_per)) * (1.0 + cfg.relative_threshold):
                return "data_imbalance"
        if compute > 0 and row.gc_ms >= 0.05 * compute:
            return "cpu_bound"
        return "slow_device"

    sync = row.stages_ms.get("dp_grad_sync", 0.0)
    sync_ref = _peer_median(rows, row.rank, lambda item: item.stages_ms.get("dp_grad_sync", 0.0))
    if sync_ref is not None and _low(sync, sync_ref, cfg):
        if (sync_ref - sync) >= 0.05 * max(compute_ref, cfg.absolute_ms_threshold):
            return "late_arrival"

    recv = row.stages_ms.get("pp_recv", 0.0)
    recv_ref = _peer_median(rows, row.rank, lambda item: item.stages_ms.get("pp_recv", 0.0))
    if recv_ref is not None and _high(recv, recv_ref, cfg):
        return "upstream_wait"
    return None


def judge(
    rows: list[RankWindow], cfg: StragglerConfig, state: JudgeState | None = None
) -> tuple[list[Diagnostic], JudgeState]:
    """Return alerts that have stayed over the line for ``persist_windows``
    windows."""
    state = state or JudgeState()
    grouped: dict[int, list[RankWindow]] = {}
    for row in rows:
        grouped.setdefault(row.pp, []).append(row)

    alerts: list[Diagnostic] = []
    seen: set[tuple[int, int]] = set()
    for pp, group in grouped.items():
        for row in group:
            key = (pp, row.rank)
            seen.add(key)
            reason = _candidate(row, group, cfg)
            prev_reason, count = state.streak.get(key, ("", 0))
            if reason is None:
                state.streak.pop(key, None)
                continue
            count = count + 1 if reason == prev_reason else 1
            state.streak[key] = (reason, count)
            if count < cfg.persist_windows:
                continue
            if reason == "upstream_wait":
                detail = "waiting on the previous PP stage"
            else:
                detail = reason
            alerts.append(Diagnostic(rank=row.rank, pp=pp, reason=reason, detail=detail))
    for key in list(state.streak):
        if key not in seen:
            state.streak.pop(key, None)
    return alerts, state


def stage_spread(rows: list[RankWindow], stage: str) -> dict[str, float]:
    values = [row.stages_ms.get(stage, 0.0) for row in rows]
    if not values:
        return {}
    mid = float(median(values))
    top = max(range(len(rows)), key=lambda idx: values[idx])
    spread = 0.0 if mid <= 0 else values[top] / mid - 1.0
    return {
        "median_ms": mid,
        "max_ms": values[top],
        "max_rank": float(rows[top].rank),
        "spread": spread,
    }


def metrics_from(rows: list[RankWindow], alerts: list[Diagnostic], dropped: int) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {"straggler/dropped_events": float(dropped)}
    for stage in STAGES:
        summary = stage_spread(rows, stage)
        if not summary or summary["max_ms"] <= 0:
            continue
        metrics[f"straggler/{stage}/median_ms"] = summary["median_ms"]
        metrics[f"straggler/{stage}/max_ms"] = summary["max_ms"]
        metrics[f"straggler/{stage}/max_rank"] = summary["max_rank"]
        metrics[f"straggler/{stage}/spread"] = summary["spread"]
    computes = [row.compute_ms for row in rows]
    if computes:
        mid = float(median(computes))
        top = max(range(len(rows)), key=lambda idx: computes[idx])
        metrics["straggler/self/median_ms"] = mid
        metrics["straggler/self/max_ms"] = computes[top]
        metrics["straggler/self/max_rank"] = float(rows[top].rank)
    tokens = [row.tokens for row in rows if row.tokens is not None]
    if tokens:
        mid = float(median(tokens))
        top_i = max(range(len(rows)), key=lambda idx: rows[idx].tokens or 0.0)
        top_tok = rows[top_i].tokens or 0.0
        metrics["straggler/tokens/median"] = mid
        metrics["straggler/tokens/max"] = float(top_tok)
        metrics["straggler/tokens/spread"] = 0.0 if mid <= 0 else top_tok / mid - 1.0
    by_pp: dict[int, list[float]] = {}
    for row in rows:
        by_pp.setdefault(row.pp, []).append(row.compute_ms)
    if len(by_pp) > 1:
        pp_meds = [float(median(vals)) for vals in by_pp.values() if vals]
        lo = min(pp_meds)
        metrics["straggler/pp_stage_imbalance"] = 1.0 if lo <= 0 else max(pp_meds) / lo
    flagged = [item for item in alerts if item.reason != "upstream_wait"]
    metrics["straggler/flagged/count"] = float(len(flagged))
    if flagged:
        metrics["straggler/flagged/rank"] = float(flagged[0].rank)
        metrics["straggler/flagged/reason"] = flagged[0].reason
    return metrics


def rows_from_payload(payload: Mapping) -> RankWindow:
    stages = {name: float(payload.get(name, 0.0)) for name in STAGES}
    tokens = payload.get("tokens")
    return RankWindow(
        rank=int(payload["rank"]),
        pp=int(payload.get("pp", 0)),
        stages_ms=stages,
        tokens=None if tokens is None else float(tokens),
        gc_ms=float(payload.get("gc_ms", 0.0)),
    )
