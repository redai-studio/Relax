# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Cross-rank comparison of stage timings: who is holding the cohort back.

The detector is the only part of the profiler that *judges*. It deliberately
keeps the judgement narrow, because a false accusation is worse than a missed
one:

* comparison happens only inside a :mod:`~relax.utils.straggler.identity`
  cohort — ranks that run the same program and differ only in the
  data-parallel dimension;
* the reference is the **fastest** peer in the window, not the mean, so one slow
  rank cannot drag the baseline towards itself (the classic way a 2-rank
  comparison hides a 2x straggler);
* a rank must deviate by more than ``work_tolerance`` for
  ``persist_windows`` consecutive windows before it is reported;
* a cohort smaller than ``min_cohort_size`` yields ``uncertain`` instead of a
  guess, and a single-rank run yields nothing at all.

Window membership is computed from each rank's *own* first observation, not from
an absolute clock: two ranks that began profiling seconds apart (a slow import, a
late actor start) still compare their first window against each other instead of
landing in windows that never overlap. On a real run where every rank starts
stepping together the two definitions coincide; when they do not, the relative
one is the one that compares like with like.

Host and device intervals are compared separately, and the verdict says which of
the two moved:

* ``host_only_stall`` — the host interval grew while the GPU-timeline interval did
  not, so the extra time is demonstrably outside the stream (host work, waiting);
* ``gpu_stream_stall`` — both grew, so the extra time is inside the GPU timeline.
  This is *not* proof of slow hardware: CUDA events measure elapsed time on the
  stream, so a host stall that leaves the stream idle is indistinguishable from
  slow kernels. Kernel-level attribution needs per-kernel data, which C2
  deliberately does not collect;
* ``attribution_unknown`` — the device interval is unavailable (no CUDA events).

Coverage gaps are counted, never guessed: a window holding a single rank of a
multi-rank cohort increments ``incomplete_windows`` instead of producing a
verdict, so a real finding is never buried under thousands of "I could not
compare" lines.
"""

import json
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from relax.utils.logging_utils import get_logger
from relax.utils.straggler.config import StragglerConfig


logger = get_logger(__name__)

#: Close window ``W`` once an envelope from ``W + 1 + WINDOW_GRACE`` arrives, so
#: cross-rank skew of up to one full window cannot drop a peer's samples.
WINDOW_GRACE = 1

#: Bound on simultaneously open windows; a silent rank cannot grow memory.
MAX_PENDING_WINDOWS = 8

#: Bound on retained verdicts (the report reads the newest ones).
MAX_VERDICTS = 512

VERDICT_STRAGGLER = "straggler"
VERDICT_RECOVERED = "recovered"
VERDICT_UNCERTAIN = "uncertain"


def _median(values: List[float]) -> float:
    """Median of a non-empty list."""
    return float(statistics.median(values))


def _relative_deviation(value: float, reference: float) -> float:
    """Signed slowdown of ``value`` against ``reference``, 0 when
    unmeasurable."""
    if reference <= 0.0:
        return 0.0
    return (value - reference) / reference


@dataclass(frozen=True)
class Verdict:
    """One judgement about one rank, one stage, one window."""

    kind: str
    cohort: str
    name: str
    rank: int
    label: str
    window_index: int
    deviation: float
    consecutive_windows: int
    rank_host_ms: float
    reference_host_ms: float
    rank_device_ms: Optional[float]
    reference_device_ms: Optional[float]
    host_only: bool
    cohort_size: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat JSON-friendly mapping."""
        return {
            "kind": self.kind,
            "cohort": self.cohort,
            "name": self.name,
            "rank": self.rank,
            "label": self.label,
            "window_index": self.window_index,
            "deviation": self.deviation,
            "consecutive_windows": self.consecutive_windows,
            "rank_host_ms": self.rank_host_ms,
            "reference_host_ms": self.reference_host_ms,
            "rank_device_ms": self.rank_device_ms,
            "reference_device_ms": self.reference_device_ms,
            "host_only": self.host_only,
            "cohort_size": self.cohort_size,
            "reason": self.reason,
        }

    def to_json(self) -> str:
        """Serialise as one JSONL line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def describe(self) -> str:
        """One human-readable line for the periodic report."""
        if self.kind == VERDICT_UNCERTAIN:
            return (
                f"uncertain: {self.label} cohort={self.cohort} stage={self.name} "
                f"window={self.window_index} cohort_size={self.cohort_size} ({self.reason})"
            )
        change = "slower" if self.kind == VERDICT_STRAGGLER else "back within tolerance"
        return (
            f"{self.kind}: {self.label} stage={self.name} window={self.window_index} "
            f"{self.rank_host_ms:.2f}ms vs fastest {self.reference_host_ms:.2f}ms "
            f"({self.deviation:+.1%}, {change} for {self.consecutive_windows} window(s), "
            f"{'host-only' if self.host_only else 'stream-visible'})"
        )


class _Window:
    """Per-window samples: ``(cohort, name) -> rank -> [host_ms,
    device_ms]``."""

    __slots__ = ("index", "samples", "labels", "world_size")

    def __init__(self, index: int) -> None:
        self.index = index
        self.samples: Dict[Tuple[str, str], Dict[int, List[Tuple[float, Optional[float]]]]] = {}
        self.labels: Dict[int, str] = {}
        self.world_size = 1

    def add(self, envelope: Any) -> None:
        """Record one envelope."""
        key = (envelope.cohort, envelope.name)
        per_rank = self.samples.setdefault(key, {})
        per_rank.setdefault(envelope.rank, []).append((envelope.host_ms, envelope.device_ms))
        label = getattr(envelope, "label", "") or f"rank{envelope.rank}"
        self.labels.setdefault(envelope.rank, label)
        self.world_size = max(self.world_size, int(getattr(envelope, "world_size", 1) or 1))


class StragglerDetector:
    """Accumulates envelopes into windows and reports persistent outliers."""

    def __init__(self, config: StragglerConfig) -> None:
        self._config = config
        self._windows: Dict[int, _Window] = {}
        self._verdicts: Deque[Verdict] = deque(maxlen=MAX_VERDICTS)
        self._streak: Dict[Tuple[str, str, int], int] = {}
        self._epoch: Dict[int, float] = {}
        self._active: Set[Tuple[str, str, int]] = set()
        self._labels: Dict[Tuple[str, str, int], str] = {}
        self._counters: Dict[str, int] = {
            "envelopes": 0,
            "windows_closed": 0,
            "windows_forced": 0,
            "warmup_windows_skipped": 0,
            "incomplete_windows": 0,
            "cohort_stage_pairs": 0,
            "uncertain_judgements": 0,
            "single_rank_windows": 0,
            "stragglers_reported": 0,
            "recoveries_reported": 0,
        }

    @property
    def window_seconds(self) -> float:
        """Window length in seconds."""
        return self._config.window_seconds

    def observe(self, envelope: Any) -> List[Verdict]:
        """Add one envelope and return any verdicts the window boundary
        produced."""
        self._counters["envelopes"] += 1
        try:
            host_start = float(envelope.host_start)
            epoch = self._epoch.get(int(envelope.rank))
            if epoch is None:
                epoch = host_start
                self._epoch[int(envelope.rank)] = epoch
            # Integer microseconds: `(4.1 - 0.1) / 1.0` floors to 3 in binary
            # floating point, which would silently merge two windows.
            window_us = max(1, int(round(self._config.window_seconds * 1e6)))
            index = int(round(max(0.0, host_start - epoch) * 1e6)) // window_us
            window = self._windows.get(index)
            if window is None:
                window = _Window(index)
                self._windows[index] = window
            window.add(envelope)
        except Exception:
            self._counters["uncertain_judgements"] += 1
            return []

        verdicts: List[Verdict] = []
        try:
            verdicts.extend(self._close_ready_windows())
        except Exception:
            logger.warning("straggler detector failed to close a window", exc_info=True)
        return verdicts

    def _close_ready_windows(self) -> List[Verdict]:
        """Close windows that are old enough or that exceeded the memory
        bound."""
        verdicts: List[Verdict] = []
        while self._windows:
            newest = max(self._windows)
            oldest = min(self._windows)
            too_old = oldest <= newest - (1 + WINDOW_GRACE)
            overflow = len(self._windows) > MAX_PENDING_WINDOWS
            if not (too_old or overflow):
                break
            if overflow and not too_old:
                self._counters["windows_forced"] += 1
            verdicts.extend(self._close(oldest))
        return verdicts

    def _close(self, index: int) -> List[Verdict]:
        """Evaluate and drop one window."""
        window = self._windows.pop(index, None)
        if window is None:
            return []
        self._counters["windows_closed"] += 1
        if index < self._config.warmup_windows:
            # Startup is not a straggler: lazy CUDA allocation and the first data
            # batch make every rank look momentarily slow. Counted, not hidden.
            self._counters["warmup_windows_skipped"] += 1
            return []
        verdicts: List[Verdict] = []
        for (cohort, name), per_rank in window.samples.items():
            verdicts.extend(self._judge(cohort, name, window, per_rank))
        return verdicts

    def _judge(
        self,
        cohort: str,
        name: str,
        window: _Window,
        per_rank: Dict[int, List[Tuple[float, Optional[float]]]],
    ) -> List[Verdict]:
        """Compare each rank of one (cohort, stage) pair against its fastest
        peer."""
        self._counters["cohort_stage_pairs"] += 1
        ranks = sorted(per_rank)
        host_medians = {rank: _median([host for host, _ in per_rank[rank]]) for rank in ranks}
        device_medians: Dict[int, Optional[float]] = {}
        for rank in ranks:
            devices = [device for _, device in per_rank[rank] if device is not None]
            device_medians[rank] = _median(devices) if devices else None

        cohort_size = len(ranks)
        if cohort_size < 2:
            # A single reporting rank cannot be compared against anything: the
            # fast ranks are idle waiting for the straggler, so their envelopes
            # are simply elsewhere. Counting is honest; guessing is not.
            if window.world_size <= 1:
                self._counters["single_rank_windows"] += 1
            else:
                self._counters["incomplete_windows"] += 1
            return []
        if cohort_size < self._config.min_cohort_size:
            self._counters["uncertain_judgements"] += 1
            return [
                self._verdict(
                    kind=VERDICT_UNCERTAIN,
                    cohort=cohort,
                    name=name,
                    rank=rank,
                    label=window.labels.get(rank, f"rank{rank}"),
                    window_index=window.index,
                    deviation=0.0,
                    consecutive_windows=0,
                    rank_host_ms=host_medians[rank],
                    reference_host_ms=min(host_medians.values()),
                    rank_device_ms=device_medians[rank],
                    reference_device_ms=min(
                        (value for value in device_medians.values() if value is not None), default=None
                    ),
                    host_only=False,
                    cohort_size=cohort_size,
                    reason="cohort_below_min_size",
                )
                for rank in ranks
            ]

        host_reference = min(host_medians.values())
        device_values = [value for value in device_medians.values() if value is not None]
        device_reference = min(device_values) if device_values else None
        verdicts: List[Verdict] = []
        for rank in ranks:
            deviation = _relative_deviation(host_medians[rank], host_reference)
            key = (cohort, name, rank)
            label = window.labels.get(rank, f"rank{rank}")
            self._labels[key] = label
            slow = deviation > self._config.work_tolerance
            streak = self._streak.get(key, 0) + 1 if slow else 0
            self._streak[key] = streak
            device_deviation = (
                _relative_deviation(device_medians[rank], device_reference)
                if device_medians[rank] is not None and device_reference is not None
                else 0.0
            )
            stream_visible = device_medians[rank] is not None and device_reference is not None
            host_only = slow and stream_visible and device_deviation <= self._config.work_tolerance
            if streak >= self._config.persist_windows and key not in self._active:
                self._active.add(key)
                self._counters["stragglers_reported"] += 1
                verdicts.append(
                    self._verdict(
                        kind=VERDICT_STRAGGLER,
                        cohort=cohort,
                        name=name,
                        rank=rank,
                        label=label,
                        window_index=window.index,
                        deviation=deviation,
                        consecutive_windows=streak,
                        rank_host_ms=host_medians[rank],
                        reference_host_ms=host_reference,
                        rank_device_ms=device_medians[rank],
                        reference_device_ms=device_reference,
                        host_only=host_only,
                        cohort_size=cohort_size,
                        reason=(
                            "host_only_stall"
                            if host_only
                            else ("gpu_stream_stall" if stream_visible else "attribution_unknown")
                        ),
                    )
                )
            elif streak == 0 and key in self._active:
                self._active.discard(key)
                self._counters["recoveries_reported"] += 1
                verdicts.append(
                    self._verdict(
                        kind=VERDICT_RECOVERED,
                        cohort=cohort,
                        name=name,
                        rank=rank,
                        label=label,
                        window_index=window.index,
                        deviation=deviation,
                        consecutive_windows=streak,
                        rank_host_ms=host_medians[rank],
                        reference_host_ms=host_reference,
                        rank_device_ms=device_medians[rank],
                        reference_device_ms=device_reference,
                        host_only=False,
                        cohort_size=cohort_size,
                        reason="within_tolerance",
                    )
                )
        return verdicts

    def _verdict(self, **kwargs: Any) -> Verdict:
        """Build and retain a verdict."""
        verdict = Verdict(**kwargs)
        self._verdicts.append(verdict)
        return verdict

    def flush(self) -> List[Verdict]:
        """Close every open window, e.g. at the end of a run."""
        verdicts: List[Verdict] = []
        while self._windows:
            verdicts.extend(self._close(min(self._windows)))
        return verdicts

    def drain_verdicts(self) -> List[Verdict]:
        """Return and clear the retained verdicts."""
        verdicts = list(self._verdicts)
        self._verdicts.clear()
        return verdicts

    def active_stragglers(self) -> List[Dict[str, Any]]:
        """Return the currently flagged ``(cohort, stage, rank)`` triples."""
        return [
            {"cohort": cohort, "name": name, "rank": rank, "label": self._labels.get((cohort, name, rank), "")}
            for cohort, name, rank in sorted(self._active)
        ]

    def stats(self) -> Dict[str, Any]:
        """Return counters plus open-window state."""
        stats: Dict[str, Any] = dict(self._counters)
        stats["open_windows"] = len(self._windows)
        stats["retained_verdicts"] = len(self._verdicts)
        stats["active_stragglers"] = len(self._active)
        stats["aligned_ranks"] = len(self._epoch)
        stats["work_tolerance"] = self._config.work_tolerance
        stats["persist_windows"] = self._config.persist_windows
        return stats


__all__ = [
    "MAX_PENDING_WINDOWS",
    "MAX_VERDICTS",
    "WINDOW_GRACE",
    "VERDICT_RECOVERED",
    "VERDICT_STRAGGLER",
    "VERDICT_UNCERTAIN",
    "StragglerDetector",
    "Verdict",
]
