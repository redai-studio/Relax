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

Host and device intervals are compared separately, and the measurement
classification says which of the two moved:

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

Measured facts and inferred causes are kept strictly apart. Everything a verdict
reports under ``facts`` was observed or counted; the only forward statement it
makes about *why* something happened is ``candidate_causes``, which always
starts at ``("undetermined",)`` and is populated only from evidence the facts
actually support. A reviewer must be able to read a number as a measurement and
never as a guess, so the detector refuses three tempting inferences even when
they look obvious:

* a large interval on a communication-named stage is **not** by itself a network
  or fault cause — the stage name is a label, not evidence, and a slow peer can
  hold up every rank at a collective;
* a late-arriving rank is **not** by itself a host/CPU cause — lateness has no
  direction and no owner until it is correlated with another clock;
* two moved CUDA-event intervals do **not** separate a slow kernel from an idle
  stream, so "slow GPU" is never emitted.

Those conclusions need data this profiler does not collect (peer-to-peer
transfer times, per-kernel durations), so the verdict reports the observable
gap and stops.

Workload is reported, never corrected. When an envelope carries a ``workload``
mapping the detector records the relative difference between the rank's workload
and its peers' median in ``facts["workload_delta"]``; a rank that legitimately
did more tokens/sequences/microbatches therefore shows it next to the timing gap
instead of silently reading as a straggler. C2 is a *reporter*: it does not
normalise timings by workload and does not suppress a verdict because a workload
difference exists.
"""

import json
import math
import statistics
from collections import deque
from dataclasses import dataclass, field
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

#: Bound on ``(cohort, stage, rank)`` persistence trackers; a long run must not
#: grow the streak map without limit.
MAX_STREAK_ENTRIES = 4096

#: Bound on cached rank labels; labels repeat, so a modest cache suffices.
MAX_LABEL_ENTRIES = 4096

#: Bound on per-rank window alignment epochs; malformed envelopes could
#: otherwise invent a new rank on every observation.
MAX_ALIGNED_RANKS = 4096

#: Bound on per-cohort window anchors; malformed envelopes could otherwise
#: invent a new cohort string (and therefore a new anchor) on every observation.
MAX_COHORT_ANCHORS = 1024

#: Bound on concurrently flagged ``(cohort, stage, rank)`` triples.
MAX_ACTIVE_ENTRIES = 512

#: Bound on distinct ``(cohort, stage)`` pairs retained per open window.
MAX_PAIRS_PER_WINDOW = 256

#: Bound on distinct ranks retained per ``(cohort, stage)`` in one window.
MAX_RANKS_PER_WINDOW = 1024

#: Bound on intervals retained per rank per pair in one window: a median needs a
#: handful, not thousands, and the window must stay bounded.
MAX_SAMPLES_PER_RANK = 32

VERDICT_STRAGGLER = "straggler"
VERDICT_RECOVERED = "recovered"
VERDICT_UNCERTAIN = "uncertain"

#: Measurement classification values (``Verdict.reason``); they describe *what
#: was measured*, never why.
REASON_HOST_ONLY_STALL = "host_only_stall"
REASON_GPU_STREAM_STALL = "gpu_stream_stall"
REASON_ATTRIBUTION_UNKNOWN = "attribution_unknown"
REASON_WITHIN_TOLERANCE = "within_tolerance"
REASON_COHORT_BELOW_MIN_SIZE = "cohort_below_min_size"

#: Which clocks produced a verdict.
MEASUREMENT_HOST_ONLY = "host_only"
MEASUREMENT_DEVICE_AND_HOST = "device_and_host"
MEASUREMENT_UNKNOWN = "unknown"

#: The only cause a verdict asserts without supporting evidence.
CANDIDATE_UNDETERMINED = "undetermined"
#: Host interval grew while the device interval did not.
CANDIDATE_HOST_SIDE_DELAY = "host_side_delay_possible"
#: Both intervals grew; CUDA events cannot distinguish a slow kernel from an
#: idle stream, hence the careful wording.
CANDIDATE_DEVICE_OR_STREAM_DELAY = "device_or_stream_visible_delay_possible"


def _median(values: List[float]) -> float:
    """Median of a non-empty list."""
    return float(statistics.median(values))


def _relative_deviation(value: float, reference: float) -> float:
    """Signed slowdown of ``value`` against ``reference``, 0 when
    unmeasurable."""
    if reference <= 0.0:
        return 0.0
    return (value - reference) / reference


def _ratio(value: Optional[float], reference: Optional[float]) -> Optional[float]:
    """Ratio ``value / reference``, ``None`` when it cannot be measured."""
    if value is None or reference is None or reference <= 0.0:
        return None
    return value / reference


def _workload_total(workload: Any) -> Optional[float]:
    """Total a workload mapping down to one comparable magnitude.

    Only finite numeric entries count; a mapping without any finite numeric
    entry is treated as "no workload reported", and a ``NaN``/``inf`` counter
    cannot poison the peer median the way it could poison a timing reference.
    """
    if not isinstance(workload, dict):
        return None
    total = 0.0
    seen = False
    for value in workload.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                continue
            total += number
            seen = True
    return total if seen else None


def _expected_cohort_size(envelope: Any) -> Optional[int]:
    """Expected cohort size reported by an envelope, when it knows one.

    Prefers an explicit cohort-size attribute. ``world_size`` is the global
    process count, so it is only a fallback: on a sharded run it can overstate
    the cohort, which makes ``coverage_ratio`` conservative (never flattering)
    rather than wrong in the dangerous direction.
    """
    for attribute in ("cohort_world_size", "data_parallel_world_size", "cohort_size"):
        value = getattr(envelope, attribute, None)
        if value is None or isinstance(value, bool):
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    value = getattr(envelope, "world_size", None)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _candidate_causes(host_grew: bool, device_available: bool, device_grew: bool) -> Tuple[str, ...]:
    """Return the causes the measured facts support, most conservative first.

    The default is ``("undetermined",)`` and it is only ever replaced by a
    statement the facts can carry:

    * no host growth -> nothing to explain, ``undetermined``;
    * no device timing -> the host/device split cannot be made, so the extra
      time cannot be attributed to either side;
    * host grew, device did not -> ``host_side_delay_possible``;
    * host and device both grew -> ``device_or_stream_visible_delay_possible``.

    A stage *name* never contributes: nothing here inspects the stage, so a
    communication-named stage cannot produce a network/fault cause, and a late
    rank cannot produce a host/CPU cause from lateness alone.
    """
    if not host_grew or not device_available:
        return (CANDIDATE_UNDETERMINED,)
    if device_grew:
        return (CANDIDATE_DEVICE_OR_STREAM_DELAY,)
    return (CANDIDATE_HOST_SIDE_DELAY,)


def _measurement_kind(rank_device: Optional[float], reference_device: Optional[float], cohort_size: int) -> str:
    """Describe which clocks produced a verdict."""
    if cohort_size < 2:
        return MEASUREMENT_UNKNOWN
    if rank_device is not None and reference_device is not None:
        return MEASUREMENT_DEVICE_AND_HOST
    if rank_device is None and reference_device is None:
        return MEASUREMENT_HOST_ONLY
    # One side has CUDA events and the other does not: neither a clean host-only
    # nor a clean host+device comparison.
    return MEASUREMENT_UNKNOWN


def _evict_to_cap(mapping: Dict[Any, Any], cap: int, counters: Dict[str, int], counter_name: str) -> None:
    """Drop the oldest entries until one more key fits, counting each
    eviction."""
    while len(mapping) >= cap:
        oldest = next(iter(mapping), None)
        if oldest is None:
            return
        mapping.pop(oldest, None)
        counters[counter_name] += 1


def _evict_set_to_cap(active: Set[Any], cap: int, counters: Dict[str, int], counter_name: str) -> None:
    """Drop entries from a set until one more fits, counting each eviction."""
    while len(active) >= cap:
        if not active:
            return
        active.pop()
        counters[counter_name] += 1


@dataclass(frozen=True)
class Verdict:
    """One judgement about one rank, one stage, one window.

    The verdict keeps *measurements* and *possible causes* in separate fields so
    a reader can trust either one independently.

    Attributes:
        kind: ``straggler``/``recovered``/``uncertain``.
        cohort, name, rank, label, window_index: which comparison this is.
        deviation: relative host slowdown against the fastest peer.
        consecutive_windows: how many consecutive windows the rank was slow.
        rank_host_ms, reference_host_ms: the two host medians behind the verdict.
        rank_device_ms, reference_device_ms: device medians, ``None`` without
            CUDA events.
        host_only: measurement classification; the host moved and the device did
            not.
        cohort_size: ranks that reported for this pair in the window.
        reason: **pure measurement classification**, one of
            ``host_only_stall``, ``gpu_stream_stall``, ``attribution_unknown``,
            ``within_tolerance`` or ``cohort_below_min_size``. It never names a
            cause; ``candidate_causes`` is the only place a cause appears.
        facts: everything measured or counted (rank, cohort, stage, window,
            sample counts, observed/peer_fastest/peer_median host medians, ratio,
            absolute delta, tolerance, persistence, cohort size/expected,
            coverage, device medians and availability, workload delta). Flat
            JSON-friendly values only.
        candidate_causes: the only forward statement the verdict makes; starts
            at ``("undetermined",)`` and grows only from the facts.
        measurement_kind: ``host_only``, ``device_and_host`` or ``unknown`` —
            which clocks produced this verdict.
    """

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
    facts: Dict[str, Any] = field(default_factory=dict)
    candidate_causes: Tuple[str, ...] = (CANDIDATE_UNDETERMINED,)
    measurement_kind: str = MEASUREMENT_UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat JSON-friendly mapping.

        New callers should read ``facts`` for numbers and ``candidate_causes``
        for causes; the legacy timing fields are kept for existing consumers.
        """
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
            "measurement_kind": self.measurement_kind,
            "candidate_causes": list(self.candidate_causes),
            "facts": dict(self.facts),
        }

    def to_json(self) -> str:
        """Serialise as one JSONL line."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def _cause_label(self) -> str:
        """Render the cause field, explicitly marked as an inference."""
        causes = ", ".join(self.candidate_causes) if self.candidate_causes else CANDIDATE_UNDETERMINED
        if self.candidate_causes == (CANDIDATE_UNDETERMINED,):
            return f"cause: {causes}"
        return f"candidate causes: {causes}"

    def describe(self) -> str:
        """One human-readable line: facts first, causes last."""
        facts = self.facts

        def millis(name: str) -> str:
            value = facts.get(name)
            return "n/a" if value is None else f"{value:.2f}"

        def number(name: str) -> str:
            value = facts.get(name)
            return "n/a" if value is None else f"{value:.3f}"

        return (
            f"{self.kind}: {self.label} stage={self.name} window={self.window_index} "
            f"facts(observed_ms={millis('observed_ms')} peer_fastest_ms={millis('peer_fastest_ms')} "
            f"peer_median_ms={millis('peer_median_ms')} ratio={number('ratio')} "
            f"delta_ms={millis('absolute_delta_ms')} samples={facts.get('samples_rank')}/"
            f"{facts.get('samples_peers_min')} cohort={self.cohort_size}/{facts.get('cohort_expected')} "
            f"coverage={number('coverage_ratio')} device_available={facts.get('device_available')} "
            f"workload_delta={number('workload_delta')} measurement_kind={self.measurement_kind}) "
            f"reason(measurement)={self.reason} {self._cause_label()}"
        )


class _Window:
    """Per-window samples: ``(cohort, name) -> rank -> [host, device,
    workload]``.

    Every nested mapping is capped; an evicted sample is counted in the shared
    detector counters instead of being silently dropped.
    """

    __slots__ = ("index", "samples", "labels", "world_size", "cohort_expected", "warmup_samples", "_counters")

    def __init__(self, index: int, counters: Dict[str, int]) -> None:
        self.index = index
        self.samples: Dict[
            Tuple[str, str], Dict[int, List[Tuple[float, Optional[float], Optional[Dict[str, Any]]]]]
        ] = {}
        self.labels: Dict[int, str] = {}
        self.world_size = 1
        self.cohort_expected: Optional[int] = None
        #: Samples excluded by per-rank warmup. Kept so a window that ends up
        #: with no judged sample is still reported as a warmup window.
        self.warmup_samples = 0
        self._counters = counters

    def add(self, envelope: Any) -> None:
        """Record one envelope, respecting every per-window cap."""
        self.world_size = max(self.world_size, int(getattr(envelope, "world_size", 1) or 1))
        expected = _expected_cohort_size(envelope)
        if expected is not None:
            # Largest reported expectation: coverage can only be understated.
            self.cohort_expected = expected if self.cohort_expected is None else max(self.cohort_expected, expected)

        # Sanitise before any structure is touched, so a dropped sample can
        # never leave an empty rank bucket behind (which used to crash
        # ``_median`` for the whole window).
        host_ms = float(envelope.host_ms)
        if not math.isfinite(host_ms) or host_ms < 0.0:
            # A NaN/inf/negative host duration is a malformed measurement, not a
            # fast rank. Keeping it would let one hostile (or backwards-clock)
            # packet become the ``min`` reference and silently suppress every
            # verdict in the window; it is dropped and counted instead.
            self._counters["invalid_samples"] += 1
            return
        device_ms = envelope.device_ms
        if device_ms is not None:
            device_ms = float(device_ms)
            if not math.isfinite(device_ms) or device_ms < 0.0:
                self._counters["invalid_device_samples"] += 1
                device_ms = None

        key = (envelope.cohort, envelope.name)
        per_rank = self.samples.get(key)
        if per_rank is None:
            if len(self.samples) >= MAX_PAIRS_PER_WINDOW:
                self._counters["pair_evictions"] += 1
                return
            per_rank = {}
            self.samples[key] = per_rank

        rank = int(envelope.rank)
        bucket = per_rank.get(rank)
        if bucket is None:
            if len(per_rank) >= MAX_RANKS_PER_WINDOW:
                self._counters["rank_evictions"] += 1
                return
            bucket = []
            per_rank[rank] = bucket
        if len(bucket) >= MAX_SAMPLES_PER_RANK:
            self._counters["sample_evictions"] += 1
            return

        workload = getattr(envelope, "workload", None)
        bucket.append(
            (
                host_ms,
                device_ms,
                dict(workload) if isinstance(workload, dict) else None,
            )
        )
        if rank not in self.labels:
            if len(self.labels) >= MAX_RANKS_PER_WINDOW:
                self._counters["label_evictions"] += 1
            else:
                self.labels[rank] = getattr(envelope, "label", "") or f"rank{rank}"


class StragglerDetector:
    """Accumulates envelopes into windows and reports persistent outliers.

    Not internally synchronised: ``observe``/``flush``/``stats``/
    ``active_stragglers`` mutate or read the same window/streak/active
    structures, and a reader racing a writer can drop a whole window's verdicts
    (the empty-bucket ``StatisticsError``). The only production caller is
    :class:`~relax.utils.straggler.collector.TimingCollector`, which serialises
    every one of those calls under its ``_state_lock``; a second lock here would
    only duplicate that. Direct callers must provide the same serialisation.
    """

    def __init__(self, config: StragglerConfig) -> None:
        self._config = config
        self._windows: Dict[int, _Window] = {}
        self._verdicts: Deque[Verdict] = deque()
        self._streak: Dict[Tuple[str, str, int], int] = {}
        self._epoch: Dict[int, float] = {}
        self._cohort_epoch: Dict[str, float] = {}
        self._active: Set[Tuple[str, str, int]] = set()
        self._labels: Dict[Tuple[str, str, int], str] = {}
        self._counters: Dict[str, int] = {
            "envelopes": 0,
            "windows_closed": 0,
            "windows_forced": 0,
            "warmup_windows_skipped": 0,
            "warmup_samples_skipped": 0,
            "incomplete_windows": 0,
            "cohort_stage_pairs": 0,
            "uncertain_judgements": 0,
            "single_rank_windows": 0,
            "stragglers_reported": 0,
            "recoveries_reported": 0,
            # Evictions from the bounded structures below; every cap reports.
            "verdict_evictions": 0,
            "streak_evictions": 0,
            "label_evictions": 0,
            "epoch_evictions": 0,
            "cohort_epoch_evictions": 0,
            "active_evictions": 0,
            "pair_evictions": 0,
            "rank_evictions": 0,
            "sample_evictions": 0,
            "invalid_samples": 0,
            "invalid_device_samples": 0,
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
            if not math.isfinite(host_start):
                # A non-finite clock would poison both the cohort anchor and the
                # warmup baseline; it is a malformed packet, not a fast rank.
                self._counters["invalid_samples"] += 1
                return []
            rank = int(envelope.rank)
            cohort = str(envelope.cohort)
            first = self._epoch.get(rank)
            if first is None:
                # Evicting an alignment epoch re-bases that rank's *warmup* on
                # its next observation; the eviction is counted so the reset is
                # visible instead of silently changing window membership.
                _evict_to_cap(self._epoch, MAX_ALIGNED_RANKS, self._counters, "epoch_evictions")
                first = host_start
                self._epoch[rank] = first
            anchor = self._cohort_epoch.get(cohort)
            if anchor is None:
                # Windows are anchored to the cohort's first observation, not to
                # each rank's own: a rank that starts later then lands in the
                # same wall-clock window as its peers instead of falling a fixed
                # number of windows behind them and never being compared. The
                # per-rank ``_epoch`` stays as the warmup baseline.
                _evict_to_cap(self._cohort_epoch, MAX_COHORT_ANCHORS, self._counters, "cohort_epoch_evictions")
                anchor = host_start
                self._cohort_epoch[cohort] = anchor
            # Integer microseconds: `(4.1 - 0.1) / 1.0` floors to 3 in binary
            # floating point, which would silently merge two windows.
            window_us = max(1, int(round(self._config.window_seconds * 1e6)))
            index = int(round(max(0.0, host_start - anchor) * 1e6)) // window_us
            window = self._windows.get(index)
            if window is None:
                window = _Window(index, self._counters)
                self._windows[index] = window
            warmup_span = self._config.warmup_windows * self._config.window_seconds
            if warmup_span > 0.0 and (host_start - first) < warmup_span:
                # Startup is not a straggler, and it is per rank: lazy CUDA
                # allocation and the first data batch make a late-starting rank
                # look momentarily slow. The window is created (so the skip
                # stays visible in the stats) but no sample is recorded.
                window.warmup_samples += 1
                self._counters["warmup_samples_skipped"] += 1
            else:
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
        if not window.samples:
            # Warmup samples (and malformed ones) never reached a rank bucket, so
            # there is nothing to judge. Counted, not hidden.
            if window.warmup_samples:
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
        per_rank: Dict[int, List[Tuple[float, Optional[float], Optional[Dict[str, Any]]]]],
    ) -> List[Verdict]:
        """Compare each rank of one (cohort, stage) pair against its fastest
        peer."""
        self._counters["cohort_stage_pairs"] += 1
        ranks = sorted(per_rank)
        host_medians = {rank: _median([host for host, _, _ in per_rank[rank]]) for rank in ranks}
        sample_counts = {rank: len(per_rank[rank]) for rank in ranks}
        device_medians: Dict[int, Optional[float]] = {}
        workload_totals: Dict[int, Optional[float]] = {}
        for rank in ranks:
            devices = [device for _, device, _ in per_rank[rank] if device is not None]
            device_medians[rank] = _median(devices) if devices else None
            totals = [total for _, _, workload in per_rank[rank] if (total := _workload_total(workload)) is not None]
            workload_totals[rank] = totals[-1] if totals else None

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

        host_reference = min(host_medians.values())
        host_median = _median(list(host_medians.values()))
        device_values = [value for value in device_medians.values() if value is not None]
        device_reference = min(device_values) if device_values else None

        def build(
            rank: int,
            kind: str,
            deviation: float,
            streak: int,
            host_only: bool,
            reason: str,
            label: str,
        ) -> Verdict:
            """Build one verdict whose facts are all measured or counted."""
            rank_device = device_medians[rank]
            device_available = rank_device is not None and device_reference is not None
            device_deviation = (
                _relative_deviation(rank_device, device_reference)
                if device_available and rank_device is not None and device_reference is not None
                else 0.0
            )
            host_grew = deviation > self._config.work_tolerance
            device_grew = device_available and device_deviation > self._config.work_tolerance
            peer_counts = {other: sample_counts[other] for other in ranks if other != rank}
            peer_workloads = [
                workload_totals[other] for other in ranks if other != rank and workload_totals[other] is not None
            ]
            rank_workload = workload_totals[rank]
            peer_workload_median: Optional[float] = None
            workload_delta: Optional[float] = None
            if rank_workload is not None and peer_workloads:
                peer_workload_median = float(statistics.median(peer_workloads))
                if peer_workload_median > 0.0:
                    workload_delta = (rank_workload - peer_workload_median) / peer_workload_median
            facts: Dict[str, Any] = {
                "rank": rank,
                "cohort": cohort,
                "stage": name,
                "window_index": window.index,
                "samples_rank": sample_counts[rank],
                "samples_peers": peer_counts,
                "samples_peers_min": min(peer_counts.values()) if peer_counts else None,
                "observed_ms": host_medians[rank],
                "peer_fastest_ms": host_reference,
                "peer_median_ms": host_median,
                "ratio": _ratio(host_medians[rank], host_reference),
                "absolute_delta_ms": host_medians[rank] - host_reference,
                "work_tolerance": self._config.work_tolerance,
                "persistence": streak,
                "cohort_size": cohort_size,
                "cohort_expected": window.cohort_expected,
                "coverage_ratio": (cohort_size / window.cohort_expected if window.cohort_expected else None),
                "device_ms": rank_device,
                "peer_device_ms": device_reference,
                "device_available": device_available,
                "device_ratio": _ratio(rank_device, device_reference) if device_available else None,
                # Workload is reported next to the timing gap, never divided out:
                # a genuine +100% workload looks exactly like a +100% straggler
                # until this number is read.
                "workload_delta": workload_delta,
                "workload_rank": rank_workload,
                "workload_peer_median": peer_workload_median,
                "workload_delta_beyond_tolerance": (
                    None if workload_delta is None else workload_delta > self._config.work_tolerance
                ),
            }
            return self._verdict(
                kind=kind,
                cohort=cohort,
                name=name,
                rank=rank,
                label=label,
                window_index=window.index,
                deviation=deviation,
                consecutive_windows=streak,
                rank_host_ms=host_medians[rank],
                reference_host_ms=host_reference,
                rank_device_ms=rank_device,
                reference_device_ms=device_reference,
                host_only=host_only,
                cohort_size=cohort_size,
                reason=reason,
                facts=facts,
                candidate_causes=_candidate_causes(host_grew, device_available, device_grew),
                measurement_kind=_measurement_kind(rank_device, device_reference, cohort_size),
            )

        if cohort_size < self._config.min_cohort_size:
            self._counters["uncertain_judgements"] += 1
            return [
                build(
                    rank,
                    VERDICT_UNCERTAIN,
                    0.0,
                    0,
                    False,
                    REASON_COHORT_BELOW_MIN_SIZE,
                    window.labels.get(rank, f"rank{rank}"),
                )
                for rank in ranks
            ]

        verdicts: List[Verdict] = []
        for rank in ranks:
            deviation = _relative_deviation(host_medians[rank], host_reference)
            key = (cohort, name, rank)
            label = window.labels.get(rank, f"rank{rank}")
            self._cache_label(key, label)
            slow = deviation > self._config.work_tolerance
            streak = self._streak.get(key, 0) + 1 if slow else 0
            self._set_streak(key, streak)
            rank_device = device_medians[rank]
            stream_visible = rank_device is not None and device_reference is not None
            device_deviation = (
                _relative_deviation(rank_device, device_reference)
                if stream_visible and rank_device is not None and device_reference is not None
                else 0.0
            )
            host_only = slow and stream_visible and device_deviation <= self._config.work_tolerance
            if streak >= self._config.persist_windows and key not in self._active:
                _evict_set_to_cap(self._active, MAX_ACTIVE_ENTRIES, self._counters, "active_evictions")
                self._active.add(key)
                self._counters["stragglers_reported"] += 1
                reason = (
                    REASON_HOST_ONLY_STALL
                    if host_only
                    else (REASON_GPU_STREAM_STALL if stream_visible else REASON_ATTRIBUTION_UNKNOWN)
                )
                verdicts.append(build(rank, VERDICT_STRAGGLER, deviation, streak, host_only, reason, label))
            elif streak == 0 and key in self._active:
                self._active.discard(key)
                self._counters["recoveries_reported"] += 1
                verdicts.append(
                    build(rank, VERDICT_RECOVERED, deviation, streak, False, REASON_WITHIN_TOLERANCE, label)
                )
        return verdicts

    def _cache_label(self, key: Tuple[str, str, int], label: str) -> None:
        """Remember a label under the label cap."""
        if key not in self._labels:
            _evict_to_cap(self._labels, MAX_LABEL_ENTRIES, self._counters, "label_evictions")
        self._labels[key] = label

    def _set_streak(self, key: Tuple[str, str, int], streak: int) -> None:
        """Update a persistence counter under the streak cap.

        A zero streak is removed rather than stored, so the map holds only
        ranks that are currently slow.
        """
        if streak == 0:
            self._streak.pop(key, None)
            return
        if key not in self._streak:
            _evict_to_cap(self._streak, MAX_STREAK_ENTRIES, self._counters, "streak_evictions")
        self._streak[key] = streak

    def _verdict(self, **kwargs: Any) -> Verdict:
        """Build and retain a verdict under the verdict cap."""
        verdict = Verdict(**kwargs)
        if len(self._verdicts) >= MAX_VERDICTS:
            self._verdicts.popleft()
            self._counters["verdict_evictions"] += 1
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
        """Return counters plus open-window state and the active caps."""
        stats: Dict[str, Any] = dict(self._counters)
        stats["open_windows"] = len(self._windows)
        stats["retained_verdicts"] = len(self._verdicts)
        stats["active_stragglers"] = len(self._active)
        stats["aligned_ranks"] = len(self._epoch)
        stats["aligned_cohorts"] = len(self._cohort_epoch)
        stats["streak_entries"] = len(self._streak)
        stats["label_entries"] = len(self._labels)
        stats["work_tolerance"] = self._config.work_tolerance
        stats["persist_windows"] = self._config.persist_windows
        stats["caps"] = {
            "MAX_PENDING_WINDOWS": MAX_PENDING_WINDOWS,
            "MAX_VERDICTS": MAX_VERDICTS,
            "MAX_STREAK_ENTRIES": MAX_STREAK_ENTRIES,
            "MAX_LABEL_ENTRIES": MAX_LABEL_ENTRIES,
            "MAX_ALIGNED_RANKS": MAX_ALIGNED_RANKS,
            "MAX_COHORT_ANCHORS": MAX_COHORT_ANCHORS,
            "MAX_ACTIVE_ENTRIES": MAX_ACTIVE_ENTRIES,
            "MAX_PAIRS_PER_WINDOW": MAX_PAIRS_PER_WINDOW,
            "MAX_RANKS_PER_WINDOW": MAX_RANKS_PER_WINDOW,
            "MAX_SAMPLES_PER_RANK": MAX_SAMPLES_PER_RANK,
        }
        return stats


__all__ = [
    "CANDIDATE_DEVICE_OR_STREAM_DELAY",
    "CANDIDATE_HOST_SIDE_DELAY",
    "CANDIDATE_UNDETERMINED",
    "MAX_ACTIVE_ENTRIES",
    "MAX_ALIGNED_RANKS",
    "MAX_COHORT_ANCHORS",
    "MAX_LABEL_ENTRIES",
    "MAX_PAIRS_PER_WINDOW",
    "MAX_PENDING_WINDOWS",
    "MAX_RANKS_PER_WINDOW",
    "MAX_SAMPLES_PER_RANK",
    "MAX_STREAK_ENTRIES",
    "MAX_VERDICTS",
    "MEASUREMENT_DEVICE_AND_HOST",
    "MEASUREMENT_HOST_ONLY",
    "MEASUREMENT_UNKNOWN",
    "REASON_ATTRIBUTION_UNKNOWN",
    "REASON_COHORT_BELOW_MIN_SIZE",
    "REASON_GPU_STREAM_STALL",
    "REASON_HOST_ONLY_STALL",
    "REASON_WITHIN_TOLERANCE",
    "VERDICT_RECOVERED",
    "VERDICT_STRAGGLER",
    "VERDICT_UNCERTAIN",
    "StragglerDetector",
    "Verdict",
    "WINDOW_GRACE",
]
