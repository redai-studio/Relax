# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression: insufficient evidence must never report a recovery.

The judge used to reset a rank's streak whenever it was not judged slow —
which included windows that could not judge at all (the rank's own workload
incomparable, effective comparable class too small, or the stage below the
absolute floor). One unusable window therefore cleared an active alert with a
"within_tolerance" recovery it never earned, and the same window could emit
both an uncertain verdict and a recovery for the same rank.

Recovery requires positive evidence back within tolerance. Unusable windows
hold the active flag but break onset persistence. The window index prevents
missing observations from joining separate streaks.
"""

from typing import Any, Dict

import pytest

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import (
    REASON_BELOW_ABSOLUTE_FLOOR,
    REASON_RECOVERY_EVIDENCE_INSUFFICIENT,
    REASON_WITHIN_TOLERANCE,
    REASON_WORKLOAD_INCOMPARABLE,
    VERDICT_RECOVERED,
    VERDICT_STRAGGLER,
    VERDICT_UNCERTAIN,
    StragglerDetector,
)


STAGE = "forward-compute"
COHORT = "topo0:dense:0:0:-1:-1:0:0:0"


class FakeEnvelope:
    def __init__(self, rank: int, host_ms: float, workload: Dict[str, Any], host_start: float) -> None:
        self.cohort = COHORT
        self.name = STAGE
        self.rank = rank
        self.label = f"rank{rank}/tp0/pp0"
        self.host_ms = host_ms
        self.device_ms = None
        self.host_start = host_start
        self.world_size = 4
        self.workload = workload


def make_detector(**overrides: Any) -> StragglerDetector:
    settings: Dict[str, Any] = {
        "enabled": True,
        "window_seconds": 1.0,
        "warmup_windows": 0,
        "work_tolerance": 0.05,
        "persist_windows": 1,
        "min_stage_ms": 5.0,
        "min_cohort_size": 2,
    }
    settings.update(overrides)
    return StragglerDetector(StragglerConfig(**settings))


def feed(detector: StragglerDetector, host_ms: Dict[int, float], workload: Dict[int, Dict[str, Any]]):
    """Feed exactly ONE window (4 ranks) and judge it via flush().

    Per-window control matters for stateful sequences: verdicts for a window
    normally appear only when the NEXT window's samples arrive, so feeding a
    sequence and flushing after each window gives deterministic per-window
    verdicts without leftover samples from an internal retry loop.
    """
    verdicts = []
    index = detector.stats()["windows_closed"] + detector.stats()["open_windows"]
    for rank, host in sorted(host_ms.items()):
        verdicts.extend(detector.observe(FakeEnvelope(rank, host, workload[rank], index + 0.1)))
    verdicts.extend(detector.flush())
    return verdicts


def equal_work():
    return {rank: {"tokens": 1000} for rank in range(4)}


def test_incomparable_window_does_not_recover_a_flagged_rank() -> None:
    detector = make_detector()
    # Window 1: rank 3 slow at equal work -> flagged.
    v1 = feed(detector, {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}, equal_work())
    assert any(v.kind == VERDICT_STRAGGLER and v.rank == 3 for v in v1)
    assert detector.active_stragglers()

    # Window 2: rank 3 still 2x slow, but its own workload became
    # incomparable with every peer. Pre-fix: streak reset -> "recovered".
    v2 = feed(
        detector,
        {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0},
        {0: {"tokens": 1000}, 1: {"tokens": 1000}, 2: {"tokens": 1000}, 3: {"tokens": 5000}},
    )
    assert not any(v.kind == VERDICT_RECOVERED for v in v2), "unobservable window cleared the alert"
    rank3 = [v for v in v2 if v.rank == 3]
    assert rank3 and all(v.kind == VERDICT_UNCERTAIN for v in rank3)
    assert all(v.reason == REASON_WORKLOAD_INCOMPARABLE for v in rank3)
    assert any(a["rank"] == 3 for a in detector.active_stragglers()), "active flag was dropped"

    # Window 3: evidence returns, rank 3 still slow -> the alert must still be
    # live (active flag held through the unusable window), no fresh onset count.
    v3 = feed(detector, {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}, equal_work())
    assert not any(v.kind == VERDICT_RECOVERED for v in v3)
    assert any(a["rank"] == 3 for a in detector.active_stragglers())

    # Window 4: evidence returns, rank 3 genuinely fast -> NOW it recovers.
    v4 = feed(detector, {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0}, equal_work())
    recovered = [v for v in v4 if v.kind == VERDICT_RECOVERED and v.rank == 3]
    assert recovered and recovered[0].reason == REASON_WITHIN_TOLERANCE
    assert not detector.active_stragglers()


def test_small_effective_class_does_not_recover_a_flagged_rank() -> None:
    detector = make_detector()
    v1 = feed(detector, {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}, equal_work())
    assert any(v.kind == VERDICT_STRAGGLER and v.rank == 3 for v in v1)

    # Window 2: every peer's workload shifts so rank 3 has no comparable peer
    # (its own class is just itself -> below the minimum of 2). The verdict is
    # workload_incomparable (the branch chain fires for a slow rank) and the
    # flag is held, counted as withheld recovery evidence.
    v2 = feed(
        detector,
        {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0},
        {0: {"tokens": 9000}, 1: {"tokens": 9000}, 2: {"tokens": 9000}, 3: {"tokens": 1000}},
    )
    assert not any(v.kind == VERDICT_RECOVERED for v in v2), "small reference class cleared the alert"
    assert any(a["rank"] == 3 for a in detector.active_stragglers())
    assert detector.stats()["recovery_evidence_withheld"] >= 1


def test_not_slow_with_unusable_evidence_reports_unknown_not_recovered() -> None:
    """The branch the old code could not reach: rank NOT slow this window but
    the evidence is unusable -- the pre-fix code emitted a recovery."""
    detector = make_detector()
    v1 = feed(detector, {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}, equal_work())
    assert any(v.kind == VERDICT_STRAGGLER and v.rank == 3 for v in v1)

    # Window 2: rank 3's TIMING is back at par but its workload is now
    # incomparable, so within-tolerance cannot be established either.
    v2 = feed(
        detector,
        {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0},
        {0: {"tokens": 1000}, 1: {"tokens": 1000}, 2: {"tokens": 1000}, 3: {"tokens": 9000}},
    )
    assert not any(v.kind == VERDICT_RECOVERED for v in v2), "parity without evidence was read as recovery"
    rank3 = [v for v in v2 if v.rank == 3]
    assert rank3 and all(v.reason == REASON_RECOVERY_EVIDENCE_INSUFFICIENT for v in rank3)
    assert all(v.kind == VERDICT_UNCERTAIN for v in rank3)
    assert any(a["rank"] == 3 for a in detector.active_stragglers())

    # Window 3: workload comparable again and timing at par -> real recovery.
    v3 = feed(detector, {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0}, equal_work())
    assert any(v.kind == VERDICT_RECOVERED and v.rank == 3 for v in v3)
    assert not detector.active_stragglers()


def test_counter_reports_withheld_recoveries() -> None:
    detector = make_detector()
    feed(detector, {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}, equal_work())
    feed(
        detector,
        {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0},
        {0: {"tokens": 1000}, 1: {"tokens": 1000}, 2: {"tokens": 1000}, 3: {"tokens": 9000}},
    )
    stats = detector.stats()
    assert stats["recovery_evidence_withheld"] >= 1
    assert stats["recoveries_reported"] == 0  # the onset only; no fake recovery


def test_below_floor_slow_window_holds_alert_without_recovery() -> None:
    detector = make_detector()
    feed(detector, {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}, equal_work())

    verdicts = feed(detector, {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0}, equal_work())
    rank3 = [v for v in verdicts if v.rank == 3]
    assert [(v.kind, v.reason) for v in rank3] == [(VERDICT_UNCERTAIN, REASON_BELOW_ABSOLUTE_FLOOR)]
    assert any(a["rank"] == 3 for a in detector.active_stragglers())
    assert detector.stats()["recoveries_reported"] == 0
    assert detector.stats()["recovery_evidence_withheld"] == 1

    # The floor suppresses noisy convictions, not measured parity.
    recovered = feed(detector, {rank: 1.0 for rank in range(4)}, equal_work())
    assert [(v.kind, v.rank) for v in recovered] == [(VERDICT_RECOVERED, 3)]
    assert not detector.active_stragglers()


@pytest.mark.parametrize("gap", ["incomparable", "effective_class", "small_cohort", "missing_rank", "empty"])
def test_onset_requires_adjacent_usable_windows(gap: str) -> None:
    detector = make_detector(persist_windows=2, min_cohort_size=3)
    slow = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
    assert not any(v.kind == VERDICT_STRAGGLER for v in feed(detector, slow, equal_work()))

    if gap == "incomparable":
        work = equal_work()
        work[3] = {"tokens": 5000}
        feed(detector, slow, work)
    elif gap == "effective_class":
        feed(detector, slow, {rank: {"tokens": 1000 if rank >= 2 else 5000} for rank in range(4)})
    elif gap == "small_cohort":
        feed(detector, {0: 100.0, 3: 200.0}, equal_work())
    elif gap == "missing_rank":
        feed(detector, {0: 100.0, 1: 100.0, 2: 100.0}, equal_work())

    # Explicit timestamps also test an entirely unobserved window.
    verdicts = []
    for rank, host in slow.items():
        verdicts.extend(detector.observe(FakeEnvelope(rank, host, equal_work()[rank], 2.1)))
    verdicts.extend(detector.flush())
    assert not any(v.kind == VERDICT_STRAGGLER for v in verdicts), gap
    assert not detector.active_stragglers()

    verdicts = []
    for rank, host in slow.items():
        verdicts.extend(detector.observe(FakeEnvelope(rank, host, equal_work()[rank], 3.1)))
    verdicts.extend(detector.flush())
    onset = [v for v in verdicts if v.kind == VERDICT_STRAGGLER]
    assert [(v.rank, v.consecutive_windows) for v in onset] == [(3, 2)]


@pytest.mark.parametrize("persist", [1, 2, 3])
def test_active_alert_survives_gap_without_duplicate_onset(persist: int) -> None:
    detector = make_detector(persist_windows=persist)
    slow = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
    for _ in range(persist):
        feed(detector, slow, equal_work())
    assert detector.stats()["stragglers_reported"] == 1
    work = equal_work()
    work[3] = {"tokens": 5000}
    feed(detector, slow, work)
    for _ in range(persist + 1):
        assert not any(v.kind != VERDICT_UNCERTAIN for v in feed(detector, slow, equal_work()))
    assert detector.stats()["stragglers_reported"] == 1
    assert any(a["rank"] == 3 for a in detector.active_stragglers())


def test_withheld_recovery_counter_counts_one_rank_window_once() -> None:
    detector = make_detector()
    feed(detector, {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}, equal_work())
    work = equal_work()
    work[3] = {"tokens": 5000}
    feed(detector, {rank: 100.0 for rank in range(4)}, work)
    assert detector.stats()["recovery_evidence_withheld"] == 1


def test_active_cap_eviction_and_uncertainty_do_not_repeat_onset(monkeypatch: pytest.MonkeyPatch) -> None:
    from relax.utils.straggler import detector as detector_module

    monkeypatch.setattr(detector_module, "MAX_ACTIVE_ENTRIES", 1)
    detector = make_detector()
    slow = {0: 100.0, 1: 100.0, 2: 200.0, 3: 200.0}
    feed(detector, slow, equal_work())
    assert detector.stats()["stragglers_reported"] == 2
    assert detector.stats()["active_evictions"] == 1
    feed(detector, slow, {rank: {"tokens": 1000 if rank < 2 else 1000 * (rank + 1)} for rank in range(4)})
    verdicts = feed(detector, slow, equal_work())
    assert not any(v.kind == VERDICT_STRAGGLER for v in verdicts)
    assert detector.stats()["stragglers_reported"] == 2
    assert detector.stats()["recoveries_reported"] == 0

    # A measured recovery retires both onset markers, even for the evicted key.
    feed(detector, {rank: 100.0 for rank in range(4)}, equal_work())
    assert detector.stats()["streak_entries"] == 0
    verdicts = feed(detector, slow, equal_work())
    assert len([v for v in verdicts if v.kind == VERDICT_STRAGGLER]) == 2
