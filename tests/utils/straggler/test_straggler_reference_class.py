# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression: the speed reference and the workload gate must share one peer set.

The detector judged a rank's timing against the fastest of ALL ranks in the
window while gating workload comparability against the peer-median workload.
The two reference sets diverged whenever workloads differed: one peer doing
half the work is fast because it is under-worked, not because it is healthy,
yet it became the speed baseline for everybody else. Four ranks at
``{500 tokens / 50 ms, 1000 / 100 ms, 1000 / 100 ms, 1000 / 100 ms}`` -- no
injected stall, identical per-token speed -- were reported as three
stragglers at ``ratio=2.0`` with ``workload_comparable=True``.

Each rank is now judged only against the peers whose workload is pairwise
comparable with its own. With no comparable peer the timing verdict is
withheld as ``workload_incomparable`` (uncertain), never silently judged
against the all-ranks fastest. The first test fails against the
all-ranks-fastest reference.
"""

from typing import Any, Dict, List, Optional

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import (
    REASON_WORKLOAD_INCOMPARABLE,
    VERDICT_STRAGGLER,
    VERDICT_UNCERTAIN,
    StragglerDetector,
)


STAGE = "forward-compute"
COHORT = "topo0:dense:0:0:-1:-1:0:0:0"


class FakeEnvelope:
    """Minimal stand-in for the observer's :class:`TimingEnvelope`."""

    def __init__(
        self,
        cohort: str,
        name: str,
        rank: int,
        label: str,
        host_ms: float,
        device_ms: Optional[float],
        host_start: float,
        world_size: int = 4,
        workload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cohort = cohort
        self.name = name
        self.rank = rank
        self.label = label
        self.host_ms = host_ms
        self.device_ms = device_ms
        self.host_start = host_start
        self.world_size = world_size
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


def feed_window(
    detector: StragglerDetector,
    index: int,
    host_ms: Dict[int, float],
    workload: Dict[int, Dict[str, Any]],
) -> List[Any]:
    verdicts: List[Any] = []
    for rank, host in sorted(host_ms.items()):
        envelope = FakeEnvelope(
            cohort=COHORT,
            name=STAGE,
            rank=rank,
            label=f"rank{rank}/tp0/pp0",
            host_ms=host,
            device_ms=None,
            host_start=index + 0.1,
            world_size=4,
            workload=workload.get(rank),
        )
        verdicts.extend(detector.observe(envelope))
    return verdicts


def feed_equal_windows(
    detector: StragglerDetector,
    count: int,
    host_ms: Dict[int, float],
    workload: Dict[int, Dict[str, Any]],
) -> List[Any]:
    verdicts: List[Any] = []
    for index in range(count):
        verdicts.extend(feed_window(detector, index, host_ms, workload))
    return verdicts


def test_under_worked_peer_does_not_flag_normal_ranks() -> None:
    """One peer doing half the work must not become everybody's baseline.

    Ranks 1-3 do the same work at the same speed; rank 0 does half the work in
    half the time. Against the all-ranks fastest (rank 0's 50 ms) every normal
    rank reads as a 2x straggler; against their comparable peers they are
    exactly at par.
    """
    detector = make_detector()
    host_ms = {0: 50.0, 1: 100.0, 2: 100.0, 3: 100.0}
    workload = {
        0: {"tokens": 500},
        1: {"tokens": 1000},
        2: {"tokens": 1000},
        3: {"tokens": 1000},
    }

    verdicts = feed_equal_windows(detector, 4, host_ms, workload)
    verdicts.extend(detector.flush())

    stragglers = [verdict for verdict in verdicts if verdict.kind == VERDICT_STRAGGLER]
    assert stragglers == [], f"normal ranks were flagged: {[(v.rank, v.facts['ratio']) for v in stragglers]}"
    assert detector.stats()["stragglers_reported"] == 0


def test_rank_without_comparable_peers_is_not_judged() -> None:
    """A rank whose workload matches nobody cannot be judged at all.

    Rank 0 (500 tokens) has no peer within tolerance, so its timing verdict is
    withheld as ``workload_incomparable`` rather than convicted against -- or
    silenced by -- the incomparable all-ranks reference.
    """
    detector = make_detector()
    host_ms = {0: 200.0, 1: 100.0, 2: 100.0, 3: 100.0}
    workload = {
        0: {"tokens": 500},
        1: {"tokens": 1000},
        2: {"tokens": 1000},
        3: {"tokens": 1000},
    }

    verdicts = feed_equal_windows(detector, 4, host_ms, workload)
    verdicts.extend(detector.flush())

    for verdict in verdicts:
        if verdict.rank == 0:
            assert verdict.kind == VERDICT_UNCERTAIN
            assert verdict.reason == REASON_WORKLOAD_INCOMPARABLE
    assert detector.stats()["workload_incomparable_windows"] > 0
    assert not [v for v in verdicts if v.kind == VERDICT_STRAGGLER and v.rank == 0]


def test_equal_work_genuine_straggler_is_still_detected() -> None:
    """The reference-class fix must not blind the detector to real stalls."""
    detector = make_detector()
    host_ms = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
    workload = {rank: {"tokens": 1000} for rank in range(4)}

    verdicts = feed_equal_windows(detector, 4, host_ms, workload)
    verdicts.extend(detector.flush())

    stragglers = [verdict for verdict in verdicts if verdict.kind == VERDICT_STRAGGLER]
    assert [verdict.rank for verdict in stragglers] == [3]
    assert stragglers[0].facts["reference_ranks"] == [0, 1, 2, 3]
    assert stragglers[0].facts["workload_comparable"] is True


def test_over_worked_slow_rank_is_incomparable_not_straggler() -> None:
    """A rank slow because it does double work stays ``uncertain``."""
    detector = make_detector()
    host_ms = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
    workload = {
        0: {"tokens": 1000},
        1: {"tokens": 1000},
        2: {"tokens": 1000},
        3: {"tokens": 2000},
    }

    verdicts = feed_equal_windows(detector, 4, host_ms, workload)
    verdicts.extend(detector.flush())

    rank3 = [verdict for verdict in verdicts if verdict.rank == 3]
    assert rank3, "rank 3 produced no verdict at all"
    assert all(verdict.kind == VERDICT_UNCERTAIN for verdict in rank3)
    assert all(verdict.reason == REASON_WORKLOAD_INCOMPARABLE for verdict in rank3)
    assert not [v for v in verdicts if v.kind == VERDICT_STRAGGLER]


def test_facts_expose_the_reference_class() -> None:
    """The verdict must say which peers it was actually judged against."""
    detector = make_detector()
    host_ms = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
    workload = {rank: {"tokens": 1000} for rank in range(4)}

    verdicts = feed_equal_windows(detector, 4, host_ms, workload)
    verdicts.extend(detector.flush())

    straggler = [v for v in verdicts if v.kind == VERDICT_STRAGGLER][0]
    assert straggler.facts["comparable_peers"] == [0, 1, 2]
    assert straggler.facts["peer_fastest_ms"] == 100.0
    assert straggler.facts["workload_peer_tokens"] == 1000.0
