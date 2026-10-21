# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression: min_cohort_size applies to the EFFECTIVE comparable class.

The raw topological gate counts reporting ranks; the effective class of a rank
is the set of peers whose workload is pairwise comparable with its own. When a
window fragments into small comparable classes, a configured minimum cohort of
N must require N *comparable* ranks before a conviction — four ranks split into
two comparable pairs with ``min_cohort_size=4`` must not convict either pair,
while the same data with ``min_cohort_size=2`` still detects a straggler inside
a pair. Facts distinguish raw topological coverage (``coverage_ratio``) from
effective reference coverage (``comparable_coverage_ratio``).
"""

from typing import Any, Dict

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import (
    REASON_COHORT_BELOW_MIN_SIZE,
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


def feed(detector: StragglerDetector, windows: int, host_ms: Dict[int, float], workload: Dict[int, Dict[str, Any]]):
    verdicts = []
    for index in range(windows):
        for rank, host in sorted(host_ms.items()):
            verdicts.extend(detector.observe(FakeEnvelope(rank, host, workload[rank], index + 0.1)))
    verdicts.extend(detector.flush())
    return verdicts


def two_pairs_data():
    """Ranks {0,1} at 1000 tokens, ranks {2,3} at 2000 tokens; rank 3 is
    slow."""
    host_ms = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
    workload = {0: {"tokens": 1000}, 1: {"tokens": 1000}, 2: {"tokens": 2000}, 3: {"tokens": 2000}}
    return host_ms, workload


def test_two_comparable_pairs_with_min_four_do_not_convict() -> None:
    detector = make_detector(min_cohort_size=4)
    host_ms, workload = two_pairs_data()
    verdicts = feed(detector, 3, host_ms, workload)

    stragglers = [v for v in verdicts if v.kind == VERDICT_STRAGGLER]
    assert stragglers == [], f"convicted inside a comparable class of 2 under min_cohort_size=4: {stragglers}"
    slow = [v for v in verdicts if v.rank == 3]
    assert slow and all(v.kind == VERDICT_UNCERTAIN for v in slow)
    assert all(v.reason == REASON_COHORT_BELOW_MIN_SIZE for v in slow)


def test_two_comparable_pairs_with_min_two_detect_within_pairs() -> None:
    detector = make_detector(min_cohort_size=2)
    host_ms, workload = two_pairs_data()
    verdicts = feed(detector, 3, host_ms, workload)

    stragglers = [v for v in verdicts if v.kind == VERDICT_STRAGGLER]
    assert [v.rank for v in stragglers] == [3]
    assert stragglers[0].facts["reference_ranks"] == [2, 3]


def test_facts_distinguish_raw_and_effective_coverage() -> None:
    detector = make_detector(min_cohort_size=2)
    host_ms, workload = two_pairs_data()
    verdicts = feed(detector, 3, host_ms, workload)

    straggler = [v for v in verdicts if v.kind == VERDICT_STRAGGLER][0]
    facts = straggler.facts
    # Raw: all four ranks reported -> full topological coverage.
    assert facts["cohort_size"] == 4
    assert facts["coverage_ratio"] == 1.0
    # Effective: the rank was judged against {2, 3} only.
    assert facts["comparable_class_size"] == 2
    assert facts["comparable_coverage_ratio"] == 0.5


def test_unfragmented_window_unchanged_under_raised_minimum() -> None:
    """All-equal workloads keep one class of 4: min_cohort_size=4 still
    convicts."""
    detector = make_detector(min_cohort_size=4)
    host_ms = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
    workload = {rank: {"tokens": 1000} for rank in range(4)}
    verdicts = feed(detector, 3, host_ms, workload)

    stragglers = [v for v in verdicts if v.kind == VERDICT_STRAGGLER]
    assert [v.rank for v in stragglers] == [3]
    assert stragglers[0].facts["comparable_class_size"] == 4
