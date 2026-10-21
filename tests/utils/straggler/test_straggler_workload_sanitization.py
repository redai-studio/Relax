# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Regression: degenerate workload values are degraded evidence, never poison.

A count field is valid only when it is a real number (not a bool), finite and
strictly positive. Zero, negative, NaN/Inf, boolean and missing workloads all
read as absent: the rank's own timing verdict still stands with
``workload_comparable=None`` (degraded), and — critically — a single anomalous
workload must never cost the whole window: the remaining comparable peers are
still diagnosed against each other.

The pairwise comparability introduced with the reference-class fix divides by
peer token counts; without the positivity rule a peer reporting ``tokens=0``
raised ZeroDivisionError inside the judge and the entire window was lost.
"""

from typing import Any, Dict, List, Optional

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import VERDICT_STRAGGLER, StragglerDetector


STAGE = "forward-compute"
COHORT = "topo0:dense:0:0:-1:-1:0:0:0"


class FakeEnvelope:
    def __init__(
        self,
        rank: int,
        host_ms: float,
        workload: Optional[Dict[str, Any]],
        host_start: float,
    ) -> None:
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


def feed(detector: StragglerDetector, windows: int, host_ms: Dict[int, float], workload: Dict[int, Any]) -> List[Any]:
    verdicts: List[Any] = []
    for index in range(windows):
        for rank, host in sorted(host_ms.items()):
            verdicts.extend(detector.observe(FakeEnvelope(rank, host, workload.get(rank), index + 0.1)))
    verdicts.extend(detector.flush())
    return verdicts


BAD_WORKLOADS = [
    {"tokens": 0},
    {"tokens": -5},
    {"tokens": float("nan")},
    {"tokens": float("inf")},
    {"tokens": True},
    {},
    None,
    {"tokens": "1000"},
]


def test_degenerate_workload_rank_is_degraded_not_incomparable() -> None:
    """A rank with a degenerate workload keeps its timing verdict
    (comparable=None)."""
    for bad in BAD_WORKLOADS:
        detector = make_detector()
        host_ms = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
        workload = {0: {"tokens": 1000}, 1: {"tokens": 1000}, 2: {"tokens": 1000}, 3: bad}
        verdicts = feed(detector, 2, host_ms, workload)
        rank3 = [v for v in verdicts if v.rank == 3 and v.kind == VERDICT_STRAGGLER]
        assert rank3, f"rank 3 lost its timing verdict for workload {bad!r}"
        assert rank3[0].facts["workload_comparable"] is None, f"workload {bad!r} must read as degraded"
        assert rank3[0].facts["workload_reported"] is False


def test_degenerate_workload_does_not_lose_the_window() -> None:
    """Pre-fix, a peer reporting tokens=0 raised ZeroDivisionError in the judge
    and the whole window was lost; the comparable peers must still be
    judged."""
    for bad in BAD_WORKLOADS:
        detector = make_detector()
        # rank 0 carries the degenerate workload; ranks 1-3 are comparable and
        # rank 3 is genuinely slow at equal work.
        host_ms = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
        workload = {0: bad, 1: {"tokens": 1000}, 2: {"tokens": 1000}, 3: {"tokens": 1000}}
        verdicts = feed(detector, 2, host_ms, workload)
        stragglers = [v for v in verdicts if v.kind == VERDICT_STRAGGLER]
        assert [v.rank for v in stragglers] == [3], (
            f"comparable peers lost their verdict next to workload {bad!r}: {[(v.rank, v.reason) for v in verdicts]}"
        )


def test_all_peers_degenerate_degrades_without_crashing() -> None:
    """A rank whose every peer is degenerate is judged against the whole
    cohort."""
    detector = make_detector()
    host_ms = {0: 100.0, 1: 100.0, 2: 100.0, 3: 200.0}
    workload = {0: {"tokens": 0}, 1: {"tokens": 0}, 2: {"tokens": 0}, 3: {"tokens": 1000}}
    verdicts = feed(detector, 2, host_ms, workload)
    # Rank 3 has no valid peer workload at all: degraded, judged on timing.
    rank3 = [v for v in verdicts if v.rank == 3 and v.kind == VERDICT_STRAGGLER]
    assert rank3 and rank3[0].facts["workload_comparable"] is None


def test_normal_comparable_peers_unaffected_by_degenerate_outlier() -> None:
    """The canonical case: 4 ranks, one degenerate workload, one true
    straggler."""
    detector = make_detector()
    host_ms = {0: 50.0, 1: 100.0, 2: 100.0, 3: 200.0}
    workload = {0: {"tokens": 0}, 1: {"tokens": 1000}, 2: {"tokens": 1000}, 3: {"tokens": 1000}}
    verdicts = feed(detector, 2, host_ms, workload)
    by_rank = {v.rank: v for v in verdicts if v.kind == VERDICT_STRAGGLER}
    # Rank 3: equal work, twice as slow -> straggler against peers 1-2.
    assert 3 in by_rank
    assert by_rank[3].facts["workload_comparable"] is True
    assert by_rank[3].facts["reference_ranks"] == [1, 2, 3]
