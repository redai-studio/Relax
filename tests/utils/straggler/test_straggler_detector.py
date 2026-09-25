# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for the cross-rank straggler detector.

The tests pin the properties that keep the judgement honest: the reference is
the fastest peer (not the mean), a verdict needs persistence, an incomplete
cohort is reported as ``uncertain`` rather than guessed at, and host stalls are
told apart from genuine device slowdowns.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from relax.utils.straggler.config import StragglerConfig
from relax.utils.straggler.detector import (
    MAX_PENDING_WINDOWS,
    VERDICT_RECOVERED,
    VERDICT_STRAGGLER,
    VERDICT_UNCERTAIN,
    StragglerDetector,
)


COHORT = "0:0:0:0:0"
STAGE = "forward-compute"


@dataclass
class FakeEnvelope:
    """Minimal stand-in for
    :class:`~relax.utils.straggler.observer.TimingEnvelope`."""

    cohort: str
    name: str
    rank: int
    label: str
    host_ms: float
    device_ms: Optional[float]
    host_start: float
    world_size: int = 4


def make_detector(**overrides: Any) -> StragglerDetector:
    settings: Dict[str, Any] = {
        "enabled": True,
        "window_seconds": 1.0,
        "work_tolerance": 0.05,
        "persist_windows": 3,
        "min_cohort_size": 2,
    }
    settings.update(overrides)
    return StragglerDetector(StragglerConfig(**settings))


def feed_window(
    detector: StragglerDetector,
    index: int,
    host_ms: Dict[int, float],
    device_ms: Optional[Dict[int, float]] = None,
    world_size: int = 4,
    stage: str = STAGE,
    cohort: str = COHORT,
) -> List[Any]:
    """Feed one window's worth of envelopes and return the verdicts
    produced."""
    verdicts: List[Any] = []
    for rank, host in sorted(host_ms.items()):
        envelope = FakeEnvelope(
            cohort=cohort,
            name=stage,
            rank=rank,
            label=f"rank{rank}/tp0/pp0",
            host_ms=host,
            device_ms=(device_ms or {}).get(rank),
            host_start=index + 0.1,
            world_size=world_size,
        )
        verdicts.extend(detector.observe(envelope))
    return verdicts


def feed_equal_windows(
    detector: StragglerDetector,
    count: int,
    host_ms: Dict[int, float],
    device_ms: Optional[Dict[int, float]] = None,
    start: int = 0,
) -> List[Any]:
    """Feed ``count`` windows with the same per-rank timings, from
    ``start``."""
    verdicts: List[Any] = []
    for index in range(start, start + count):
        verdicts.extend(feed_window(detector, index, host_ms, device_ms=device_ms))
    return verdicts


def test_identical_ranks_produce_no_verdict() -> None:
    """A/A control: equal work must never be reported as a straggler."""
    detector = make_detector()

    verdicts = feed_equal_windows(detector, 6, {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0})
    verdicts.extend(detector.flush())

    assert [verdict for verdict in verdicts if verdict.kind != VERDICT_UNCERTAIN] == []
    assert detector.stats()["stragglers_reported"] == 0


def test_slow_rank_needs_consecutive_windows() -> None:
    detector = make_detector(persist_windows=3)

    early = feed_equal_windows(detector, 2, {0: 100.0, 1: 100.0, 2: 200.0})
    assert [verdict for verdict in early if verdict.kind == VERDICT_STRAGGLER] == []

    later: List[Any] = []
    later.extend(feed_window(detector, 2, {0: 100.0, 1: 100.0, 2: 200.0}))  # closes window 0
    later.extend(feed_window(detector, 3, {0: 100.0, 1: 100.0, 2: 200.0}))  # closes window 1
    later.extend(feed_window(detector, 4, {0: 100.0, 1: 100.0, 2: 200.0}))  # closes window 2

    stragglers = [verdict for verdict in later if verdict.kind == VERDICT_STRAGGLER]
    assert len(stragglers) == 1
    assert stragglers[0].rank == 2
    assert stragglers[0].consecutive_windows == 3
    assert stragglers[0].cohort_size == 3


def test_reference_is_the_fastest_peer_not_the_mean() -> None:
    """A 2x slow rank must read as +100%, not as +50% against the cohort
    mean."""
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0})
    verdicts.extend(detector.flush())

    stragglers = [verdict for verdict in verdicts if verdict.kind == VERDICT_STRAGGLER]
    assert len(stragglers) == 1
    assert stragglers[0].reference_host_ms == pytest.approx(100.0)
    assert stragglers[0].deviation == pytest.approx(1.0)


def test_recovery_is_reported_once() -> None:
    detector = make_detector(persist_windows=2)

    slow = feed_equal_windows(detector, 6, {0: 100.0, 1: 100.0, 2: 200.0})
    assert [verdict for verdict in slow if verdict.kind == VERDICT_STRAGGLER]

    recovered = feed_equal_windows(detector, 6, {0: 100.0, 1: 100.0, 2: 100.0}, start=6)
    recovered.extend(detector.flush())

    recoveries = [verdict for verdict in recovered if verdict.kind == VERDICT_RECOVERED]
    assert len(recoveries) == 1
    assert recoveries[0].rank == 2
    assert recoveries[0].deviation == pytest.approx(0.0)
    assert detector.active_stragglers() == []


def test_slow_rank_is_reported_only_once_while_it_stays_slow() -> None:
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 8, {0: 100.0, 1: 100.0, 2: 300.0})
    verdicts.extend(detector.flush())

    stragglers = [verdict for verdict in verdicts if verdict.kind == VERDICT_STRAGGLER]
    assert len(stragglers) == 1
    assert detector.stats()["stragglers_reported"] == 1
    assert detector.active_stragglers()[0]["rank"] == 2


def test_host_stall_is_distinguished_from_device_slowdown() -> None:
    host_stall = make_detector(persist_windows=1)
    verdicts = feed_equal_windows(host_stall, 1, {0: 100.0, 1: 100.0, 2: 200.0}, device_ms={0: 10.0, 1: 10.0, 2: 10.0})
    verdicts.extend(host_stall.flush())
    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.host_dominated is True
    assert verdict.reason == "host_stall"

    device_slow = make_detector(persist_windows=1)
    verdicts = feed_equal_windows(
        device_slow, 1, {0: 100.0, 1: 100.0, 2: 200.0}, device_ms={0: 10.0, 1: 10.0, 2: 20.0}
    )
    verdicts.extend(device_slow.flush())
    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.host_dominated is False
    assert verdict.reason == "device_slowdown"


def test_single_rank_run_produces_no_verdict() -> None:
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 4, {0: 100.0, 1: 100.0, 2: 100.0})
    verdicts.extend(detector.flush())

    assert verdicts == []
    assert detector.stats()["single_rank_windows"] == 0  # three ranks were compared


def test_one_rank_alone_is_not_judged() -> None:
    detector = make_detector(persist_windows=1)

    verdicts: List[Any] = []
    for index in range(4):
        verdicts.extend(feed_window(detector, index, {0: 100.0}, world_size=1))
    verdicts.extend(detector.flush())

    assert verdicts == []
    assert detector.stats()["single_rank_windows"] > 0


def test_incomplete_cohort_is_uncertain() -> None:
    detector = make_detector(persist_windows=1)

    verdicts: List[Any] = []
    verdicts.extend(feed_window(detector, 0, {0: 100.0}, world_size=8))
    verdicts.extend(feed_window(detector, 2, {0: 100.0}, world_size=8))  # closes window 0

    uncertain = [verdict for verdict in verdicts if verdict.kind == VERDICT_UNCERTAIN]
    assert len(uncertain) == 1
    assert uncertain[0].reason == "cohort_incomplete"
    assert uncertain[0].cohort_size == 1


def test_cohort_below_min_size_is_uncertain() -> None:
    detector = make_detector(persist_windows=1, min_cohort_size=3)

    verdicts: List[Any] = []
    verdicts.extend(feed_window(detector, 0, {0: 100.0, 1: 100.0}))
    verdicts.extend(feed_window(detector, 2, {0: 100.0, 1: 100.0}))

    uncertain = [verdict for verdict in verdicts if verdict.kind == VERDICT_UNCERTAIN]
    assert len(uncertain) == 2
    assert {verdict.reason for verdict in uncertain} == {"cohort_below_min_size"}


def test_cohorts_are_judged_independently() -> None:
    detector = make_detector(persist_windows=1)

    verdicts = feed_window(detector, 0, {0: 100.0, 1: 100.0, 2: 200.0}, cohort="0:0:0:0:0")
    verdicts.extend(feed_window(detector, 0, {4: 100.0, 5: 100.0, 6: 100.0}, stage="optimizer-inner-step"))
    verdicts.extend(detector.flush())

    stragglers = [verdict for verdict in verdicts if verdict.kind == VERDICT_STRAGGLER]
    assert len(stragglers) == 1
    assert stragglers[0].name == STAGE


def test_window_grace_defers_closing_until_a_newer_window_arrives() -> None:
    detector = make_detector(persist_windows=1)

    feed_window(detector, 0, {0: 100.0, 1: 100.0})
    assert detector.stats()["open_windows"] == 1  # nothing closed yet

    feed_window(detector, 1, {0: 100.0, 1: 100.0})
    assert detector.stats()["open_windows"] == 2  # still nothing closed

    feed_window(detector, 2, {0: 100.0, 1: 100.0})
    assert detector.stats()["windows_closed"] == 1


def test_open_windows_stay_bounded_over_a_long_run() -> None:
    detector = make_detector(persist_windows=1)

    for index in range(60):
        feed_window(detector, index, {0: 100.0, 1: 100.0})

    assert detector.stats()["open_windows"] <= MAX_PENDING_WINDOWS
    assert detector.stats()["windows_closed"] > 50


def test_flush_closes_every_open_window() -> None:
    detector = make_detector(persist_windows=1)

    feed_window(detector, 0, {0: 100.0, 1: 100.0})
    feed_window(detector, 1, {0: 100.0, 1: 100.0})
    verdicts = detector.flush()

    assert detector.stats()["open_windows"] == 0
    assert verdicts == []


def test_drain_verdicts_is_destructive() -> None:
    detector = make_detector(persist_windows=1)

    feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 500.0})
    detector.flush()

    assert len(detector.drain_verdicts()) == 1
    assert detector.drain_verdicts() == []


def test_verdict_serialisation_and_description() -> None:
    detector = make_detector(persist_windows=1)

    feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0})
    detector.flush()
    verdict = detector.drain_verdicts()[0]

    payload = json.loads(verdict.to_json())
    assert payload["rank"] == 2
    assert payload["deviation"] == pytest.approx(1.0)
    assert payload["kind"] == VERDICT_STRAGGLER
    assert "rank2" in verdict.describe()
    assert "\n" not in verdict.to_json()


def test_stats_expose_the_configured_thresholds() -> None:
    detector = make_detector(work_tolerance=0.25, persist_windows=5)

    stats = detector.stats()

    assert stats["work_tolerance"] == pytest.approx(0.25)
    assert stats["persist_windows"] == 5
    assert stats["envelopes"] == 0


def test_malformed_envelope_does_not_raise() -> None:
    detector = make_detector()

    class Broken:
        @property
        def host_start(self) -> float:
            raise RuntimeError("boom")

    assert detector.observe(Broken()) == []
