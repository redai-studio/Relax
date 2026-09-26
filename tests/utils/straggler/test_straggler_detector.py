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
    MAX_ACTIVE_ENTRIES,
    MAX_ALIGNED_RANKS,
    MAX_LABEL_ENTRIES,
    MAX_PAIRS_PER_WINDOW,
    MAX_PENDING_WINDOWS,
    MAX_RANKS_PER_WINDOW,
    MAX_SAMPLES_PER_RANK,
    MAX_STREAK_ENTRIES,
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
    workload: Optional[Dict[str, Any]] = None


def make_detector(**overrides: Any) -> StragglerDetector:
    settings: Dict[str, Any] = {
        "enabled": True,
        "window_seconds": 1.0,
        "warmup_windows": 0,
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
    workload: Optional[Dict[int, Dict[str, Any]]] = None,
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
            workload=(workload or {}).get(rank),
        )
        verdicts.extend(detector.observe(envelope))
    return verdicts


def feed_equal_windows(
    detector: StragglerDetector,
    count: int,
    host_ms: Dict[int, float],
    device_ms: Optional[Dict[int, float]] = None,
    start: int = 0,
    workload: Optional[Dict[int, Dict[str, Any]]] = None,
    stage: str = STAGE,
    cohort: str = COHORT,
) -> List[Any]:
    """Feed ``count`` windows with the same per-rank timings, from
    ``start``."""
    verdicts: List[Any] = []
    for index in range(start, start + count):
        verdicts.extend(
            feed_window(detector, index, host_ms, device_ms=device_ms, workload=workload, stage=stage, cohort=cohort)
        )
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


def test_host_stall_is_distinguished_from_stream_visible_slowness() -> None:
    host_stall = make_detector(persist_windows=1)
    verdicts = feed_equal_windows(host_stall, 1, {0: 100.0, 1: 100.0, 2: 200.0}, device_ms={0: 10.0, 1: 10.0, 2: 10.0})
    verdicts.extend(host_stall.flush())
    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.host_only is True
    assert verdict.reason == "host_only_stall"

    device_slow = make_detector(persist_windows=1)
    verdicts = feed_equal_windows(
        device_slow, 1, {0: 100.0, 1: 100.0, 2: 200.0}, device_ms={0: 10.0, 1: 10.0, 2: 20.0}
    )
    verdicts.extend(device_slow.flush())
    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.host_only is False
    assert verdict.reason == "gpu_stream_stall"


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


def test_incomplete_cohort_is_counted_not_guessed() -> None:
    """A lone rank of a multi-rank cohort is a coverage gap, not a verdict."""
    detector = make_detector(persist_windows=1)

    verdicts: List[Any] = []
    verdicts.extend(feed_window(detector, 0, {0: 100.0}, world_size=8))
    verdicts.extend(feed_window(detector, 2, {0: 100.0}, world_size=8))  # closes window 0

    assert [verdict for verdict in verdicts if verdict.kind == VERDICT_UNCERTAIN] == []
    assert detector.stats()["incomplete_windows"] == 1
    assert detector.stats()["single_rank_windows"] == 0


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


def test_staggered_rank_starts_are_aligned_by_relative_time() -> None:
    """A rank that began profiling seconds later must still be comparable.

    Process startup is staggered in practice (imports, actor placement), but
    the training loop is collective-synchronised afterwards, so every live rank
    reports the *same wall-clock* window. Windows are anchored to the cohort's
    first observation, not to each rank's own, so a rank joining three windows
    in still shares a window with its peers. (The previous form of this test
    fed each rank's own relative window at a different wall time, which is
    exactly the non-contemporaneous comparison the cohort anchor removes.)
    """
    detector = make_detector(persist_windows=1)

    starts = {0: 0.0, 1: 0.5, 2: 3.0}  # rank 2 is three windows "late"
    flat: List[Any] = []
    for tick in range(12):
        wall = tick + 0.1
        for rank, host_ms in ((0, 100.0), (1, 100.0), (2, 400.0)):
            if wall < starts[rank]:
                continue
            flat.extend(
                detector.observe(
                    FakeEnvelope(
                        cohort=COHORT,
                        name=STAGE,
                        rank=rank,
                        label=f"rank{rank}/tp0/pp0",
                        host_ms=host_ms,
                        device_ms=None,
                        host_start=wall,
                        world_size=4,
                    )
                )
            )

    stragglers = [verdict for verdict in flat if verdict.kind == VERDICT_STRAGGLER]
    assert stragglers and {verdict.rank for verdict in stragglers} == {2}
    assert not [verdict for verdict in flat if verdict.kind == VERDICT_UNCERTAIN]
    assert detector.stats()["aligned_ranks"] == 3


def test_warmup_windows_are_never_judged() -> None:
    """A first-window outlier must not be reported as a straggler.

    Measured on real processes: an otherwise identical rank showed +23% in its
    second window purely from lazy CUDA event allocation.
    """
    detector = make_detector(persist_windows=1, warmup_windows=2)

    early: List[Any] = []
    for index in range(3):
        early.extend(feed_window(detector, index, {0: 100.0, 1: 100.0, 2: 400.0}))

    assert [verdict for verdict in early if verdict.kind == VERDICT_STRAGGLER] == []
    assert detector.stats()["warmup_windows_skipped"] >= 1

    later: List[Any] = []
    for index in range(3, 6):
        later.extend(feed_window(detector, index, {0: 100.0, 1: 100.0, 2: 400.0}))

    stragglers = [verdict for verdict in later if verdict.kind == VERDICT_STRAGGLER]
    assert stragglers and {verdict.rank for verdict in stragglers} == {2}
    assert stragglers[0].window_index >= 2


def test_warmup_skip_is_visible_in_the_stats() -> None:
    detector = make_detector(persist_windows=1, warmup_windows=3)

    feed_equal_windows(detector, 5, {0: 100.0, 1: 100.0})

    stats = detector.stats()
    assert stats["warmup_windows_skipped"] == 3
    assert stats["windows_closed"] == 3


# --- measured facts vs inferred causes -------------------------------------


def test_facts_contain_measured_references_and_sample_counts() -> None:
    """A 3-rank window records every number a reviewer needs to re-derive
    it."""
    detector = make_detector(persist_windows=1)

    # Two intervals per rank inside window 0, then a later window to close it.
    for _ in range(2):
        feed_window(detector, 0, {0: 100.0, 1: 110.0, 2: 200.0}, device_ms={0: 10.0, 1: 10.0, 2: 10.0})
    verdicts = feed_window(detector, 3, {0: 100.0, 1: 110.0, 2: 200.0})
    verdicts.extend(detector.flush())

    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER and item.rank == 2)
    facts = verdict.facts
    assert facts["observed_ms"] == pytest.approx(200.0)
    assert facts["peer_fastest_ms"] == pytest.approx(100.0)
    assert facts["peer_median_ms"] == pytest.approx(110.0)
    assert facts["ratio"] == pytest.approx(2.0)
    assert facts["absolute_delta_ms"] == pytest.approx(100.0)
    assert facts["samples_rank"] == 2
    assert facts["samples_peers"] == {0: 2, 1: 2}
    assert facts["samples_peers_min"] == 2
    assert facts["cohort_size"] == 3
    assert facts["cohort_expected"] == 4  # FakeEnvelope reports a world size of 4
    assert facts["coverage_ratio"] == pytest.approx(0.75)
    assert facts["work_tolerance"] == pytest.approx(0.05)
    assert facts["persistence"] == 1
    assert facts["device_ms"] == pytest.approx(10.0)
    assert facts["peer_device_ms"] == pytest.approx(10.0)
    assert facts["device_available"] is True
    assert verdict.measurement_kind == "device_and_host"


def test_verdict_to_dict_is_flat_json_with_facts_and_causes() -> None:
    detector = make_detector(persist_windows=1)

    feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0})
    detector.flush()
    verdict = detector.drain_verdicts()[0]

    payload = json.loads(verdict.to_json())
    assert payload["measurement_kind"] == "host_only"
    assert payload["candidate_causes"] == ["undetermined"]
    assert payload["facts"]["observed_ms"] == pytest.approx(200.0)
    assert payload["facts"]["peer_fastest_ms"] == pytest.approx(100.0)
    assert payload["facts"]["device_available"] is False
    assert json.dumps(payload)  # round-tripped without a custom encoder
    assert set(payload["facts"]) >= {
        "samples_rank",
        "samples_peers",
        "observed_ms",
        "peer_fastest_ms",
        "peer_median_ms",
        "ratio",
        "absolute_delta_ms",
        "work_tolerance",
        "persistence",
        "cohort_size",
        "cohort_expected",
        "coverage_ratio",
        "device_ms",
        "peer_device_ms",
        "device_available",
        "workload_delta",
    }


def test_describe_prints_facts_before_causes() -> None:
    detector = make_detector(persist_windows=1)

    feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0})
    detector.flush()
    verdict = detector.drain_verdicts()[0]

    text = verdict.describe()
    assert "facts(" in text
    assert "reason(measurement)=" in text
    assert text.index("facts(") < text.index("cause:")
    assert "rank2" in text


def test_candidate_causes_default_to_undetermined_without_device_timing() -> None:
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0})
    verdicts.extend(detector.flush())

    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.candidate_causes == ("undetermined",)
    assert verdict.measurement_kind == "host_only"
    assert verdict.facts["device_available"] is False
    assert "cause: undetermined" in verdict.describe()


def test_host_side_delay_possible_when_host_moved_and_device_did_not() -> None:
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0}, device_ms={0: 10.0, 1: 10.0, 2: 10.0})
    verdicts.extend(detector.flush())

    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.candidate_causes == ("host_side_delay_possible",)
    assert verdict.reason == "host_only_stall"
    assert verdict.measurement_kind == "device_and_host"
    assert "candidate causes: host_side_delay_possible" in verdict.describe()


def test_device_or_stream_cause_when_both_moved_without_overclaiming() -> None:
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0}, device_ms={0: 10.0, 1: 10.0, 2: 20.0})
    verdicts.extend(detector.flush())

    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.candidate_causes == ("device_or_stream_visible_delay_possible",)
    for cause in verdict.candidate_causes:
        assert "gpu" not in cause.lower()
        assert "slow" not in cause.lower()


def test_communication_stage_deviation_yields_no_network_or_fault_cause() -> None:
    """A communication-named stage is a label, not evidence."""
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(
        detector,
        1,
        {0: 100.0, 1: 100.0, 2: 400.0},
        device_ms={0: 10.0, 1: 10.0, 2: 40.0},
        stage="nccl-allreduce",
    )
    verdicts.extend(detector.flush())

    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.candidate_causes == ("device_or_stream_visible_delay_possible",)
    rendered = " ".join(verdict.candidate_causes).lower()
    for forbidden in ("network", "fault", "comm", "nccl", "allreduce", "host_side"):
        assert forbidden not in rendered

    # Without device timing the same stage must fall back to undetermined.
    host_only = make_detector(persist_windows=1)
    verdicts = feed_equal_windows(host_only, 1, {0: 100.0, 1: 100.0, 2: 400.0}, stage="nccl-allreduce")
    verdicts.extend(host_only.flush())
    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.candidate_causes == ("undetermined",)


# --- workload: reported, per-rank gated, never silently comparable ----------


def test_workload_delta_is_reported_when_both_sides_report_workloads() -> None:
    detector = make_detector(persist_windows=1)

    workload = {
        0: {"tokens": 100, "sequences": 10},
        1: {"tokens": 100, "sequences": 10},
        2: {"tokens": 220},
    }
    verdicts = feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0}, workload=workload)
    verdicts.extend(detector.flush())

    verdict = next(item for item in verdicts if item.reason == "workload_incomparable")
    # Comparability is driven by TOKENS alone (P0-2): summing tokens + sequences
    # + microbatches mixed units and could cancel a large token gap. Peers report
    # 100 tokens each, so the rank-2 token delta is (220-100)/100.
    assert verdict.facts["tokens_delta"] == pytest.approx(1.2)
    assert verdict.facts["workload_rank_tokens"] == pytest.approx(220.0)
    assert verdict.facts["workload_peer_tokens"] == pytest.approx(100.0)
    assert verdict.facts["workload_delta"] == pytest.approx(1.2)
    assert verdict.facts["workload_comparable"] is False
    assert verdict.facts["workload_delta_beyond_tolerance"] is True
    # RT-08's point survives: the workload difference is visible on the wire.
    # With the comparability gate, an over-tolerance work delta makes the window
    # NOT COMPARABLE, so it is reported as uncertain rather than as a straggler:
    # "rank 2 does more work" is a finding, not slowness.
    assert verdict.kind == VERDICT_UNCERTAIN
    assert detector.stats()["workload_incomparable_windows"] == 1
    assert detector.stats()["stragglers_reported"] == 0


def test_workload_delta_is_none_when_workloads_are_unavailable() -> None:
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0})
    verdicts.extend(detector.flush())

    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.facts["workload_delta"] is None
    assert verdict.facts["workload_peer_median"] is None

    # A rank that reports a workload while its peers do not is still unknown.
    partially_reported = make_detector(persist_windows=1)
    verdicts = feed_equal_windows(partially_reported, 1, {0: 100.0, 1: 100.0, 2: 200.0}, workload={2: {"tokens": 200}})
    verdicts.extend(partially_reported.flush())
    verdict = next(item for item in verdicts if item.kind == VERDICT_STRAGGLER)
    assert verdict.facts["workload_delta"] is None


# --- strict bounds ---------------------------------------------------------


def test_window_structures_stay_bounded_and_evictions_are_counted() -> None:
    # Distinct stage names per window are capped.
    pairs = make_detector(persist_windows=1)
    for pair in range(MAX_PAIRS_PER_WINDOW + 5):
        feed_window(pairs, 0, {0: 100.0, 1: 100.0}, stage=f"stage-{pair}")
    assert pairs.stats()["pair_evictions"] >= 5
    assert len(pairs._windows[0].samples) <= MAX_PAIRS_PER_WINDOW

    # Distinct ranks per (cohort, stage) are capped.
    ranks_detector = make_detector(persist_windows=1)
    many_ranks = {rank: 100.0 for rank in range(MAX_RANKS_PER_WINDOW + 5)}
    feed_window(ranks_detector, 0, many_ranks)
    assert ranks_detector.stats()["rank_evictions"] >= 5
    tracked = next(iter(ranks_detector._windows[0].samples.values()))
    assert len(tracked) <= MAX_RANKS_PER_WINDOW

    # Intervals retained per rank are capped.
    samples = make_detector(persist_windows=1)
    for _ in range(MAX_SAMPLES_PER_RANK + 5):
        feed_window(samples, 0, {0: 100.0, 1: 100.0})
    assert samples.stats()["sample_evictions"] >= 5
    tracked = next(iter(samples._windows[0].samples.values()))
    assert len(tracked[0]) == MAX_SAMPLES_PER_RANK


def test_judgement_state_maps_stay_bounded_and_count_evictions() -> None:
    detector = make_detector(persist_windows=1)

    # Fixed ranks (so their windows advance) with a fresh stage name per window:
    # every window contributes new (cohort, stage, rank) streak and label keys.
    window = 0
    while detector.stats()["streak_evictions"] == 0 and window < 64:
        for pair in range(MAX_PAIRS_PER_WINDOW):
            feed_window(detector, window, {0: 100.0, 1: 200.0}, stage=f"s{window}-{pair}")
        window += 1

    # The cap has to be reached by genuinely filling it: each window contributes
    # MAX_PAIRS_PER_WINDOW fresh (cohort, stage, rank) streak keys, so the keys
    # fed before the first eviction must exceed the documented bound. This
    # assertion fails if the bound is quietly lowered to satisfy the test.
    assert window * MAX_PAIRS_PER_WINDOW > MAX_STREAK_ENTRIES

    stats = detector.stats()
    assert stats["streak_evictions"] > 0
    assert stats["label_evictions"] > 0
    assert stats["streak_entries"] <= MAX_STREAK_ENTRIES
    assert stats["label_entries"] <= MAX_LABEL_ENTRIES


def test_aligned_rank_map_stays_bounded_and_counts_evictions() -> None:
    detector = make_detector(persist_windows=1)

    # A malformed producer could invent a new rank forever: each first sighting
    # adds an alignment epoch, so that map needs its own cap.
    for rank in range(MAX_ALIGNED_RANKS + 5):
        feed_window(detector, 0, {rank: 100.0})

    stats = detector.stats()
    assert stats["epoch_evictions"] >= 5
    assert stats["aligned_ranks"] <= MAX_ALIGNED_RANKS


def test_active_set_and_verdict_deque_stay_bounded() -> None:
    detector = make_detector(persist_windows=1)

    # More slow ranks than the active cap, all in one window with one fast peer.
    ranks = {rank: 400.0 for rank in range(1, MAX_ACTIVE_ENTRIES + 50)}
    ranks[0] = 100.0
    feed_window(detector, 0, ranks)
    detector.flush()

    stats = detector.stats()
    assert stats["active_evictions"] > 0
    assert stats["verdict_evictions"] > 0
    assert stats["active_stragglers"] <= MAX_ACTIVE_ENTRIES
    assert stats["retained_verdicts"] <= stats["caps"]["MAX_VERDICTS"]
    assert stats["streak_entries"] <= MAX_STREAK_ENTRIES


def test_a_stall_is_reported_once_even_when_the_active_cap_evicts_it() -> None:
    """``stragglers_reported`` counts onsets, not re-observations.

    With more distinct slow keys than ``MAX_ACTIVE_ENTRIES``, an evicted entry
    used to be re-added (and re-counted) on the next window even though the
    stall never recovered. Onset is now the exact window in which the
    persistence threshold is first crossed.
    """
    detector = make_detector(persist_windows=1)
    slow_count = MAX_ACTIVE_ENTRIES + 10

    # One pair-window can hold at most MAX_RANKS_PER_WINDOW ranks, so the active
    # cap is exceeded by many slow ranks of one pair rather than many pairs
    # (distinct pairs are capped tighter, at MAX_PAIRS_PER_WINDOW).
    ranks = {0: 100.0}
    ranks.update({rank: 400.0 for rank in range(1, slow_count + 1)})
    for window in range(2):
        feed_window(detector, window, ranks)
    detector.flush()

    stats = detector.stats()
    assert stats["active_evictions"] == 10
    assert stats["stragglers_reported"] == slow_count, "an ongoing stall must not be counted twice"


def test_uncertain_judgements_counts_uncertain_verdicts_once_each() -> None:
    """A sub-floor window emits three uncertain verdicts; the counter says 3.

    ``sub_floor_judgements`` is the sub-floor subset and may equal the total;
    ``uncertain_judgements`` must no longer stay 0 while verdicts are emitted.
    """
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 1, {0: 2.1, 1: 2.2, 2: 0.2, 3: 0.25}, stage="params-all-gather")
    verdicts.extend(detector.flush())

    uncertain = [v for v in verdicts if v.kind == VERDICT_UNCERTAIN]
    assert len(uncertain) == 3
    assert all(v.reason == "below_absolute_floor" for v in uncertain)
    stats = detector.stats()
    assert stats["sub_floor_judgements"] == 3
    assert stats["uncertain_judgements"] == 3


def test_malformed_input_counts_as_an_observe_error_not_a_judgement() -> None:
    detector = make_detector()

    class Broken:
        @property
        def host_start(self) -> float:
            raise RuntimeError("boom")

    assert detector.observe(Broken()) == []

    stats = detector.stats()
    assert stats["observe_errors"] == 1
    assert stats["uncertain_judgements"] == 0


# --- workload absence is degraded evidence ---------------------------------


def test_missing_workload_is_recorded_and_never_reads_as_comparable() -> None:
    """A rank that publishes no workload must not look comparable.

    Detection still runs (a vehicle that never publishes work must still be
    detectable), but every workload fact records the degraded measurement
    instead of a false "checked and equal".
    """
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0})
    verdicts.extend(detector.flush())

    verdict = next(v for v in verdicts if v.kind == VERDICT_STRAGGLER)
    facts = verdict.facts
    assert facts["workload_reported"] is False
    assert facts["workload_delta"] is None
    assert facts["workload_delta_beyond_tolerance"] is None
    assert facts["workload_comparable"] is None
    assert facts["workload_evidence_degraded"] is True
    assert detector.stats()["workload_missing_windows"] == 1


def test_partially_reported_workload_is_degraded_not_comparable() -> None:
    """One rank reporting while its peers do not is still an unknown."""
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 1, {0: 100.0, 1: 100.0, 2: 200.0}, workload={2: {"tokens": 200}})
    verdicts.extend(detector.flush())

    assert verdicts
    assert all(v.facts["workload_evidence_degraded"] is True for v in verdicts)
    assert all(v.facts["workload_comparable"] is None for v in verdicts)
    assert detector.stats()["workload_missing_windows"] == 1


def test_cohort_below_min_size_makes_no_workload_comparability_claim() -> None:
    """The reason and the workload facts must agree in the small-cohort branch.

    With ``min_cohort_size=3`` and two ranks there is no peer median robust
    enough to judge, so the verdicts must not carry ``comparable=False`` and
    ``delta_beyond=True`` alongside ``cohort_below_min_size``.
    """
    detector = make_detector(persist_windows=1, min_cohort_size=3)
    workload = {0: {"tokens": 100.0}, 1: {"tokens": 300.0}}

    verdicts: List[Any] = []
    verdicts.extend(feed_window(detector, 0, {0: 100.0, 1: 100.0}, workload=workload))
    verdicts.extend(feed_window(detector, 2, {0: 100.0, 1: 100.0}, workload=workload))
    verdicts.extend(detector.flush())

    uncertain = [v for v in verdicts if v.reason == "cohort_below_min_size"]
    assert uncertain
    for verdict in uncertain:
        assert verdict.facts["workload_comparable"] is None
        assert verdict.facts["workload_delta_beyond_tolerance"] is None
        assert verdict.facts["workload_evidence_degraded"] is True


def test_sub_millisecond_stage_jitter_is_never_a_straggler() -> None:
    """The measured healthy-run false positive: 0.2 ms -> 2.1 ms is 10x, not a
    straggler.

    Below the absolute floor the gap is host/launch jitter, so the stage must
    be reported ``uncertain``, counted, and never judged.
    """
    detector = make_detector(persist_windows=1)

    verdicts = feed_equal_windows(detector, 4, {0: 2.1, 1: 2.2, 2: 0.2, 3: 0.25}, stage="params-all-gather")
    verdicts.extend(detector.flush())

    assert [v for v in verdicts if v.kind == VERDICT_STRAGGLER] == []
    uncertain = [v for v in verdicts if v.kind == VERDICT_UNCERTAIN]
    assert uncertain, "a sub-floor slow rank must still be reported, as uncertain"
    assert all(v.reason == "below_absolute_floor" for v in uncertain)
    assert detector.stats()["sub_floor_judgements"] > 0
    assert detector.stats()["stragglers_reported"] == 0


def test_large_stage_delays_and_small_relative_gaps_still_fire() -> None:
    """The floor must not silently kill detection on a real compute stage.

    ``forward-compute`` measured ~134 ms, so +20 ms, +5% and +10% are all above
    the floor and must still be reported. Exactly +5% sits on the tolerance
    boundary and deliberately does not fire, because the comparison is strict.
    """
    for observed, expected in ((154.0, True), (141.0, True), (147.4, True), (140.7, False), (134.0, False)):
        detector = make_detector(persist_windows=1)

        verdicts = feed_equal_windows(
            detector, 1, {0: observed, 1: 134.0, 2: 134.0, 3: 134.0}, stage="forward-compute"
        )
        verdicts.extend(detector.flush())

        got = [v for v in verdicts if v.kind == VERDICT_STRAGGLER]
        assert bool(got) is expected, f"observed={observed} ms produced {len(got)} straggler verdicts"
        if expected:
            assert got[0].facts["absolute_delta_ms"] > StragglerConfig(enabled=True).min_stage_ms


def test_the_floor_is_config_driven_not_hardcoded() -> None:
    """A 10x gap on a metadata stage fires only when the floor is disabled."""
    unfenced = make_detector(persist_windows=1, min_stage_ms=0.0)
    verdicts = feed_equal_windows(unfenced, 1, {0: 2.1, 1: 0.2}, stage="params-all-gather")
    verdicts.extend(unfenced.flush())
    assert [v for v in verdicts if v.kind == VERDICT_STRAGGLER]

    fenced = make_detector(persist_windows=1, min_stage_ms=5.0)
    suppressed = feed_equal_windows(fenced, 1, {0: 2.1, 1: 0.2}, stage="params-all-gather")
    suppressed.extend(fenced.flush())
    assert [v for v in suppressed if v.kind == VERDICT_STRAGGLER] == []
    assert fenced.stats()["sub_floor_judgements"] > 0


def test_floor_separates_the_measured_metadata_and_compute_stages() -> None:
    """Report the measured inventory against the chosen floor.

    Inventory measured on the DP4 SFT ON smoke
    (``gpu_campaign/on-smoke-final``): the metadata stages sit 0.06-4.0 ms, the
    real compute stages at 81 ms and 134 ms. The default floor must fall in that
    gap, so it suppresses noise without hiding a genuine compute-stage stall.
    """
    measured = {
        "non-tensor-parallel-grads-all-reduce": 0.060,
        "embedding-grads-all-reduce": 0.069,
        "conditional-embedder-grads-all-reduce": 0.130,
        "params-all-gather": 0.906,
        "optimizer-inner-step": 1.220,
        "optimizer-copy-to-main-grad": 1.981,
        "optimizer-copy-main-to-model-params": 2.414,
        "all-grads-sync": 3.965,
        "backward-compute": 81.214,
        "forward-compute": 133.521,
    }
    floor = StragglerConfig(enabled=True).min_stage_ms

    assert floor == 5.0
    assert [name for name, ms in measured.items() if ms < floor] == list(measured)[:8]
    assert [name for name, ms in measured.items() if ms >= floor] == ["backward-compute", "forward-compute"]


def test_over_tolerance_workload_makes_a_window_not_comparable() -> None:
    """A cohort whose local work differs by >5% must not judge slowness.

    This is the rank3 case from the healthy ON smoke: the detector structurally
    cannot tell "rank 2 does more work" from "rank 2 is slow", so the window is
    reported as incomparable and counted, never as a straggler.
    """
    detector = make_detector(persist_windows=1)
    workload = {0: {"tokens": 100.0}, 1: {"tokens": 100.0}, 2: {"tokens": 220.0}}

    verdicts = feed_equal_windows(detector, 4, {0: 134.0, 1: 134.0, 2: 146.0}, workload=workload)
    verdicts.extend(detector.flush())

    assert [v for v in verdicts if v.kind == VERDICT_STRAGGLER] == []
    assert detector.stats()["stragglers_reported"] == 0
    assert detector.stats()["workload_incomparable_windows"] == 4
    assert all(v.reason == "workload_incomparable" for v in verdicts if v.kind == VERDICT_UNCERTAIN)


def test_within_tolerance_workload_still_judges_normally() -> None:
    """A balanced cohort keeps the original detection behaviour."""
    detector = make_detector(persist_windows=1)
    workload = {0: {"tokens": 100.0}, 1: {"tokens": 100.0}, 2: {"tokens": 104.0}}

    verdicts = feed_equal_windows(detector, 1, {0: 134.0, 1: 134.0, 2: 154.0}, workload=workload)
    verdicts.extend(detector.flush())

    stragglers = [v for v in verdicts if v.kind == VERDICT_STRAGGLER]
    assert len(stragglers) == 1 and stragglers[0].rank == 2
    assert detector.stats()["workload_incomparable_windows"] == 0


def test_over_worked_peer_does_not_withhold_an_equal_work_straggler() -> None:
    """A peer's incomparability must not suppress an unrelated equal-work rank.

    Executed counterexample: four ranks with host times 100/100/100/300 ms and
    tokens 1000/1200/1000/1000. Rank 1 does +20 % tokens, so rank 1 is not
    comparable; rank 3 is genuinely slow with tokens EQUAL to the peer median.
    Under the pair-wide gate rank 1's incomparability withheld rank 3, which was
    reported ``uncertain``/``workload_incomparable`` with its own facts saying
    ``tokens_delta=0.0``. Comparability is a per-rank test, so rank 3 is a
    straggler and its facts agree with its verdict.
    """
    detector = make_detector(persist_windows=1)
    workload = {
        0: {"tokens": 1000.0},
        1: {"tokens": 1200.0},
        2: {"tokens": 1000.0},
        3: {"tokens": 1000.0},
    }

    verdicts = feed_equal_windows(detector, 4, {0: 100.0, 1: 100.0, 2: 100.0, 3: 300.0}, workload=workload)
    verdicts.extend(detector.flush())

    rank3 = [v for v in verdicts if v.rank == 3 and v.kind == VERDICT_STRAGGLER]
    assert rank3, "an equal-work genuine straggler must survive a peer's incomparability"
    facts = rank3[0].facts
    assert facts["tokens_delta"] == pytest.approx(0.0)
    assert facts["workload_delta"] == pytest.approx(0.0)
    assert facts["workload_comparable"] is True
    assert facts["workload_delta_beyond_tolerance"] is False
    # The over-worked peer is counted as incomparable for the pair-window ...
    assert detector.stats()["workload_incomparable_windows"] == 4
    # ... and never leaks its reason into rank 3's verdict.
    assert [v for v in verdicts if v.rank == 3 and v.reason == "workload_incomparable"] == []


def test_each_rank_gets_its_own_workload_reason_and_matching_facts() -> None:
    """The reason and the facts agree per rank, even in a mixed pair-window.

    Two slow ranks: rank 1 did more tokens (incomparable) and rank 3 did the
    peer-median tokens (comparable). Rank 1 is ``uncertain`` with
    ``workload_comparable=False`` and ``workload_delta_beyond_tolerance=True``;
    rank 3 is a straggler with the opposite, self-consistent pair.
    """
    detector = make_detector(persist_windows=1)
    workload = {
        0: {"tokens": 1000.0},
        1: {"tokens": 1200.0},
        2: {"tokens": 1000.0},
        3: {"tokens": 1000.0},
    }

    verdicts = feed_equal_windows(detector, 2, {0: 100.0, 1: 300.0, 2: 100.0, 3: 300.0}, workload=workload)
    verdicts.extend(detector.flush())

    rank1 = [v for v in verdicts if v.rank == 1 and v.kind == VERDICT_UNCERTAIN]
    assert rank1 and all(v.reason == "workload_incomparable" for v in rank1)
    assert rank1[0].facts["workload_delta_beyond_tolerance"] is True
    assert rank1[0].facts["workload_comparable"] is False

    rank3 = [v for v in verdicts if v.rank == 3 and v.kind == VERDICT_STRAGGLER]
    assert rank3
    assert rank3[0].facts["workload_delta_beyond_tolerance"] is False
    assert rank3[0].facts["workload_comparable"] is True


def test_workload_aggregate_is_arrival_order_independent() -> None:
    """The same samples in two arrival orders give identical verdicts and facts.

    Windows are time windows with no step order, so the last-arriving packet is
    not authoritative. Rank 0 is the slow rank and its three token readings are
    1000/1000/2000 (median 1000). When the stale 2000 packet arrives last the
    old last-wins rule flipped rank 0 to
    ``uncertain``/``workload_incomparable``; reversed, it stayed a straggler.
    The per-window aggregate is now the median of the rank's readings, so the
    order cannot change the verdict or any fact.
    """
    host = {0: 300.0, 1: 100.0, 2: 100.0}
    rounds = {
        0: [{"tokens": 1000.0}, {"tokens": 1000.0}, {"tokens": 2000.0}],
        1: [{"tokens": 1000.0}, {"tokens": 1000.0}, {"tokens": 1000.0}],
        2: [{"tokens": 1000.0}, {"tokens": 1000.0}, {"tokens": 1000.0}],
    }

    def run(order: List[int]) -> Dict[int, Any]:
        detector = make_detector(persist_windows=1)
        for step in order:
            feed_window(detector, 0, host, workload={rank: rounds[rank][step] for rank in host})
        return {verdict.rank: verdict for verdict in detector.flush()}

    stale_last = run([0, 1, 2])
    correct_last = run([2, 1, 0])

    assert stale_last[0].kind == correct_last[0].kind == VERDICT_STRAGGLER
    assert stale_last[0].facts == correct_last[0].facts
    assert stale_last[0].facts["workload_delta"] == pytest.approx(0.0)
    assert stale_last[0].facts["workload_comparable"] is True


def test_published_workload_is_per_rank_and_differs_under_unequal_batches() -> None:
    """The producer must publish LOCAL counts, never a rank-invariant
    figure."""
    from relax.utils.straggler.context import (
        publish_step_workload,
        reset_training_context_for_tests,
        set_training_context,
        snapshot,
    )

    reset_training_context_for_tests()
    # Two ranks of one rollout with deliberately unequal local batches.
    publish_step_workload(7, [(1000, 8, 1), (700, 6, 1)])
    publish_step_workload(8, [(5, 1, 1)])

    set_training_context(7, 0, num_steps_per_rollout=2)
    assert snapshot()["tokens"] == 1000 and snapshot()["sequences"] == 8
    set_training_context(7, 1, num_steps_per_rollout=2)
    assert snapshot()["tokens"] == 700 and snapshot()["sequences"] == 6
    # Another rollout's numbers never leak into this one, and an unpublished
    # step reads as absent rather than as the neighbour's work.
    set_training_context(8, 0, num_steps_per_rollout=1)
    assert snapshot()["tokens"] == 5
    set_training_context(7, 5, num_steps_per_rollout=2)
    assert "tokens" not in snapshot()
    reset_training_context_for_tests()
