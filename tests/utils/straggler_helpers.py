# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shared fakes for the CPU-only straggler profiler tests."""

from time import perf_counter, time
from weakref import WeakKeyDictionary

from relax.utils.straggler.analysis.detector import DetectorConfig
from relax.utils.straggler.analysis.report import WindowAnalyzer
from relax.utils.straggler.core.collector import StragglerCollector
from relax.utils.straggler.core.stats import FIELD_INDEX, NUM_FIELDS, RankMeta
from relax.utils.straggler.megatron_adapter import phase_timers


class FakeEvent:
    """Stands in for ``torch.cuda.Event``: ``record`` stamps a global counter
    so ``elapsed_time`` is the number of records between the two events."""

    clock = 0.0

    def __init__(self):
        self.stamp = None
        self.done = True

    def record(self):
        FakeEvent.clock += 1.0
        self.stamp = FakeEvent.clock

    def query(self):
        return self.done

    def elapsed_time(self, other):
        return other.stamp - self.stamp


def row(steps=1, **fields):
    """One per-rank window vector; ``fields`` are per-step values summed over
    ``steps``."""
    values = [0.0] * NUM_FIELDS
    values[FIELD_INDEX["num_steps"]] = steps
    for name, value in fields.items():
        values[FIELD_INDEX[name]] = value * steps
    return values


def make_collector(
    world=2,
    interval=1,
    *,
    is_primary=True,
    gather=None,
    gather_objects=None,
    persist=1,
    background=False,
    register_gc=False,
    is_capturing=None,
    analyzer=None,
    trace_sink=None,
    on_report=None,
    **kwargs,
):
    """A CPU collector for rank 0 of ``world`` fake ranks; by default every
    rank reports the same vector as rank 0, and the windows go to a
    ``WindowAnalyzer`` on the collector's clocks."""
    analyzer = analyzer or WindowAnalyzer(
        DetectorConfig(persist_windows=persist),
        trace_sink=trace_sink,
        on_report=on_report,
        clock=kwargs.get("clock", perf_counter),
        wall_clock=kwargs.get("wall_clock", time),
    )
    return StragglerCollector(
        rank_meta=RankMeta(rank=0, dp=0, tp=0, pp=0),
        is_primary=is_primary,
        report_interval=interval,
        analyze=analyzer,
        event_factory=FakeEvent,
        pool_size=kwargs.pop("pool_size", 4),
        gather=gather or (lambda values: [list(values) for _ in range(world)]),
        gather_objects=gather_objects or (lambda obj: meta(world)),
        register_gc_callback=register_gc,
        is_capturing=is_capturing or (lambda: False),
        background=background,
        **kwargs,
    )


_TIMERS = WeakKeyDictionary()


def timers(collector, phase="train"):
    """The ``config.timers`` of ``phase`` feeding ``collector``, one set per
    collector as the profiler holds them."""
    if collector not in _TIMERS:
        _TIMERS[collector] = phase_timers(collector.recorder)
    return _TIMERS[collector][phase]


def bracket(collector, name="forward-compute", phase="train"):
    """One start / stop of a Megatron timer name."""
    timer = timers(collector, phase)(name)
    timer.start()
    timer.stop()


def meta(world, pp_size=1, tp_size=1):
    """``world`` ranks laid out as PP stages of equal size, TP fastest."""
    ranks_per_stage = world // pp_size
    return [
        RankMeta(
            rank=r,
            dp=(r % ranks_per_stage) // tp_size,
            tp=r % tp_size,
            pp=r // ranks_per_stage,
            host=f"h{r // 8}",
            device=r % 8,
        )
        for r in range(world)
    ]
